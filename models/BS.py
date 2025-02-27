import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import math

class HEBackgroundSeparation(nn.Module):
    def __init__(self):
        super(HEBackgroundSeparation, self).__init__()

        # Define the color deconvolution matrix for H&E staining
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], 
                                          [0.07, 0.99, 0.11], 
                                          [0.27, 0.57, 0.78]]).cuda()

        self.coeffs = torch.tensor([0.2125, 0.7154, 0.0721]).view(3,1).cuda()        
        self.hed_from_rgb = torch.linalg.inv(self.rgb_from_hed).cuda()
        self.log_adjust = torch.log(torch.tensor(1e-6)).cuda()

        # focal FOD , alpha 
        self.alpha = 1.8 
        self.adjust_Calibration = torch.tensor(10**(-(math.e)**(1/self.alpha))).cuda()
        
        # Set a threshold to identify and zero out FOD values that are too low,/
        # thereby reducing their impact on computing the tumor expression level inference.
        self.thresh_FOD = 0.15
        # thresh_FOD for getting pseudo mask
        self.thresh_mask = 0.68

        self.log_adjust = torch.log(torch.tensor(1e-6)).cuda()
        
        self.mse_loss = nn.MSELoss().cuda()
        self.mse_loss_2 = nn.MSELoss(reduce = False).cuda()
        
    def forward(self, inputs, targets, mode='h'):
        """
        Generate the background-only image by separating and subtracting the H&E channels.

        Args:
            image (torch.Tensor): Input RGB image tensor of shape (C, H, W).

        Returns:
            background_only (torch.Tensor): Background-only image tensor of shape (C, H, W).
        """
        inputs_reshape = inputs.permute(0, 2, 3, 1)
        targets_reshape = targets.permute(0, 2, 3, 1)

        inputs_OD,input_block,input_histo,input_mask = self.compute_OD(inputs_reshape, mode)
        targets_OD,target_block,target_histo,target_mask = self.compute_OD(targets_reshape, mode)
        
        loss_MLPA = 0.0
        
        MLPA_avg = self.mse_loss_2(inputs_OD,targets_OD)/(inputs.shape[2]*inputs.shape[3])**2
        MLPA_histo = (((input_histo/(inputs.shape[2]*inputs.shape[3])-target_histo/(inputs.shape[2]*inputs.shape[3]))**2).sum(1))/inputs.shape[0] 
        MLPA_block = self.mse_loss((input_block/(inputs.shape[2]*inputs.shape[3]/16)),(target_block/(inputs.shape[2]*inputs.shape[3]/16)))

        loss_MLPA += torch.sum(torch.where((inputs_OD - targets_OD >= targets_OD * -0.4) & (inputs_OD - targets_OD <= targets_OD * 0.4), MLPA_histo, MLPA_avg + MLPA_histo))
        loss_MLPA  += MLPA_block
        
        return loss_MLPA, input_mask.unsqueeze(0), target_mask.unsqueeze(0)

    def compute_OD(self, image, mode):
        assert image.shape[-1] == 3

         # Focal Optical Density map
        ihc_hed = self.separate_stains(image, self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:,:, :, 0])
        # select DAB stain OD and generate RGB image only with DAB OD

        if mode == 'h':
            # For H only
            ihc_he = self.combine_stains(torch.stack((ihc_hed[:, :, :, 0], null, null), axis=-1),self.rgb_from_hed)
        elif mode == 'e':
            # For E only
            ihc_he = self.combine_stains(torch.stack((null, ihc_hed[:, :, :, 1], null), axis=-1),self.rgb_from_hed)
        elif mode =='he':
            # For HE
            ihc_he = self.combine_stains(torch.stack((ihc_hed[:, :, :, 0], ihc_hed[:, :, :, 1], null), axis=-1),self.rgb_from_hed)
        elif mode == 'd':
            # For D only
            ihc_he = self.combine_stains(torch.stack((null, null, ihc_hed[:, :, :, 2]), axis=-1),self.rgb_from_hed)
        else:
            # For all
            ihc_he = self.combine_stains(ihc_hed,self.rgb_from_hed)

        # turn into gray
        grey_he = self.rgb2gray(ihc_he)
        grey_he[grey_he<0.0] = torch.tensor(0.0).cuda()
        grey_he[grey_he>1.0] = torch.tensor(1.0).cuda()
        # get FOD in later process
        FOD = torch.log10(1/(grey_he+self.adjust_Calibration))

        FOD[FOD<0] = torch.tensor(0.0).cuda()
        
        FOD = FOD**self.alpha
        # Set a threshold to identify and zero out FOD values that are too low
        FOD_relu = torch.where(FOD < self.thresh_FOD, torch.tensor(0.0).cuda(), FOD)

        # mask_OD generate a pseudo mask for IHC image(real or fake)
        mask_OD = torch.where(FOD < self.thresh_mask, torch.tensor(0.0).cuda(), FOD)
        mask_OD = mask_OD.squeeze(-1).detach()
        mask_OD[mask_OD > 0] = torch.tensor(1.0)
        
        # flattened_img = FOD_relu.squeeze(-1).flatten(1,2)
        flattened_img_2 = FOD.squeeze(-1).flatten(1,2)
        
        # avg
        avg = torch.sum(FOD_relu,dim=(1,2,3))

        # block 
        num_blocks = 16
        tensor_blocks = FOD_relu.squeeze(-1).unfold(1, image.shape[1]//int(math.sqrt(num_blocks)), image.shape[1]//int(math.sqrt(num_blocks)))\
            .unfold(2, image.shape[2]//int(math.sqrt(num_blocks)), image.shape[2]//int(math.sqrt(num_blocks)))
        block = tensor_blocks.sum(dim=(3, 4))

        # histo
        num_bins = 20
        histo = self.calculate_histo_sums(flattened_img_2, num_bins,0.0,math.e)
        
        return avg, block, histo, mask_OD

    def separate_stains(self, rgb, conv_matrix):
        """
        Separate the RGB image into stain components using the provided deconvolution matrix.

        Args:
            rgb (torch.Tensor): RGB image tensor of shape (N, H, W, C).
            conv_matrix (torch.Tensor): Deconvolution matrix.

        Returns:
            stains (torch.Tensor): Stain components tensor of shape (N, H, W, C).
        """
        rgb = torch.maximum(rgb, torch.tensor(1e-6).cuda())  # Avoiding log artifacts

        stains = torch.matmul(torch.log(rgb) / self.log_adjust, conv_matrix)
        stains = torch.maximum(stains, torch.tensor(0).cuda())

        return stains

        return stains

    def combine_stains(self, stains, conv_matrix):
        """
        Reconstruct the RGB image from stain components using the provided convolution matrix.

        Args:
            stains (torch.Tensor): Stain components tensor of shape (N, H, W, C).
            conv_matrix (torch.Tensor): Convolution matrix.

        Returns:
            rgb (torch.Tensor): Reconstructed RGB image tensor of shape (N, H, W, C).
        """
        log_rgb = -torch.matmul((stains * -self.log_adjust), conv_matrix)
        rgb = torch.exp(log_rgb)

        return torch.clamp(rgb, min=0, max=1)

    def rgb2gray(self,rgb, *, channel_axis=-1):

        return torch.matmul(rgb ,self.coeffs)

    def calculate_histo_sums(self,input, num_histos, min_val, max_val):
        # Calculate the width of each histo.
        features = input.clone()
        bucket_width = (max_val - min_val) / num_histos
        
        # Normalize the feature values to the range [0, num_histos-1]
        normalized_features = (features - min_val) / bucket_width
        
        # Map the feature values to their corresponding histo indices
        histo_indices = (normalized_features.clamp(0, num_histos-1)).long()
        
        # Initialize a tensor to store the sum of feature values in each histo
        batch_sums = torch.zeros((features.shape[0], num_histos)).cuda()
        
        # Calculate the sum of feature values within each histo
        for i in range(features.shape[0]):
            for j in range(num_histos):
                # Find the indices of feature values that fall into the current histo
                indices_in_histo = (histo_indices[i] == j).nonzero()
                if indices_in_histo.numel() > 0:
                    # Sum up the feature values that fall into the current histo.
                    batch_sums[i, j] = torch.sum(features[i, indices_in_histo])
        
        return batch_sums
    
    def weighted_mse_loss(self,input,target,weights = None):
        assert input.size() == target.size()
        size = input.size()
        if weights == None:
            weights = torch.ones(size = size[0])
        
        se = ((input - target)**2)
        
        return (se*weights).mean()