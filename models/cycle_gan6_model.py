import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel

from . import networks
# from . import networks2 as networks

import numpy as np
import cv2
import torchvision.utils as vutils
# from skimage.color import rgb2hed

from .BS import HEBackgroundSeparation
from .DiceBCE import DiceBCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
from util.losses import MS_SSIM_Loss
from torch.nn import CrossEntropyLoss

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=0.01):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        #h_tv = torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]).sum()
        #w_tv = torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]).sum()
        #return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        return self.TVLoss_weight*2*torch.sqrt(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class CycleGAN6Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--mask_loss', type=float, default=1.0, help='weight for mask loss')
            parser.add_argument('--tv_loss', type=float, default=0.0, help='weight for tv loss')
            parser.add_argument('--ssim_loss', type=float, default=0.0, help='weight for ssim loss')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'ssim', 'tv', 'mask_opposite', 'mask_same']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A_center', 'fake_B', 'rec_A_center']
        visual_names_B = ['real_B_center', 'fake_A', 'rec_B_center']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.loss_names += ['idt_A', 'idt_B']
        if self.isTrain and self.opt.mask_loss > 0.0:
            visual_names_A.append('fakeB_mask')
            visual_names_A.append('realA_mask')
            visual_names_B.append('realB_mask')
            visual_names_B.append('fakeA_mask')
            # self.loss_names.append += ['mask_opposite', 'mask_same']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device1)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids).to(self.device2)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain,
                                            self.gpu_ids, num_classes=7).to(self.device1)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain,
                                            self.gpu_ids, num_classes=7).to(self.device2)

        

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device1)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMask = DiceBCELoss()
            self.criterion_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
            self.tvloss = TVLoss(TVLoss_weight=0.01).to(self.device1)
            self.criterionCE = CrossEntropyLoss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(
                list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.HEBS = HEBackgroundSeparation()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device1)
        self.real_B = input['B' if AtoB else 'A'].to(self.device2)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # Get class labels
        self.real_A_class = input['A_class' if AtoB else 'B_class']
        self.real_B_class = input['B_class' if AtoB else 'A_class']

        # Convert class labels to tensors
        self.real_A_class = torch.tensor(self.real_A_class, dtype=torch.long).to(self.device1)
        self.real_B_class = torch.tensor(self.real_B_class, dtype=torch.long).to(self.device2)

        center_size = 256
        large_size = self.real_A.shape[2]
        start = (large_size - center_size) // 2
        end = start + center_size
        self.real_A_center = self.real_A[:, :, start:end, start:end]
        self.real_B_center = self.real_B[:, :, start:end, start:end]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, full=True)  # G_A(A)
        self.fake_B_on_device2 = self.fake_B.to(self.device2)
        self.rec_A_center = self.netG_B(self.fake_B_on_device2)   # G_B(G_A(A))
        self.rec_A_center = self.rec_A_center.to(self.device1)
        
        self.fake_A = self.netG_B(self.real_B, full=True)  # G_B(B)
        self.fake_A_on_device1 = self.fake_A.to(self.device1)
        self.rec_B_center = self.netG_A(self.fake_A_on_device1)   # G_A(G_B(B))
        self.rec_B_center = self.rec_B_center.to(self.device2)

        # Get the size of the original large image
        large_size = self.real_A.shape[2]
        # Define the center size (256x256)
        center_size = 256
        # Calculate the starting and ending indices for the center patch
        start = (large_size - center_size) // 2
        end = start + center_size
        # Replace the center region in `x_large` with `x_reconstructed`
        self.rec_A = self.real_A.clone()
        self.rec_A[:, :, start:end, start:end] = self.rec_A_center
        self.rec_B = self.real_B.clone()
        self.rec_B[:, :, start:end, start:end] = self.rec_B_center

    def backward_D_basic(self, netD, real, real_labels, fake, fake_labels):
        """Calculate loss for the discriminator with cross-entropy.

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            real_labels (tensor) -- class labels for real images
            fake (tensor array) -- images generated by a generator
            fake_labels (tensor) -- class labels for fake images (should be class index 6)

        Returns the discriminator loss.
        """
        # Concatenate real and fake images
        inputs = torch.cat((real, fake), dim=0)
        # Get discriminator predictions
        preds = netD(inputs)
        # Flatten predictions and labels
        preds = preds.view(preds.size(0), preds.size(1), -1)
        preds = preds.mean(2)  # Average over spatial dimensions
        # Concatenate labels
        labels = torch.cat((real_labels, fake_labels), dim=0)
        # Compute loss
        loss_D = self.criterionCE(preds, labels)
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.real_B_on_device1 = self.real_B_center.to(self.device1)
        fake_B = self.fake_B_pool.query(self.fake_B)
        # Fake labels (class index 6)
        fake_labels = torch.full((fake_B.size(0),), 6, dtype=torch.long, device=self.device1)
        # Real labels
        real_labels = self.real_B_class
        # Ensure labels are tensors of correct size
        real_labels = real_labels.view(-1).to(self.device1)
        fake_labels = fake_labels.view(-1).to(self.device1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_on_device1 , real_labels, fake_B, fake_labels)

    def backward_D_B(self):
        self.real_A_on_device2 = self.real_A_center.to(self.device2)
        fake_A = self.fake_A_pool.query(self.fake_A)
        # Fake labels (class index 6)
        fake_labels = torch.full((fake_A.size(0),), 6, dtype=torch.long, device=self.device2)
        # Real labels
        real_labels = self.real_A_class
        # Ensure labels are tensors of correct size
        real_labels = real_labels.view(-1).to(self.device2)
        fake_labels = fake_labels.view(-1).to(self.device2)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_on_device2, real_labels, fake_A, fake_labels)


    def threshold_light_sheet(self, image_tensor, threshold=0.05):
        """
        Apply thresholding to a grayscale light-sheet image to separate the background.
        
        Args:
            image_tensor (torch.Tensor): Grayscale light-sheet image as a PyTorch tensor of shape (H, W).
            threshold (float): Threshold value for separating background.
            
        Returns:
            mask (torch.Tensor): Binary mask where the background is white and the non-background regions are black.
        """
        # Apply threshold directly on the tensor to identify bright parts
        mask = (image_tensor > threshold).float()
        
        return mask
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_mask = self.opt.mask_loss
        lambda_tv = self.opt.tv_loss
        lambda_ssim = self.opt.ssim_loss
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.real_B_center_on_device1 = self.real_B_center.to(self.device1)
            self.idt_A = self.netG_A(self.real_B_center_on_device1)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B_center_on_device1) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.real_A_center_on_device2 = self.real_A_center.to(self.device2)
            self.idt_B = self.netG_B(self.real_A_center_on_device2)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A_center_on_device2) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Mask loss
        if lambda_mask > 0:
            self.real_B_center_on_device1 = self.real_B_center.to(self.device1)
            self.loss_mask_opposite, self.fakeB_mask, self.realB_mask = self.HEBS(self.fake_B, self.real_B_center_on_device1)
            self.loss_mask_opposite = 0

            # Convert real_A and fake_A to grayscale and add a batch dimension
            real_A_gray = torch.mean(self.real_A_center, dim=1, keepdim=True)  # Grayscale with batch dimension (N, 1, H, W)
            self.fake_A_on_device1 = self.fake_A.to(self.device1)
            fake_A_gray = torch.mean(self.fake_A_on_device1, dim=1, keepdim=True)  # Grayscale with batch dimension (N, 1, H, W)

            # Apply thresholding
            self.realA_mask = self.threshold_light_sheet(real_A_gray)
            self.fakeA_mask = self.threshold_light_sheet(fake_A_gray)
            # Apply adaptive thresholding
            # self.realA_mask = self.adaptive_threshold_light_sheet(real_A_gray)
            # self.fakeA_mask = self.adaptive_threshold_light_sheet(fake_A_gray)

            # Calculate mask loss
            self.loss_mask_same = self.criterionMask(self.realA_mask, self.fakeB_mask) * lambda_mask +\
                                  self.criterionMask(self.fakeA_mask, self.realB_mask) * lambda_mask
            self.loss_mask_same = self.loss_mask_same / 2
        else:
            self.loss_mask_opposite = 0
            self.loss_mask_same = 0

        if lambda_ssim > 0:
            self.loss_ssim = ((self.criterion_ssim(self.fake_B, self.real_B_center) + self.criterion_ssim(self.fake_A, self.real_A_center)) / 2) * 0.05
        else:
            self.loss_ssim = 0

        
        if lambda_tv > 0:
            self.loss_tv_A = self.tvloss(self.fake_B)
            self.loss_tv_B = self.tvloss(self.fake_A)
            self.loss_tv_rec_A = self.tvloss(self.rec_A)
            self.loss_tv_rec_B = self.tvloss(self.rec_B)
            self.loss_tv = (self.loss_tv_A + self.loss_tv_B + self.loss_tv_rec_A + self.loss_tv_rec_B) * lambda_tv
        else:
            self.loss_tv = 0

        # Generator adversarial loss
        # Get discriminator predictions for fake images
        pred_fake_B = self.netD_A(self.fake_B)
        pred_fake_B = pred_fake_B.view(pred_fake_B.size(0), pred_fake_B.size(1), -1)
        pred_fake_B = pred_fake_B.mean(2)
        # We want the generator to produce images that the discriminator classifies as real classes (0 to 5)
        # For simplicity, we can use the class labels of the real images as targets
        target_labels = self.real_A_class  # Or choose randomly from real classes
        self.loss_G_A = self.criterionCE(pred_fake_B, target_labels)

        # Similarly for G_B
        pred_fake_A = self.netD_B(self.fake_A)
        pred_fake_A = pred_fake_A.view(pred_fake_A.size(0), pred_fake_A.size(1), -1)
        pred_fake_A = pred_fake_A.mean(2)
        target_labels = self.real_B_class  # Or choose randomly from real classes
        self.loss_G_B = self.criterionCE(pred_fake_A, target_labels)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.real_A_center_on_device1 = self.real_A_center.to(self.device1)
        self.loss_cycle_A = self.criterionCycle(self.rec_A_center, self.real_A_center_on_device1) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.real_B_center_on_device2 = self.real_B_center.to(self.device2)
        self.loss_cycle_B = self.criterionCycle(self.rec_B_center, self.real_B_center_on_device2) * lambda_B
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A.to(self.device1) + self.loss_G_B.to(self.device1) +\
                      self.loss_cycle_A.to(self.device1) + self.loss_cycle_B.to(self.device1) +\
                      self.loss_idt_A + self.loss_idt_B +\
                      self.loss_mask_opposite + self.loss_mask_same.to(self.device1) +\
                      self.loss_ssim + self.loss_tv

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights