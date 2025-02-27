import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
import torch.nn.functional as F

# from . import networks
from . import networks7 as networks

import numpy as np
import cv2
import torchvision.utils as vutils

from .BS import HEBackgroundSeparation
from .DiceBCE import DiceBCELoss

class CycleGAN8Model(BaseModel):
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
            parser.add_argument('--mask_loss', type=float, default=0.0, help='weight for mask loss')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'mask_opposite', 'mask_same']
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
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_A = self.netG_A.to(self.device)
        self.netG_B = self.netG_B.to(self.device)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain,
                                            self.gpu_ids, num_classes=13)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain,
                                            self.gpu_ids, num_classes=13)
            self.netD_A = self.netD_A.to(self.device)
            self.netD_B = self.netD_B.to(self.device)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionCE = nn.CrossEntropyLoss().to(self.device)
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.isTrain:
            # Get class labels
            self.real_A_class = input['A_class' if AtoB else 'B_class']
            self.real_B_class = input['B_class' if AtoB else 'A_class']

            # Convert class labels to tensors
            self.real_A_class = torch.tensor(self.real_A_class, dtype=torch.long).to(self.device)
            self.real_B_class = torch.tensor(self.real_B_class, dtype=torch.long).to(self.device)

        center_size = 256
        large_size = self.real_A.shape[2]
        start = (large_size - center_size) // 2
        end = start + center_size
        self.real_A_center = self.real_A[:, :, start:end, start:end]
        self.real_B_center = self.real_B[:, :, start:end, start:end]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, full=True)  # G_A(A)
        self.rec_A_center = self.netG_B(self.fake_B)   # G_B(G_A(A))
        
        self.fake_A = self.netG_B(self.real_B, full=True)  # G_B(B)
        self.rec_B_center = self.netG_A(self.fake_A)   # G_A(G_B(B))

        # # Get the size of the original large image
        # large_size = self.real_A.shape[2]
        # # Define the center size (256x256)
        # center_size = 256
        # # Calculate the starting and ending indices for the center patch
        # start = (large_size - center_size) // 2
        # end = start + center_size
        # # Replace the center region in `x_large` with `x_reconstructed`
        # self.rec_A = self.real_A.clone()
        # self.rec_A[:, :, start:end, start:end] = self.rec_A_center
        # self.rec_B = self.real_B.clone()
        # self.rec_B[:, :, start:end, start:end] = self.rec_B_center

    def backward_D_basic(self, netD, real, real_labels, fake, fake_labels):
        """Calculate RpGAN loss for the discriminator with R1 and R2 gradient penalties.

        Parameters:
            netD (network)      -- the discriminator D
            real_data (tensor)  -- real images
            real_labels (tensor) -- class labels for real images
            fake_data (tensor)  -- images generated by a generator

        Returns the discriminator loss.
        """
        # Gradient penalty coefficient
        gp_lambda = 10.0  # You can adjust this value

        # Enable gradient computation with respect to inputs
        real.requires_grad_(True)
        fake.requires_grad_(True)

        # Get discriminator predictions
        preds_real = netD(real)
        preds_fake = netD(fake) # [batch_size, num_classes]

        # Compute CrossEntropyLoss
        loss_real = self.criterionCE(preds_real, real_labels)
        loss_fake = self.criterionCE(preds_fake, fake_labels)

        # Total loss
        loss_D = loss_real + loss_fake

        # Compute gradient penalty for real data
        real_class_indices = real_labels.view(-1)
        preds_real_per_sample = preds_real[torch.arange(preds_real.size(0)), real_class_indices]

        grad_outputs_real = torch.ones_like(preds_real_per_sample)

        gradients_real = torch.autograd.grad(
            outputs=preds_real_per_sample,
            inputs=real,
            grad_outputs=grad_outputs_real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients_real = gradients_real.view(gradients_real.size(0), -1)
        gradient_norms_real = gradients_real.norm(2, dim=1)
        loss_gp_real = (gradient_norms_real ** 2).mean()

        # Compute gradient penalty for fake data
        fake_class_indices = fake_labels.view(-1)
        preds_fake_per_sample = preds_fake[torch.arange(preds_fake.size(0)), fake_class_indices]

        grad_outputs_fake = torch.ones_like(preds_fake_per_sample)

        gradients_fake = torch.autograd.grad(
            outputs=preds_fake_per_sample,
            inputs=fake,
            grad_outputs=grad_outputs_fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients_fake = gradients_fake.view(gradients_fake.size(0), -1)
        gradient_norms_fake = gradients_fake.norm(2, dim=1)
        loss_gp_fake = (gradient_norms_fake ** 2).mean()

        # Total gradient penalty
        gradient_penalty = gp_lambda * (loss_gp_real + loss_gp_fake) / 2

        # Total loss with gradient penalty
        loss_D_total = loss_D + gradient_penalty

        loss_D_total.backward()
        return loss_D_total

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)

        # Fake labels (class index 6)
        fake_labels = torch.full((fake_B.size(0),), 12, dtype=torch.long, device=self.device)
        # Real labels
        real_labels = self.real_B_class.view(-1)

        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_center, real_labels, fake_B, fake_labels)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)

        # Fake labels (class index 6)
        fake_labels = torch.full((fake_A.size(0),), 12, dtype=torch.long, device=self.device)
        # Real labels
        real_labels = self.real_A_class.view(-1)

        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_center, real_labels, fake_A, fake_labels)

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
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B_center)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B_center) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A_center)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A_center) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Mask loss
        if lambda_mask > 0:
            self.loss_mask_opposite, self.fakeB_mask, self.realB_mask = self.HEBS(self.fake_B, self.real_B_center)
            self.loss_mask_opposite = 0

            # Convert real_A and fake_A to grayscale and add a batch dimension
            real_A_gray = torch.mean(self.real_A_center, dim=1, keepdim=True)  # Grayscale with batch dimension (N, 1, H, W)
            fake_A_gray = torch.mean(self.fake_A, dim=1, keepdim=True)  # Grayscale with batch dimension (N, 1, H, W)

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

        # Generator adversarial loss
        # Get discriminator predictions for fake images
        pred_fake_B = self.netD_A(self.fake_B)
        target_labels = self.real_A_class  # Or choose randomly from real classes
        self.loss_G_A = self.criterionCE(pred_fake_B, target_labels)

        # Similarly for G_B
        pred_fake_A = self.netD_B(self.fake_A)
        target_labels = self.real_B_class  # Or choose randomly from real classes
        self.loss_G_B = self.criterionCE(pred_fake_A, target_labels)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A_center, self.real_A_center) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B_center, self.real_B_center) * lambda_B

        # Total generator loss
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                    self.loss_cycle_A + self.loss_cycle_B + \
                    self.loss_idt_A + self.loss_idt_B + \
                    self.loss_mask_opposite + self.loss_mask_same

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     self.forward()
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