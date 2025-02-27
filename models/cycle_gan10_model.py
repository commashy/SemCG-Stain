import itertools
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks8 as networks
from transformers import CLIPProcessor, CLIPModel
import random

class CycleGAN10Model(BaseModel):
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
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping.')
            parser.add_argument('--lambda_emb_A', type=float, default=10.0, help='weight for embedding loss for domain A')
            parser.add_argument('--lambda_emb_B', type=float, default=10.0, help='weight for embedding loss for domain B')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'emb_A', 'emb_B', 'emb_consistency']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A_center', 'fake_B_center', 'rec_A_center']
        visual_names_B = ['real_B_center', 'fake_A_center', 'rec_B_center']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.loss_names += ['idt_A', 'idt_B']

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

        self.netPLIP_A = CLIPModel.from_pretrained("vinid/plip").to(self.device)
        self.netPLIP_B = CLIPModel.from_pretrained("vinid/plip").to(self.device)
        self.processor_A = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)
        self.processor_B = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)

        # Freeze netPLIP_B parameters
        for param in self.netPLIP_B.parameters():
            param.requires_grad = False

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
            self.criterionEmb = nn.CosineSimilarity(dim=1, eps=1e-08).to(self.device)
            self.lambda_emb_A = opt.lambda_emb_A
            self.lambda_emb_B = opt.lambda_emb_B

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netPLIP_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def extract_patches(self, images, patch_size):
        # images: (batch_size, channels, height, width)
        batch_size, channels, height, width = images.size()
        assert height % patch_size == 0 and width % patch_size == 0, "Image dimensions must be divisible by patch size"

        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: (batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)

        patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
        # patches: (batch_size, channels, num_patches, patch_size, patch_size)

        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # patches: (batch_size, num_patches, channels, patch_size, patch_size)

        patches = patches.view(-1, channels, patch_size, patch_size)
        # patches: (batch_size * num_patches, channels, patch_size, patch_size)

        return patches, num_patches_h, num_patches_w

    def reconstruct_from_patches(self, patches, num_patches_h, num_patches_w, batch_size, channels, patch_size):
        # patches: (batch_size * num_patches, channels, patch_size, patch_size)
        num_patches = num_patches_h * num_patches_w

        patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
        # patches: (batch_size, num_patches, channels, patch_size, patch_size)

        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # patches: (batch_size, channels, num_patches, patch_size, patch_size)

        patches = patches.view(batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        # patches: (batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)

        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        # patches: (batch_size, channels, num_patches_h * patch_size, num_patches_w * patch_size)

        images = patches.view(batch_size, channels, num_patches_h * patch_size, num_patches_w * patch_size)
        return images

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

    def unnormalize_image(self, tensor):
        # Convert from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        # Clamp values to ensure they are within [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Generate embeddings for real images
        with torch.no_grad():
            images_real_A = self.unnormalize_image(self.real_A)
            images_real_B = self.unnormalize_image(self.real_B)

            image_inputs_real_A = self.processor_A(images=[img.cpu() for img in images_real_A], return_tensors="pt").to(self.device)
            image_inputs_real_B = self.processor_B(images=[img.cpu() for img in images_real_B], return_tensors="pt").to(self.device)

            self.embeddings_real_A = self.netPLIP_A.get_image_features(**image_inputs_real_A)
            self.embeddings_real_B = self.netPLIP_B.get_image_features(**image_inputs_real_B)

            if self.isTrain:
                images_real_A_center = self.unnormalize_image(self.real_A_center)
                images_real_B_center = self.unnormalize_image(self.real_B_center)

                image_inputs_real_A_center = self.processor_A(images=[img.cpu() for img in images_real_A_center], return_tensors="pt").to(self.device)
                image_inputs_real_B_center = self.processor_B(images=[img.cpu() for img in images_real_B_center], return_tensors="pt").to(self.device)

                self.embeddings_real_B_consistency = self.netPLIP_A.get_image_features(**image_inputs_real_B)

                self.embeddings_real_A_center = self.netPLIP_A.get_image_features(**image_inputs_real_A_center)
                self.embeddings_real_B_center = self.netPLIP_B.get_image_features(**image_inputs_real_B_center)

        if self.isTrain:
            # Split real images into patches
            patch_size = 256
            patches_real_A, num_patches_h, num_patches_w = self.extract_patches(self.real_A, patch_size)
            patches_real_B, _, _ = self.extract_patches(self.real_B, patch_size)

            # Process patches through netG_A and netG_B
            batch_size_patches = patches_real_A.size(0)
            batch_size_processing = 3  # Max batch size you can handle

            batch_size = self.real_A.size(0)
            num_patches_per_image = patches_real_A.size(0) // batch_size
            embedding_dim = self.embeddings_real_A.size(1)

            # Expand embeddings_real_A to match patches_real_A
            embeddings_real_A_expanded = self.embeddings_real_A.unsqueeze(1).expand(-1, num_patches_per_image, -1).reshape(-1, embedding_dim)
            embeddings_real_B_expanded = self.embeddings_real_B.unsqueeze(1).expand(-1, num_patches_per_image, -1).reshape(-1, embedding_dim)

            fake_B_patches = []
            fake_A_patches = []

            for i in range(0, batch_size_patches, batch_size_processing):
                end = min(i + batch_size_processing, batch_size_patches)
                input_patches_A = patches_real_A[i:end]
                input_patches_B = patches_real_B[i:end]

                # Get corresponding embeddings
                embeddings_patches_A = embeddings_real_A_expanded[i:end]
                embeddings_patches_B = embeddings_real_B_expanded[i:end]

                # G_A(A)
                fake_B_patch = self.netG_A(input_patches_A, embeddings_patches_A)
                fake_B_patches.append(fake_B_patch)

                # G_B(B)
                fake_A_patch = self.netG_B(input_patches_B, embeddings_patches_B)
                fake_A_patches.append(fake_A_patch)

            # Concatenate all the patches
            fake_B_patches = torch.cat(fake_B_patches, dim=0)
            fake_A_patches = torch.cat(fake_A_patches, dim=0)

            # Reconstruct fake images from patches
            self.fake_B = self.reconstruct_from_patches(fake_B_patches, num_patches_h, num_patches_w, self.real_A.size(0), self.real_A.size(1), patch_size)
            self.fake_A = self.reconstruct_from_patches(fake_A_patches, num_patches_h, num_patches_w, self.real_B.size(0), self.real_B.size(1), patch_size)

            center_size = 256
            large_size = self.real_A.shape[2]
            start = (large_size - center_size) // 2
            end = start + center_size
            self.fake_A_center = self.fake_A[:, :, start:end, start:end]
            self.fake_B_center = self.fake_B[:, :, start:end, start:end]

            # Generate embeddings for fake images
            images_fake_A = self.unnormalize_image(self.fake_A)
            images_fake_B = self.unnormalize_image(self.fake_B)
            images_fake_A_center = self.unnormalize_image(self.fake_A_center)
            images_fake_B_center = self.unnormalize_image(self.fake_B_center)

            image_inputs_fake_A = self.processor_A(images=[img.cpu() for img in images_fake_A], return_tensors="pt").to(self.device)
            image_inputs_fake_B = self.processor_B(images=[img.cpu() for img in images_fake_B], return_tensors="pt").to(self.device)
            image_inputs_fake_A_center = self.processor_A(images=[img.cpu() for img in images_fake_A_center], return_tensors="pt").to(self.device)
            image_inputs_fake_B_center = self.processor_B(images=[img.cpu() for img in images_fake_B_center], return_tensors="pt").to(self.device)

            self.embeddings_fake_A = self.netPLIP_A.get_image_features(**image_inputs_fake_A)
            self.embeddings_fake_B = self.netPLIP_B.get_image_features(**image_inputs_fake_B)
            self.embeddings_fake_B_consistency = self.netPLIP_A.get_image_features(**image_inputs_fake_B)

            self.embeddings_fake_A_center = self.netPLIP_A.get_image_features(**image_inputs_fake_A_center)
            self.embeddings_fake_B_center = self.netPLIP_B.get_image_features(**image_inputs_fake_B_center)

            # Reconsturct only the center region
            self.rec_A_center = self.netG_B(self.fake_B_center, self.embeddings_fake_B)
            self.rec_B_center = self.netG_A(self.fake_A_center, self.embeddings_fake_A)
        else:
            # G_A(A)
            self.fake_B = self.netG_A(self.real_A, self.embeddings_real_A)
            # G_B(B)
            self.fake_A = self.netG_B(self.real_B, self.embeddings_real_B)
        
    def normalize_embeddings(self, embeddings):
        norms = embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-8  # Add epsilon to avoid division by zero
        return embeddings / norms

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
        fake_B = self.fake_B_pool.query(self.fake_B.detach())

        # Fake labels (class index 12)
        fake_labels = torch.full((fake_B.size(0),), 12, dtype=torch.long, device=self.device)
        # Real labels
        real_labels = self.real_B_class.view(-1)

        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, real_labels, fake_B, fake_labels)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A.detach())

        # Fake labels (class index 12)
        fake_labels = torch.full((fake_A.size(0),), 12, dtype=torch.long, device=self.device)
        # Real labels
        real_labels = self.real_A_class.view(-1)

        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, real_labels, fake_A, fake_labels)
    
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
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
        
        # Generator adversarial loss
        # Get discriminator predictions for fake images
        pred_fake_B = self.netD_A(self.fake_B)
        target_labels_B = self.real_A_class  # Or choose randomly from real classes
        self.loss_G_A = self.criterionCE(pred_fake_B, target_labels_B)

        # Similarly for G_B
        pred_fake_A = self.netD_B(self.fake_A)
        target_labels_A = self.real_B_class  # Or choose randomly from real classes
        self.loss_G_B = self.criterionCE(pred_fake_A, target_labels_A)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A_center, self.real_A_center) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B_center, self.real_B_center) * lambda_B

        # Embedding loss
        if lambda_A > 0:
            self.loss_emb_A_large = self.criterionEmb(self.embeddings_real_A, self.embeddings_fake_B).mean()
            self.loss_emb_B_large = self.criterionEmb(self.embeddings_real_B, self.embeddings_fake_A).mean()

            self.loss_emb_A_center = self.criterionEmb(self.embeddings_real_A_center, self.embeddings_fake_B_center).mean()
            self.loss_emb_B_center = self.criterionEmb(self.embeddings_real_B_center, self.embeddings_fake_A_center).mean()

            self.loss_emb_consistency_1 = self.criterionEmb(self.embeddings_real_B_consistency, self.embeddings_real_B).mean()
            self.loss_emb_consistency_2 = self.criterionEmb(self.embeddings_fake_B_consistency, self.embeddings_fake_B).mean()

            # Clamp similarities to [-1, 1] to handle numerical errors
            self.loss_emb_A_large = torch.clamp(self.loss_emb_A_large, -1, 1)
            self.loss_emb_B_large = torch.clamp(self.loss_emb_B_large, -1, 1)

            self.loss_emb_A_center = torch.clamp(self.loss_emb_A_center, -1, 1)
            self.loss_emb_B_center = torch.clamp(self.loss_emb_B_center, -1, 1)

            self.loss_emb_consistency_1 = torch.clamp(self.loss_emb_consistency_1, -1, 1)
            self.loss_emb_consistency_2 = torch.clamp(self.loss_emb_consistency_2, -1, 1)

            # Convert similarities to positive number
            self.loss_emb_A_large = (self.loss_emb_A_large + 1) / 2
            self.loss_emb_B_large = (self.loss_emb_B_large + 1) / 2

            self.loss_emb_A_center = (self.loss_emb_A_center + 1) / 2
            self.loss_emb_B_center = (self.loss_emb_B_center + 1) / 2

            self.loss_emb_A = (self.loss_emb_A_large + self.loss_emb_A_center) / 2
            self.loss_emb_B = (self.loss_emb_B_large + self.loss_emb_B_center) / 2

            self.loss_emb_consistency_1 = (self.loss_emb_consistency_1 + 1) / 2
            self.loss_emb_consistency_2 = (self.loss_emb_consistency_2 + 1) / 2
            self.loss_emb_consistency = (self.loss_emb_consistency_1 + self.loss_emb_consistency_2) / 2

            # Multipy by lambda_emb
            self.loss_emb_A = (1-self.loss_emb_A) * self.lambda_emb_A
            self.loss_emb_B = (1-self.loss_emb_B) * self.lambda_emb_B
            self.loss_emb_consistency = (1-self.loss_emb_consistency) * self.lambda_emb_A
        else:
            self.loss_emb_A = 0
            self.loss_emb_B = 0
            self.loss_emb_consistency = 0

        # Total generator loss
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + \
                      self.loss_emb_A + self.loss_emb_B + self.loss_emb_consistency

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A, G_B, and netPLIP_A's gradients to zero
        self.backward_G()             # calculate gradients for G_A, G_B, and netPLIP_A
        self.optimizer_G.step()       # update G_A, G_B, and netPLIP_A's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def save_networks(self, epoch):
        """Save all the networks to the disk."""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, torch.nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)
        # Save PLIP_A using Hugging Face's save_pretrained method
        save_dir = os.path.join(self.save_dir, '%s_net_PLIP_A' % epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.netPLIP_A.save_pretrained(save_dir)

    def load_networks(self, epoch):
        """Load all the networks from the disk."""
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, torch.nn.DataParallel):
                    net_module = net.module
                else:
                    net_module = net

                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                net_module.load_state_dict(state_dict)
                net_module.to(self.device)
        # Load PLIP_A using Hugging Face's from_pretrained method
        load_dir = os.path.join(self.save_dir, '%s_net_PLIP_A' % epoch)
        self.netPLIP_A = CLIPModel.from_pretrained(load_dir).to(self.device)

