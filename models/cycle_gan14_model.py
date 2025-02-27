import os
import copy
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks9 as networks

class CycleGAN14Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.05, help='use identity mapping.')

            # >>> Contrastive CLIp
            parser.add_argument('--use_clip_contrast', action='store_true', 
                                help='whether to enable CLIP-based memory bank contrastive loss')
            parser.add_argument('--lambda_clip_contrast', type=float, default=1.0, 
                                help='weight for CLIP contrastive loss')
            parser.add_argument('--contrast_mem_size', type=int, default=1000, 
                                help='max memory bank size for clip embeddings')
            parser.add_argument('--contrast_negatives', type=int, default=256,
                                help='number of negatives to sample from memory bank')
            parser.add_argument('--contrast_temp', type=float, default=0.07,
                                help='temperature for contrastive logits')
            # <<<

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN14Model class."""
        BaseModel.__init__(self, opt)
        #----------------------------------
        # Setup Loss Names & Visuals
        #----------------------------------
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        visual_names_A = ['real_A_center', 'fake_B_center', 'rec_A_center']
        visual_names_B = ['real_B_center', 'fake_A_center', 'rec_B_center']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.loss_names += ['idt_A', 'idt_B']

        # If we use clip contrast, let's add it to the loss_names
        if self.isTrain and self.opt.use_clip_contrast:
            self.loss_names += ['clip_contrast']

        self.visual_names = visual_names_A + visual_names_B

        #----------------------------------
        # Models to Save
        #----------------------------------
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'G_A_EMA', 'G_B_EMA']
        else:
            self.model_names = ['G_A', 'G_B']

        #----------------------------------
        # Generators
        #----------------------------------
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                        opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, self.gpu_ids).to(self.device)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
                                        opt.normG, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, self.gpu_ids).to(self.device)

        #----------------------------------
        # PLIP (CLIP) Models (Partial Fine-Tune)
        #----------------------------------
        self.netPLIP_A = CLIPModel.from_pretrained("vinid/plip").to(self.device)
        self.netPLIP_B = CLIPModel.from_pretrained("vinid/plip").to(self.device)
        self.processor_A = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)
        self.processor_B = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)

        # Freeze except last 1-2 layers in netPLIP_A
        for name, param in self.netPLIP_A.named_parameters():
            if "vision_model.encoder.layers.10" in name or "vision_model.encoder.layers.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Freeze except last 1-2 layers in netPLIP_B
        for name, param in self.netPLIP_B.named_parameters():
            if "vision_model.encoder.layers.10" in name or "vision_model.encoder.layers.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        #----------------------------------
        # Discriminators
        #----------------------------------
        self.num_classes = 10  # domain classes + 1 fake label
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type,
                                            opt.init_gain, self.gpu_ids, num_classes=self.num_classes).to(self.device)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type,
                                            opt.init_gain, self.gpu_ids, num_classes=self.num_classes).to(self.device)

        #----------------------------------
        # If Training: Setup Losses, Optimizers
        #----------------------------------
        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            #--- Losses
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = nn.L1Loss().to(self.device)
            self.criterionIdt = nn.L1Loss().to(self.device)
            self.criterionCE = nn.CrossEntropyLoss().to(self.device)

            # Optionally, define new memory banks for clip contrast
            self.use_clip_contrast = opt.use_clip_contrast
            self.lambda_clip_contrast = opt.lambda_clip_contrast
            self.contrast_mem_size = opt.contrast_mem_size
            self.contrast_negatives = opt.contrast_negatives
            self.contrast_temp = opt.contrast_temp

            if self.use_clip_contrast:
                # memory for domain A and domain B
                # we'll store embeddings as 1D vectors [D]
                self.memory_bank_A_real = []
                self.memory_bank_A_fake = []
                self.memory_bank_B_real = []
                self.memory_bank_B_fake = []

            #--- Optimizers
            backbone_lr = opt.lr * 0.5
            generator_lr = opt.lr
            discriminator_lr = opt.lr * 2

            g_params = list(self.netG_A.parameters()) + list(self.netG_B.parameters())
            plipA_params = (p for p in self.netPLIP_A.parameters() if p.requires_grad)
            plipB_params = (p for p in self.netPLIP_B.parameters() if p.requires_grad)

            param_groups = [
                {"params": g_params, "lr": generator_lr},
                {"params": plipA_params, "lr": backbone_lr},
                {"params": plipB_params, "lr": backbone_lr},
            ]
            self.optimizer_G = torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=discriminator_lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            #----------------------------------
            # Create EMA copies of G_A, G_B
            #----------------------------------
            self.netG_A_EMA = copy.deepcopy(self.netG_A)
            self.netG_B_EMA = copy.deepcopy(self.netG_B)
            for param in self.netG_A_EMA.parameters():
                param.requires_grad = False
            for param in self.netG_B_EMA.parameters():
                param.requires_grad = False

    #===================================================
    # Simple function to do a memory-based contrastive
    #  anchor= real, positive= fake, negatives = memory
    #===================================================
    def clip_contrastive_loss(self, emb_anchor, emb_positive, memory_list):
        """
        emb_anchor, emb_positive: shape [1, D] for batch=1
        memory_list: list of torch tensors (each shape [D]) 
                     from previous iterations (real or fake).
        We'll do an InfoNCE with 1 anchor, 1 positive, 
        K negatives from memory_list.
        
        Return a scalar loss.
        """
        if len(memory_list) < (self.contrast_negatives / 2):
            # No negatives => forced to do something trivial
            return 0.0

        # Convert to [D] => [1, D]
        emb_anchor = F.normalize(emb_anchor, dim=1)  # shape [1, D]
        emb_positive = F.normalize(emb_positive, dim=1)
        
        # gather negatives from memory
        # we can just random sample up to e.g. 32 from memory
        # or use them all, up to you. We'll do a small sample for performance
        sample_size = min(self.contrast_negatives, len(memory_list))
        neg_samples = random.sample(memory_list, sample_size)
        neg_tensors = torch.stack(neg_samples, dim=0).to(self.device)  # shape [sample_size, D]
        neg_tensors = F.normalize(neg_tensors, dim=1)

        # anchor vs. positive sim
        pos_logit = torch.matmul(emb_anchor, emb_positive.t()) / self.contrast_temp  # shape [1,1]

        # anchor vs. negatives
        neg_logits = torch.matmul(emb_anchor, neg_tensors.t()) / self.contrast_temp  # shape [1, sample_size]

        # Combine => shape [1, 1 + sample_size]
        logits = torch.cat([pos_logit, neg_logits], dim=1)  # [1, 1 + sample_size]

        # Label = 0 => first column is correct (the positive)
        labels = torch.zeros((1,), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def extract_patches(self, images, patch_size):
        batch_size, channels, height, width = images.size()
        assert height % patch_size == 0 and width % patch_size == 0, \
            "Image dimensions must be divisible by patch size"
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size

        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(-1, channels, patch_size, patch_size)
        return patches, num_patches_h, num_patches_w

    def reconstruct_from_patches(self, patches, num_patches_h, num_patches_w, batch_size, channels, patch_size):
        num_patches = num_patches_h * num_patches_w
        patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        images = patches.view(batch_size, channels, num_patches_h * patch_size, num_patches_w * patch_size)
        return images

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.isTrain:
            # class labels
            self.real_A_class = input['A_class' if AtoB else 'B_class']
            self.real_B_class = input['B_class' if AtoB else 'A_class']
            self.real_A_class = torch.tensor(self.real_A_class, dtype=torch.long).to(self.device)
            self.real_B_class = torch.tensor(self.real_B_class, dtype=torch.long).to(self.device)

        # center crop to 256x256
        center_size = 256
        large_size = self.real_A.shape[2]
        start = (large_size - center_size) // 2
        end = start + center_size
        self.real_A_center = self.real_A[:, :, start:end, start:end]
        self.real_B_center = self.real_B[:, :, start:end, start:end]

    def unnormalize_image(self, tensor):
        """(B, C, H, W) in [-1,1] -> [0,1]."""
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

    def forward(self):
        """
        Forward pass, includes partial fine-tuning of netPLIP_A/B for domains.
        We do not compute embedding loss, but we gather embeddings for cross-attention if needed.
        """
        # 1) Embeddings for real B
        with torch.no_grad():
            images_real_B = self.unnormalize_image(self.real_B)
            image_inputs_real_B = self.processor_B(
                images=[img.cpu() for img in images_real_B],
                return_tensors="pt"
            ).to(self.device)
            self.embeddings_real_B = self.netPLIP_B.get_image_features(**image_inputs_real_B)

        # 2) Embeddings for real A (partial fine-tuning netPLIP_A)
        images_real_A = self.unnormalize_image(self.real_A)
        image_inputs_real_A = self.processor_A(
            images=[img.cpu() for img in images_real_A],
            return_tensors="pt"
        ).to(self.device)
        self.embeddings_real_A = self.netPLIP_A.get_image_features(**image_inputs_real_A)

        if self.isTrain:
            # 2b) Also get center embeddings if needed
            # with torch.no_grad():
            #     images_real_B_center = self.unnormalize_image(self.real_B_center)
            #     inputs_B_center = self.processor_B(
            #         images=[img.cpu() for img in images_real_B_center],
            #         return_tensors="pt"
            #     ).to(self.device)
            #     self.embeddings_real_B_center = self.netPLIP_B.get_image_features(**inputs_B_center)

            #     images_real_A_center = self.unnormalize_image(self.real_A_center)
            #     inputs_A_center = self.processor_A(
            #         images=[img.cpu() for img in images_real_A_center],
            #         return_tensors="pt"
            #     ).to(self.device)
            #     self.embeddings_real_A_center = self.netPLIP_A.get_image_features(**inputs_A_center)

            # Split real images into patches for G_A / G_B
            patch_size = 256
            patches_real_A, num_patches_h, num_patches_w = self.extract_patches(self.real_A, patch_size)
            patches_real_B, _, _ = self.extract_patches(self.real_B, patch_size)

            batch_size_patches = patches_real_A.size(0)
            batch_size_processing = 3  # small batch chunk
            batch_size = self.real_A.size(0)
            num_patches_per_image = patches_real_A.size(0) // batch_size
            embedding_dim = self.embeddings_real_A.size(1)

            # Expand embeddings to match patches
            embeddings_real_A_expanded = self.embeddings_real_A.unsqueeze(1).expand(-1, num_patches_per_image, -1)
            embeddings_real_A_expanded = embeddings_real_A_expanded.reshape(-1, embedding_dim)

            embeddings_real_B_expanded = self.embeddings_real_B.unsqueeze(1).expand(-1, num_patches_per_image, -1)
            embeddings_real_B_expanded = embeddings_real_B_expanded.reshape(-1, embedding_dim)

            fake_B_patches = []
            fake_A_patches = []

            # 3) Process patches in chunks
            for i in range(0, batch_size_patches, batch_size_processing):
                end = min(i + batch_size_processing, batch_size_patches)
                input_patches_A = patches_real_A[i:end]
                input_patches_B = patches_real_B[i:end]

                emb_patches_A = embeddings_real_A_expanded[i:end]
                emb_patches_B = embeddings_real_B_expanded[i:end]

                fake_B_patch = self.netG_A(input_patches_A, emb_patches_A)
                fake_B_patches.append(fake_B_patch)

                fake_A_patch = self.netG_B(input_patches_B, emb_patches_B)
                fake_A_patches.append(fake_A_patch)

            fake_B_patches = torch.cat(fake_B_patches, dim=0)
            fake_A_patches = torch.cat(fake_A_patches, dim=0)

            # 4) Reconstruct full images
            self.fake_B = self.reconstruct_from_patches(
                fake_B_patches, num_patches_h, num_patches_w,
                self.real_A.size(0), self.real_A.size(1), patch_size
            )
            self.fake_A = self.reconstruct_from_patches(
                fake_A_patches, num_patches_h, num_patches_w,
                self.real_B.size(0), self.real_B.size(1), patch_size
            )

            # 5) Extract center region of fakes
            center_size = 256
            start = (self.real_A.shape[2] - center_size) // 2
            end = start + center_size
            self.fake_A_center = self.fake_A[:, :, start:end, start:end]
            self.fake_B_center = self.fake_B[:, :, start:end, start:end]

            # 6) Embeddings for fake images
            images_fake_A = self.unnormalize_image(self.fake_A)
            input_fake_A = self.processor_A(
                images=[img.cpu() for img in images_fake_A],
                return_tensors="pt"
            ).to(self.device)
            self.embeddings_fake_A = self.netPLIP_A.get_image_features(**input_fake_A)

            images_fake_B = self.unnormalize_image(self.fake_B)
            input_fake_B = self.processor_B(
                images=[img.cpu() for img in images_fake_B],
                return_tensors="pt"
            ).to(self.device)
            self.embeddings_fake_B = self.netPLIP_B.get_image_features(**input_fake_B)

            # 7) Reconstruct center region with slight noise
            gaussian_noise = 0.01 * torch.randn(self.fake_A_center.size(), device=self.device)
            self.rec_A_center = self.netG_B(self.fake_B_center + gaussian_noise, self.embeddings_fake_B)
            self.rec_B_center = self.netG_A(self.fake_A_center + gaussian_noise, self.embeddings_fake_A)

        else:
            # In test mode, directly pass real images + partial fine-tuned embeddings
            self.fake_B = self.netG_A(self.real_A, self.embeddings_real_A)
            self.fake_A = self.netG_B(self.real_B, self.embeddings_real_B)

    def calculate_disc_loss(self, preds, labels):
        """Sum cross-entropy across multi-scale predictions."""
        loss = 0
        for pred in preds:
            loss += self.criterionCE(pred, labels)
        return loss

    def backward_D_basic(self, netD, real, real_labels, fake, fake_labels):
        """
        netD(...) -> list of [B, num_classes] at multiple scales.
        We do cross-entropy for real->real_labels, fake->fake_labels.
        Then R1+R2 gradient penalty.
        """
        gp_lambda = 0.1
        real.requires_grad_(True)
        fake.requires_grad_(True)

        preds_real = netD(real)   # list of [B, num_classes]
        preds_fake = netD(fake)   # list of [B, num_classes]

        loss_real = self.calculate_disc_loss(preds_real, real_labels)
        loss_fake = self.calculate_disc_loss(preds_fake, fake_labels)
        loss_D = loss_real + loss_fake

        # Gradient penalty: R1 + R2
        real_class_indices = real_labels.view(-1)
        preds_real_sum = 0
        for pred_scale in preds_real:
            preds_real_sum += pred_scale[torch.arange(pred_scale.size(0)), real_class_indices]

        grad_outputs_real = torch.ones_like(preds_real_sum)
        gradients_real = torch.autograd.grad(
            outputs=preds_real_sum,
            inputs=real,
            grad_outputs=grad_outputs_real,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradient_norms_real = gradients_real.view(gradients_real.size(0), -1).norm(2, dim=1)
        loss_gp_real = (gradient_norms_real ** 2).mean()

        fake_class_indices = fake_labels.view(-1)
        preds_fake_sum = 0
        for pred_scale in preds_fake:
            preds_fake_sum += pred_scale[torch.arange(pred_scale.size(0)), fake_class_indices]

        grad_outputs_fake = torch.ones_like(preds_fake_sum)
        gradients_fake = torch.autograd.grad(
            outputs=preds_fake_sum,
            inputs=fake,
            grad_outputs=grad_outputs_fake,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradient_norms_fake = gradients_fake.view(gradients_fake.size(0), -1).norm(2, dim=1)
        loss_gp_fake = (gradient_norms_fake ** 2).mean()

        gradient_penalty = gp_lambda * (loss_gp_real + loss_gp_fake) / 2
        loss_D_total = loss_D + gradient_penalty
        loss_D_total.backward()
        return loss_D_total

    def backward_D_A(self):
        """
        netD_A sees real_B vs. fake_B.
        Real => self.real_B_class
        Fake => self.num_classes-1
        """
        fake_B = self.fake_B_pool.query(self.fake_B.detach())
        fake_labels = torch.full((fake_B.size(0),), self.num_classes-1, dtype=torch.long, device=self.device)
        real_labels = self.real_B_class.view(-1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, real_labels, fake_B, fake_labels)

    def backward_D_B(self):
        """
        netD_B sees real_A vs. fake_A.
        Real => self.real_A_class
        Fake => self.num_classes-1
        """
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        fake_labels = torch.full((fake_A.size(0),), self.num_classes-1, dtype=torch.long, device=self.device)
        real_labels = self.real_A_class.view(-1)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, real_labels, fake_A, fake_labels)

    def backward_G(self):
        """
        - netD_A: fake_B => domain_label_B
        - netD_B: fake_A => domain_label_A
        + cycle, identity
        + optional: CLIP contrast with memory
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B_center, self.embeddings_real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B_center) * lambda_B * lambda_idt

            self.idt_B = self.netG_B(self.real_A_center, self.embeddings_real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A_center) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # netD_A wants fake_B => domain_label_B
        domain_label_B = self.real_B_class
        out_fake_B = self.netD_A(self.fake_B)
        loss_G_A = 0.0
        for out_scale in out_fake_B:
            loss_G_A += self.criterionCE(out_scale, domain_label_B)
        self.loss_G_A = loss_G_A

        # netD_B wants fake_A => domain_label_A
        domain_label_A = self.real_A_class
        out_fake_A = self.netD_B(self.fake_A)
        loss_G_B = 0.0
        for out_scale in out_fake_A:
            loss_G_B += self.criterionCE(out_scale, domain_label_A)
        self.loss_G_B = loss_G_B

        # Cycle
        self.loss_cycle_A = self.criterionCycle(self.rec_A_center, self.real_A_center) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B_center, self.real_B_center) * lambda_B

        self.loss_G = (self.loss_G_A + self.loss_G_B +
                       self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B)

        # If we want CLIP contrast
        if self.use_clip_contrast:
            # realA <-> fakeB is a pair => anchor= realA, positive= fakeB
            # realB <-> fakeA is a pair => anchor= realB, positive= fakeA

            # shape [B=1, D], we have self.embeddings_real_A, self.embeddings_fake_B
            # same for B domain
            # We'll do memory-based negative sampling
            clip_loss_AB = self.clip_contrastive_loss(self.embeddings_real_A, self.embeddings_fake_B,
                                                      self.memory_bank_A_real + self.memory_bank_A_fake)
            clip_loss_BA = self.clip_contrastive_loss(self.embeddings_real_B, self.embeddings_fake_A,
                                                      self.memory_bank_B_real + self.memory_bank_B_fake)

            self.loss_clip_contrast = (clip_loss_AB + clip_loss_BA) * self.lambda_clip_contrast
            self.loss_G += self.loss_clip_contrast
        else:
            self.loss_clip_contrast = 0.0

        self.loss_G.backward()

    # EMA Update
    def update_ema(self, alpha=0.999):
        """Update EMA for netG_A_EMA, netG_B_EMA with decay=alpha."""
        with torch.no_grad():
            for p_ema, p in zip(self.netG_A_EMA.parameters(), self.netG_A.parameters()):
                p_ema.data = alpha * p_ema.data + (1.0 - alpha) * p.data
            for p_ema, p in zip(self.netG_B_EMA.parameters(), self.netG_B.parameters()):
                p_ema.data = alpha * p_ema.data + (1.0 - alpha) * p.data

    def optimize_parameters(self):
        self.forward()

        # 1) Update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.update_ema(alpha=0.999)

        # 2) Update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

        # 3) Update memory banks with real/fake embeddings
        if self.use_clip_contrast:
            # store realA => memory_bank_A_real
            embA = self.embeddings_real_A.detach().cpu().squeeze(0)  # shape [D]
            self.memory_bank_A_real.append(embA)
            if len(self.memory_bank_A_real) > self.contrast_mem_size:
                self.memory_bank_A_real.pop(0)

            # store fakeB => memory_bank_A_fake
            embB = self.embeddings_fake_B.detach().cpu().squeeze(0)
            self.memory_bank_A_fake.append(embB)
            if len(self.memory_bank_A_fake) > self.contrast_mem_size:
                self.memory_bank_A_fake.pop(0)

            # store realB => memory_bank_B_real
            embRB = self.embeddings_real_B.detach().cpu().squeeze(0)
            self.memory_bank_B_real.append(embRB)
            if len(self.memory_bank_B_real) > self.contrast_mem_size:
                self.memory_bank_B_real.pop(0)

            # store fakeA => memory_bank_B_fake
            embFA = self.embeddings_fake_A.detach().cpu().squeeze(0)
            self.memory_bank_B_fake.append(embFA)
            if len(self.memory_bank_B_fake) > self.contrast_mem_size:
                self.memory_bank_B_fake.pop(0)

    def save_networks(self, epoch):
        """Save all networks (including PLIP_A, PLIP_B, and EMA)."""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name, None)
                if net is None:
                    continue
                save_filename = f'{epoch}_net_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                if isinstance(net, nn.DataParallel):
                    torch.save(net.module.state_dict(), save_path)
                else:
                    torch.save(net.state_dict(), save_path)

        # Save partially fine-tuned PLIP_A + PLIP_B
        save_dir_A = os.path.join(self.save_dir, f'{epoch}_net_PLIP_A')
        if not os.path.exists(save_dir_A):
            os.makedirs(save_dir_A)
        self.netPLIP_A.save_pretrained(save_dir_A)

        save_dir_B = os.path.join(self.save_dir, f'{epoch}_net_PLIP_B')
        if not os.path.exists(save_dir_B):
            os.makedirs(save_dir_B)
        self.netPLIP_B.save_pretrained(save_dir_B)

    def load_networks(self, epoch):
        """Load all networks (including partially fine-tuned PLIP_A & PLIP_B)."""
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f'{epoch}_net_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name, None)
                if net is None:
                    continue
                print(f'loading the model from {load_path}')
                state_dict = torch.load(load_path, map_location=self.device)
                if isinstance(net, nn.DataParallel):
                    net.module.load_state_dict(state_dict)
                else:
                    net.load_state_dict(state_dict)
                net.to(self.device)

        # Load partially fine-tuned PLIP_A + PLIP_B
        load_dir_A = os.path.join(self.save_dir, f'{epoch}_net_PLIP_A')
        print(f'Loading PLIP_A from {load_dir_A}')
        self.netPLIP_A = CLIPModel.from_pretrained(load_dir_A).to(self.device)

        load_dir_B = os.path.join(self.save_dir, f'{epoch}_net_PLIP_B')
        print(f'Loading PLIP_B from {load_dir_B}')
        self.netPLIP_B = CLIPModel.from_pretrained(load_dir_B).to(self.device)
