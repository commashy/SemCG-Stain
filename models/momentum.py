import os
import copy
import itertools
import random
from collections import deque
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import torchvision

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks10 as networks

class CycleGAN15Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0,
                                help='use identity mapping.')

            # >>> Contrastive CLIp
            parser.add_argument('--use_clip_contrast', action='store_true', 
                                help='enable CLIP-based memory bank contrastive loss')
            parser.add_argument('--lambda_clip_contrast', type=float, default=1.0, 
                                help='weight for CLIP contrastive loss')
            parser.add_argument('--contrast_mem_size', type=int, default=1000, 
                                help='max memory bank size for clip embeddings')
            parser.add_argument('--contrast_negatives', type=int, default=256,
                                help='number of negatives to sample (for random) or top-k if using hard neg')
            parser.add_argument('--contrast_temp', type=float, default=0.3,
                                help='temperature for contrastive logits')

            # Additional argument for toggling hard negative sampling
            parser.add_argument('--use_hard_neg', action='store_true',
                                help='if set, we do top-K hard negative sampling instead of random.')
            # <<<
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN15Model class."""
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
        self.netG_A = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
            opt.normG, not opt.no_dropout, opt.init_type,
            opt.init_gain, self.gpu_ids
        ).to(self.device)

        self.netG_B = networks.define_G(
            opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
            opt.normG, not opt.no_dropout, opt.init_type,
            opt.init_gain, self.gpu_ids
        ).to(self.device)

        #----------------------------------
        # PLIP (CLIP) Models (Partial Fine-Tune)
        #----------------------------------
        self.netPLIP_A = CLIPModel.from_pretrained("vinid/plip").to(self.device)
        self.netPLIP_B = CLIPModel.from_pretrained("vinid/plip").to(self.device)
        self.processor_A = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)
        self.processor_B = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)

        # Freeze everything except patch embedding + last 4 layers in netPLIP_A (example)
        for name, param in self.netPLIP_A.named_parameters():
            if (
                "vision_model.encoder.layers.8" in name
                or "vision_model.encoder.layers.9" in name 
                or "vision_model.encoder.layers.10" in name 
                or "vision_model.encoder.layers.11" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Freeze except last 1-2 layers in netPLIP_B (example)
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
            self.netD_A = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.normD, opt.init_type,
                opt.init_gain, self.gpu_ids, num_classes=self.num_classes
            ).to(self.device)

            self.netD_B = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.normD, opt.init_type,
                opt.init_gain, self.gpu_ids, num_classes=self.num_classes
            ).to(self.device)

        #----------------------------------
        # If Training: Setup Losses, Optimizers
        #----------------------------------
        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert (opt.input_nc == opt.output_nc)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            #--- Losses
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = nn.L1Loss().to(self.device)
            self.criterionIdt = nn.L1Loss().to(self.device)
            self.criterionCE = nn.CrossEntropyLoss().to(self.device)

            #--- Contrastive
            self.use_clip_contrast = opt.use_clip_contrast
            self.lambda_clip_contrast = opt.lambda_clip_contrast
            self.contrast_mem_size = opt.contrast_mem_size
            self.contrast_negatives = opt.contrast_negatives
            self.contrast_temp = opt.contrast_temp

            # Param for toggling hard negative sampling
            self.use_hard_neg = opt.use_hard_neg

            if self.use_clip_contrast:
                # Use deque for memory banks (FIFO queue)
                self.memory_bank_A_real = deque(maxlen=self.contrast_mem_size)
                self.memory_bank_A_fake = deque(maxlen=self.contrast_mem_size)
                self.memory_bank_B_real = deque(maxlen=self.contrast_mem_size)
                self.memory_bank_B_fake = deque(maxlen=self.contrast_mem_size)

            #--- Optimizers
            backbone_lr = opt.lr * 0.2
            generator_lr = opt.lr
            discriminator_lr = opt.lr

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

            #----------------------------------
            # Momentum Encoders for CLIP
            #----------------------------------
            self.netPLIP_A_momentum = copy.deepcopy(self.netPLIP_A)
            self.netPLIP_B_momentum = copy.deepcopy(self.netPLIP_B)
            for param in self.netPLIP_A_momentum.parameters():
                param.requires_grad = False
            for param in self.netPLIP_B_momentum.parameters():
                param.requires_grad = False

    #===================================================
    # Hard Negative Sampling vs. Random
    #===================================================
    def gather_hard_negatives(self, emb_anchor: torch.Tensor, memory_list, top_k: int = 256) -> Optional[torch.Tensor]:
        """
        Hard negative sampling:
        1) Convert memory_list -> shape [M, D], and anchor -> shape [B, D].
        2) For each anchor, find the top_k embeddings in memory that yield
        the highest cosine similarity (most "hard" negatives).
        3) Return them stacked as shape [B*k, D].
        """
        if len(memory_list) == 0:
            return None

        emb_anchor = F.normalize(emb_anchor, dim=1)    # [B, D]
        memory_tensor = torch.stack(list(memory_list), dim=0).detach()  # [M, D]
        memory_tensor = F.normalize(memory_tensor, dim=1)

        B = emb_anchor.shape[0]
        M = memory_tensor.shape[0]
        # similarity => [B, M]
        sim = emb_anchor @ memory_tensor.t()

        k = min(top_k, M)
        # shape [B, k]
        _, idxs = torch.topk(sim, k=k, dim=1)

        # gather => shape [B*k, D]
        negs = memory_tensor[idxs].view(-1, memory_tensor.size(1))  # [B*k, D]
        return negs

    def clip_contrastive_loss(self, emb_anchor: torch.Tensor, emb_positive: torch.Tensor, memory_list) -> float:
        """
        InfoNCE-like contrastive loss with memory bank negatives.
        Hard negative sampling if self.use_hard_neg else random sample.
        """
        if len(memory_list) < max(1, self.contrast_negatives // 2):
            return 0.0

        B, D = emb_anchor.shape
        emb_anchor = F.normalize(emb_anchor, dim=1)
        emb_positive = F.normalize(emb_positive, dim=1)

        # anchor vs. positive => [B,1]
        
        pos_logit = (emb_anchor * emb_positive).sum(dim=1, keepdim=True) / self.contrast_temp

        if self.use_hard_neg:
            sample_size = min(self.contrast_negatives, len(memory_list))
            neg_tensors = self.gather_hard_negatives(emb_anchor, memory_list, top_k=sample_size)

            if neg_tensors is None:
                return 0.0

            # Ensure negatives are detached
            neg_tensors = neg_tensors.detach()

            Bk = neg_tensors.shape[0]  # B * sample_size
            anchor_expanded = torch.repeat_interleave(emb_anchor, repeats=sample_size, dim=0) 
            pos_logit_expanded = torch.repeat_interleave(pos_logit, repeats=sample_size, dim=0)

            neg_logit = (anchor_expanded * neg_tensors).sum(dim=1, keepdim=True) / self.contrast_temp

            logits = torch.cat([pos_logit_expanded, neg_logit], dim=1)
            labels = torch.zeros((Bk,), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            return loss

        else:
            # random sampling
            sample_size = min(self.contrast_negatives, len(memory_list))
            neg_samples = random.sample(memory_list, sample_size)
            neg_tensors = torch.stack(neg_samples, dim=0).detach()  # [sample_size, D]
            neg_tensors = F.normalize(neg_tensors, dim=1)

            neg_logits = emb_anchor.matmul(neg_tensors.t()) / self.contrast_temp

            # shape [B, 1 + sample_size]
            logits = torch.cat([pos_logit, neg_logits], dim=1)
            labels = torch.zeros((B,), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            return loss

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.isTrain:
            self.real_A_class = input['A_class' if AtoB else 'B_class'].long().to(self.device)
            self.real_B_class = input['B_class' if AtoB else 'A_class'].long().to(self.device)

        center_size = 256
        _, _, h, w = self.real_A.size()
        start_h = (h - center_size) // 2
        end_h = start_h + center_size
        start_w = (w - center_size) // 2
        end_w = start_w + center_size

        self.real_A_center = self.real_A[:, :, start_h:end_h, start_w:end_w]
        self.real_B_center = self.real_B[:, :, start_h:end_h, start_w:end_w]

    def unnormalize_image(self, tensor):
        tensor = (tensor + 1) / 2
        return torch.clamp(tensor, 0, 1)

    def forward(self):
        """Forward pass, center 256×256 only."""
        # embeddings for real_B
        images_real_B_center = self.unnormalize_image(self.real_B_center)
        inputs_real_B_center = self.processor_B(
            images=[img.cpu() for img in images_real_B_center],
            return_tensors="pt"
        ).to(self.device)
        self.embeddings_real_B = self.netPLIP_B.get_image_features(**inputs_real_B_center)

        # embeddings for real_A
        images_real_A_center = self.unnormalize_image(self.real_A_center)
        inputs_real_A_center = self.processor_A(
            images=[img.cpu() for img in images_real_A_center],
            return_tensors="pt"
        ).to(self.device)
        self.embeddings_real_A = self.netPLIP_A.get_image_features(**inputs_real_A_center)

        if self.isTrain:
            # generate fake_B, fake_A
            self.fake_B_center = self.netG_A(self.real_A_center, self.embeddings_real_A)
            self.fake_A_center = self.netG_B(self.real_B_center, self.embeddings_real_B)

            # embeddings for fakes
            images_fake_B_center = self.unnormalize_image(self.fake_B_center)
            input_fake_B_center = self.processor_B(
                images=[img.cpu() for img in images_fake_B_center],
                return_tensors="pt"
            ).to(self.device)
            self.embeddings_fake_B = self.netPLIP_B.get_image_features(**input_fake_B_center)

            images_fake_A_center = self.unnormalize_image(self.fake_A_center)
            input_fake_A_center = self.processor_A(
                images=[img.cpu() for img in images_fake_A_center],
                return_tensors="pt"
            ).to(self.device)
            self.embeddings_fake_A = self.netPLIP_A.get_image_features(**input_fake_A_center)

            # reconstruct cycle
            gaussian_noise = 0.01 * torch.randn_like(self.fake_A_center)
            self.rec_A_center = self.netG_B(self.fake_B_center + gaussian_noise, self.embeddings_fake_B)
            self.rec_B_center = self.netG_A(self.fake_A_center + gaussian_noise, self.embeddings_fake_A)

            self.fake_A = self.fake_A_center
            self.fake_B = self.fake_B_center
        else:
            # test mode
            self.fake_B_center = self.netG_A(self.real_A_center, self.embeddings_real_A)
            self.fake_A_center = self.netG_B(self.real_B_center, self.embeddings_real_B)
            self.fake_A = self.fake_A_center
            self.fake_B = self.fake_B_center

    def calculate_disc_loss(self, preds, labels):
        loss = 0
        for pred in preds:
            loss += self.criterionCE(pred, labels)
        return loss

    def backward_D_basic(self, netD, real, real_labels, fake, fake_labels):
        gp_lambda = 10.0
        real.requires_grad_(True)
        fake.requires_grad_(True)

        preds_real = netD(real)
        preds_fake = netD(fake)

        loss_real = self.calculate_disc_loss(preds_real, real_labels)
        loss_fake = self.calculate_disc_loss(preds_fake, fake_labels)
        loss_D = loss_real + loss_fake

        # gradient penalty
        if gp_lambda > 0:
            real_class_indices = real_labels.view(-1)
            preds_real_sum = 0
            for pred_scale in preds_real:
                preds_real_sum += pred_scale[torch.arange(pred_scale.size(0)), real_class_indices]

            grad_outputs_real = torch.ones_like(preds_real_sum)
            gradients_real = torch.autograd.grad(
                outputs=preds_real_sum, inputs=real,
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
                outputs=preds_fake_sum, inputs=fake,
                grad_outputs=grad_outputs_fake,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            gradient_norms_fake = gradients_fake.view(gradients_fake.size(0), -1).norm(2, dim=1)
            loss_gp_fake = (gradient_norms_fake ** 2).mean()

            gradient_penalty = gp_lambda * (loss_gp_real + loss_gp_fake) / 2
            loss_D_total = loss_D + gradient_penalty
        else:
            loss_D_total = loss_D

        loss_D_total.backward()

        return loss_D_total

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B.detach())
        fake_labels = torch.full((fake_B.size(0),), self.num_classes-1,
                                 dtype=torch.long, device=self.device)
        real_labels = self.real_B_class.view(-1)
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self.real_B_center, real_labels, fake_B, fake_labels
        )

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        fake_labels = torch.full((fake_A.size(0),), self.num_classes-1,
                                 dtype=torch.long, device=self.device)
        real_labels = self.real_A_class.view(-1)
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.real_A_center, real_labels, fake_A, fake_labels
        )

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B_center, self.embeddings_real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B_center) * lambda_B * lambda_idt

            self.idt_B = self.netG_B(self.real_A_center, self.embeddings_real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A_center) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # adversarial terms
        domain_label_B = self.real_B_class
        out_fake_B = self.netD_A(self.fake_B)
        loss_G_A = sum(self.criterionCE(scale_out, domain_label_B) for scale_out in out_fake_B)
        self.loss_G_A = loss_G_A

        domain_label_A = self.real_A_class
        out_fake_A = self.netD_B(self.fake_A)
        loss_G_B = sum(self.criterionCE(scale_out, domain_label_A) for scale_out in out_fake_A)
        self.loss_G_B = loss_G_B

        # cycle
        self.loss_cycle_A = self.criterionCycle(self.rec_A_center, self.real_A_center) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B_center, self.real_B_center) * lambda_B

        self.loss_G = (
            self.loss_G_A + self.loss_G_B +
            self.loss_cycle_A + self.loss_cycle_B +
            self.loss_idt_A + self.loss_idt_B
        )

        # optional clip contrast
        if self.use_clip_contrast:
            # realA<->fakeB
            clip_loss_AB = self.clip_contrastive_loss(
                self.embeddings_real_A,
                self.embeddings_fake_B,
                list(self.memory_bank_A_real) + list(self.memory_bank_A_fake)
            )
            # realB<->fakeA
            clip_loss_BA = self.clip_contrastive_loss(
                self.embeddings_real_B,
                self.embeddings_fake_A,
                list(self.memory_bank_B_real) + list(self.memory_bank_B_fake)
            )
            self.loss_clip_contrast = (clip_loss_AB + clip_loss_BA) * self.lambda_clip_contrast
            self.loss_G += self.loss_clip_contrast
        else:
            self.loss_clip_contrast = 0.0

        self.loss_G.backward()

    def update_ema(self, alpha=0.999):
        """
        Update EMA for netG_A_EMA, netG_B_EMA with decay=alpha.
        """
        with torch.no_grad():
            for p_ema, p in zip(self.netG_A_EMA.parameters(), self.netG_A.parameters()):
                p_ema.data = alpha * p_ema.data + (1.0 - alpha) * p.data
            for p_ema, p in zip(self.netG_B_EMA.parameters(), self.netG_B.parameters()):
                p_ema.data = alpha * p_ema.data + (1.0 - alpha) * p.data

    #===================================================
    # Momentum Encoders
    #===================================================
    def update_momentum_clip(self, alpha=0.99):
        """Update netPLIP_A_momentum, netPLIP_B_momentum with EMA of netPLIP_A, netPLIP_B."""
        if not self.use_clip_contrast:
            return
        with torch.no_grad():
            for p_online, p_momentum in zip(self.netPLIP_A.parameters(), self.netPLIP_A_momentum.parameters()):
                p_momentum.data = p_momentum.data * alpha + p_online.data * (1.0 - alpha)
            for p_online, p_momentum in zip(self.netPLIP_B.parameters(), self.netPLIP_B_momentum.parameters()):
                p_momentum.data = p_momentum.data * alpha + p_online.data * (1.0 - alpha)

    def encode_momentum(self, netPLIP_momentum, processor, images):
        """Encode images using the momentum CLIP model (no gradient)."""
        images_unnorm = self.unnormalize_image(images)
        inputs = processor(
            images=[img.cpu() for img in images_unnorm],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            emb = netPLIP_momentum.get_image_features(**inputs)  # [B, D]
        return emb

    def optimize_parameters(self):
        # 1) Forward
        self.forward()

        # 2) G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        with torch.autograd.detect_anomaly():
            self.backward_G()
        self.optimizer_G.step()
        self.update_ema(alpha=0.999)

        # 3) Update momentum CLIP encoders
        self.update_momentum_clip(alpha=0.99)

        # 4) D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        with torch.autograd.detect_anomaly():
            self.backward_D_A()
            self.backward_D_B()
        self.optimizer_D.step()

        # 5) Update memory banks with momentum embeddings
        if self.use_clip_contrast:
            with torch.no_grad():
                emb_realA_m = self.encode_momentum(self.netPLIP_A_momentum, self.processor_A, self.real_A_center)
                emb_fakeB_m = self.encode_momentum(self.netPLIP_B_momentum, self.processor_B, self.fake_B_center)
                emb_realB_m = self.encode_momentum(self.netPLIP_B_momentum, self.processor_B, self.real_B_center)
                emb_fakeA_m = self.encode_momentum(self.netPLIP_A_momentum, self.processor_A, self.fake_A_center)

            B = emb_realA_m.size(0)
            for i in range(B):
                self.memory_bank_A_real.append(emb_realA_m[i].clone())
                self.memory_bank_A_fake.append(emb_fakeB_m[i].clone())
                self.memory_bank_B_real.append(emb_realB_m[i].clone())
                self.memory_bank_B_fake.append(emb_fakeA_m[i].clone())

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

        # Also save momentum models
        save_dir_A_mom = os.path.join(self.save_dir, f'{epoch}_net_PLIP_A_momentum')
        if not os.path.exists(save_dir_A_mom):
            os.makedirs(save_dir_A_mom)
        self.netPLIP_A_momentum.save_pretrained(save_dir_A_mom)

        save_dir_B_mom = os.path.join(self.save_dir, f'{epoch}_net_PLIP_B_momentum')
        if not os.path.exists(save_dir_B_mom):
            os.makedirs(save_dir_B_mom)
        self.netPLIP_B_momentum.save_pretrained(save_dir_B_mom)

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

        # Load momentum PLIP_A, PLIP_B if they exist
        load_dir_A_mom = os.path.join(self.save_dir, f'{epoch}_net_PLIP_A_momentum')
        if os.path.isdir(load_dir_A_mom):
            print(f'Loading PLIP_A_momentum from {load_dir_A_mom}')
            self.netPLIP_A_momentum = CLIPModel.from_pretrained(load_dir_A_mom).to(self.device)
            for param in self.netPLIP_A_momentum.parameters():
                param.requires_grad = False

        load_dir_B_mom = os.path.join(self.save_dir, f'{epoch}_net_PLIP_B_momentum')
        if os.path.isdir(load_dir_B_mom):
            print(f'Loading PLIP_B_momentum from {load_dir_B_mom}')
            self.netPLIP_B_momentum = CLIPModel.from_pretrained(load_dir_B_mom).to(self.device)
            for param in self.netPLIP_B_momentum.parameters():
                param.requires_grad = False

