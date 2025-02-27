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

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_SemCG as networks

class SemCGStainModel(BaseModel):
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
            # >>> Contrastive CLIP
            parser.add_argument('--use_clip_contrast', action='store_true', 
                                help='enable CLIP-based memory bank contrastive loss')
            parser.add_argument('--lambda_clip_contrast', type=float, default=0.2, 
                                help='weight for CLIP contrastive loss')
            parser.add_argument('--contrast_mem_size', type=int, default=1000, 
                                help='max memory bank size for clip embeddings')
            parser.add_argument('--contrast_negatives', type=int, default=256,
                                help='number of negatives to sample (for random) or top-k if using hard neg')
            parser.add_argument('--contrast_temp', type=float, default=0.07,
                                help='temperature for contrastive logits')
            parser.add_argument('--use_hard_neg', action='store_true',
                                help='if set, we do top-K hard negative sampling instead of random.')
            # <<<

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        #----------------------------------
        # Setup Loss Names & Visuals
        #----------------------------------
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'tv']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        if self.isTrain:
            visual_names_A.append('rec_A')
            visual_names_B.append('rec_B')
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.loss_names += ['idt_A', 'idt_B']
        if self.isTrain and self.opt.use_clip_contrast:
            self.loss_names += ['clip_contrast']

        self.visual_names = visual_names_A + visual_names_B

        #----------------------------------
        # Models to Save
        #----------------------------------
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
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
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(self.device)
        
        if self.isTrain and self.opt.use_clip_contrast:
            self.netPLIP_A = CLIPModel.from_pretrained("vinid/plip").to(self.device)
            self.netPLIP_B = CLIPModel.from_pretrained("vinid/plip").to(self.device)
            self.processor_A = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)
            self.processor_B = CLIPProcessor.from_pretrained("vinid/plip", do_rescale=False)

            for name, param in self.netPLIP_A.named_parameters():
                if ("vision_model.encoder.layers.10" in name or
                    "vision_model.encoder.layers.11" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            for param in self.netPLIP_B.parameters():
                param.requires_grad = False

        #----------------------------------
        # Discriminators: multi-scale with single-channel output (LSGAN)
        #----------------------------------
        if self.isTrain:
            if self.opt.netD == 'multi':
                self.num_classes = 8
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
            else:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)

        #----------------------------------
        # Setup Losses and Optimizers
        #----------------------------------
        if not self.isTrain:
            self.opt.use_clip_contrast = True
        
        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # Losses
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = nn.L1Loss().to(self.device)
            self.criterionIdt = nn.L1Loss().to(self.device)
            self.criterionCE = nn.CrossEntropyLoss().to(self.device)
            self.logsoft=torch.nn.LogSoftmax(dim=1).to(self.device)
            self.tv_loss = TVLoss().to(self.device)

            # Hyperparameters for contrastive loss
            if self.opt.use_clip_contrast:
                self.lambda_clip_contrast = opt.lambda_clip_contrast
                self.contrast_mem_size = opt.contrast_mem_size
                self.contrast_negatives = opt.contrast_negatives
                self.contrast_temp = opt.contrast_temp
                self.use_hard_neg = opt.use_hard_neg

                self.memory_bank_A_real = deque(maxlen=self.contrast_mem_size)
                self.memory_bank_A_fake = deque(maxlen=self.contrast_mem_size)
                self.memory_bank_B_real = deque(maxlen=self.contrast_mem_size)
                self.memory_bank_B_fake = deque(maxlen=self.contrast_mem_size)

            # Optimizers
            if self.opt.use_clip_contrast:
                backbone_lr = opt.lr * 0.5 
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
                self.optimizer_G = torch.optim.AdamW(param_groups, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.AdamW(
                    itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                    lr=discriminator_lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D)
            else:
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                    lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizer_D = torch.optim.Adam(
                    itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                    lr=opt.lr, betas=(opt.beta1, 0.999)
                )
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_D)

    #---------------------------------------------
    # Hard Negative Sampling and Contrastive Loss
    #---------------------------------------------
    def gather_hard_negatives_two_stage(self, emb_anchor: torch.Tensor, memory_list, top_k: int = 1000, sample_size: int = 256) -> Optional[torch.Tensor]:
        # if memory_list is empty, return None
        if len(memory_list) == 0:
            return None
        
        memory_tensor = torch.stack(list(memory_list), dim=0).detach()  # [M, D]
        B, D = emb_anchor.shape
        M = memory_tensor.shape[0]
        sim = emb_anchor @ memory_tensor.t()

        # sample negatives
        k = min(top_k, M)
        _, idxs = torch.topk(sim, k=k, dim=1)
        topk_tensors = memory_tensor[idxs.view(-1)].view(B, k, D)
        sample_size = min(sample_size, k)

        negatives_list = []
        for b in range(B):
            candidate_neg = topk_tensors[b]
            perm = torch.randperm(k, device=candidate_neg.device)
            chosen_idxs = perm[:sample_size]
            sampled_neg = candidate_neg[chosen_idxs]
            negatives_list.append(sampled_neg)
        final_negatives = torch.cat(negatives_list, dim=0)
        return final_negatives

    def clip_contrastive_loss(self, emb_anchor: torch.Tensor, emb_positive: torch.Tensor, memory_list) -> float:
        # if memory_list is empty, return 0.0
        if len(memory_list) < max(1, self.contrast_negatives // 2):
            return 0.0

        B, D = emb_anchor.shape
        pos_logit = (emb_anchor * emb_positive).sum(dim=1, keepdim=True) / self.contrast_temp

        if self.use_hard_neg:
            # sample 1000 topK hard negatives and then sample 256 from them
            big_k = 1000
            sample_size = min(self.contrast_negatives, len(memory_list))
            neg_tensors = self.gather_hard_negatives_two_stage(emb_anchor, memory_list, top_k=big_k, sample_size=sample_size)

            if neg_tensors is None:
                return 0.0
            
            neg_tensors = neg_tensors.detach()
            Bk = neg_tensors.shape[0]
            anchor_expanded = torch.repeat_interleave(emb_anchor, repeats=sample_size, dim=0)
            pos_logit_expanded = torch.repeat_interleave(pos_logit, repeats=sample_size, dim=0)
            neg_logit = (anchor_expanded * neg_tensors).sum(dim=1, keepdim=True) / self.contrast_temp
            logits = torch.cat([pos_logit_expanded, neg_logit], dim=1)
            labels = torch.zeros((Bk,), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)

            return loss
        else:
            # sample 256 random negatives
            sample_size = min(self.contrast_negatives, len(memory_list))
            neg_samples = random.sample(memory_list, sample_size)
            neg_tensors = torch.stack(neg_samples, dim=0).detach()
            neg_logits = emb_anchor.matmul(neg_tensors.t()) / self.contrast_temp
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

    def unnormalize_image(self, tensor):
        # unnormalize [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        return torch.clamp(tensor, 0, 1)

    def process_image(self, image):
        # resize and normalize image for PLIP
        resized = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        normalized = (resized - self.mean) / self.std
        return {"pixel_values": normalized}

    def forward(self):
        if self.opt.use_clip_contrast:
            images_real_B = self.unnormalize_image(self.real_B)
            input_real_B = self.process_image(images_real_B)
            self.embeddings_real_B = self.netPLIP_B.get_image_features(**input_real_B)

            images_real_A = self.unnormalize_image(self.real_A)
            inputs_real_A = self.process_image(images_real_A)
            self.embeddings_real_A = self.netPLIP_A.get_image_features(**inputs_real_A)

        if self.isTrain:
            if self.opt.use_clip_contrast:
                self.fake_B = self.netG_A(self.real_A, self.embeddings_real_A)
                self.fake_A = self.netG_B(self.real_B, self.embeddings_real_B)

                images_fake_B = self.unnormalize_image(self.fake_B)
                input_fake_B = self.process_image(images_fake_B)
                self.embeddings_fake_B = self.netPLIP_B.get_image_features(**input_fake_B)

                images_fake_A = self.unnormalize_image(self.fake_A)
                input_fake_A = self.process_image(images_fake_A)
                self.embeddings_fake_A = self.netPLIP_A.get_image_features(**input_fake_A)

                # Add Gaussian noise to fake images
                gaussian_noise = 0.01 * torch.randn(self.fake_A.size(), device=self.device)
                self.rec_A = self.netG_B(self.fake_B + gaussian_noise, self.embeddings_fake_B)
                self.rec_B = self.netG_A(self.fake_A + gaussian_noise, self.embeddings_fake_A)

                # Normalize embeddings
                self.embeddings_real_A = F.normalize(self.embeddings_real_A, dim=1)
                self.embeddings_real_B = F.normalize(self.embeddings_real_B, dim=1)
                self.embeddings_fake_A = F.normalize(self.embeddings_fake_A, dim=1)
                self.embeddings_fake_B = F.normalize(self.embeddings_fake_B, dim=1)
            else:
                self.fake_B = self.netG_A(self.real_A)
                self.fake_A = self.netG_B(self.real_B)

                # Add Gaussian noise to fake images
                gaussian_noise = 0.01 * torch.randn(self.fake_A.size(), device=self.device)
                self.rec_A = self.netG_B(self.fake_B + gaussian_noise)
                self.rec_B = self.netG_A(self.fake_A + gaussian_noise)
        else:
            if self.opt.use_clip_contrast:
                self.fake_B = self.netG_A(self.real_A, self.embeddings_real_A)
                self.fake_A = self.netG_B(self.real_B, self.embeddings_real_B)
            else:
                self.fake_B = self.netG_A(self.real_A)
                self.fake_A = self.netG_B(self.real_B)

    def zero_centered_gradient_penalty(self, samples, scores):
        gradients = torch.autograd.grad(
            outputs=scores.sum(), inputs=samples,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        # Compute total number of elements per sample (patches)
        num_elements = gradients.view(gradients.size(0), -1).size(1)
        return gradients.view(gradients.size(0), -1).pow(2).sum(1).mean() / num_elements

    def backward_D_basic(self, netD, real, fake, real_labels=None, fake_labels=None, gamma=3.0):
        if self.opt.netD == 'multi':
            real.requires_grad_(True)
            fake.requires_grad_(True)
            outputs_real = netD(real)
            outputs_fake = netD(fake)

            loss_total = 0
            for (pred1_real, pred2_real), (pred1_fake, pred2_fake) in zip(outputs_real, outputs_fake):
                # Classification loss:
                loss_class_real = self.criterionCE(pred1_real, real_labels) * 0.1
                loss_class_fake = self.criterionCE(pred1_fake, fake_labels) * 0.1

                # Adversarial loss on patch-level predictions using LSGAN:
                loss_adv_real = self.criterionGAN(pred2_real, True)
                loss_adv_fake = self.criterionGAN(pred2_fake, False)

                loss_scale = (loss_class_real + loss_class_fake + loss_adv_real + loss_adv_fake)

                # Gradient penalty:
                gp1 = (self.zero_centered_gradient_penalty(real, pred1_real) + self.zero_centered_gradient_penalty(fake, pred1_fake)) / 2
                gp2 = (self.zero_centered_gradient_penalty(real, pred2_real) + self.zero_centered_gradient_penalty(fake, pred2_fake)) / 2
                gp = (gp1 + gp2) / 2

                loss_total += loss_scale + (gamma * gp)

            loss_D_scalar = loss_total
            loss_D_scalar.backward()
            return loss_D_scalar
        else:
            # Real
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D

    def compute_G_loss(self, netD, fake, real_labels):
        outputs_fake = netD(fake)
        loss_total = 0

        for (pred1_fake, pred2_fake) in outputs_fake:
            loss_total += self.criterionCE(pred1_fake, real_labels) * 0.1
            loss_total += self.criterionGAN(pred2_fake, True)
        return loss_total

    def backward_D_A(self):
        if self.opt.netD == 'multi':
            fake_B, fake_labels = self.fake_B_pool.query(self.fake_B.detach(), self.real_A_class.detach())
            
            # fake labels are the last class in shape [B]
            fake_labels = fake_labels.view(-1)
            real_labels = self.real_B_class.view(-1)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, real_labels, fake_labels)
        else:
            fake_B = self.fake_B_pool.query(self.fake_B)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        if self.opt.netD == 'multi':
            fake_A, fake_labels = self.fake_A_pool.query(self.fake_A.detach(), self.real_B_class.detach())

            # fake labels are the last class in shape [B]
            fake_labels = fake_labels.view(-1)
            real_labels = self.real_A_class.view(-1)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, real_labels, fake_labels)
        else:
            fake_A = self.fake_A_pool.query(self.fake_A)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B, self.embeddings_real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A, self.embeddings_real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        if self.opt.netD == 'multi':
            self.loss_G_A = self.compute_G_loss(self.netD_A, self.fake_B, self.real_A_class)
            self.loss_G_B = self.compute_G_loss(self.netD_B, self.fake_A, self.real_B_class)
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle consistency loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_G = (self.loss_G_A + self.loss_G_B +
                       self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B)

        # Contrastive loss
        if self.opt.use_clip_contrast:
            clip_loss_AB = self.clip_contrastive_loss(
                self.embeddings_real_A,
                self.embeddings_fake_B,
                list(self.memory_bank_A_real) + list(self.memory_bank_A_fake)
            )
            clip_loss_BA = self.clip_contrastive_loss(
                self.embeddings_real_B,
                self.embeddings_fake_A,
                list(self.memory_bank_B_real) + list(self.memory_bank_B_fake)
            )
            self.loss_clip_contrast = self.lambda_clip_contrast * (clip_loss_AB + clip_loss_BA) / 2
            self.loss_G += self.loss_clip_contrast
        else:
            self.loss_clip_contrast = 0.0
            
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Update G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Update D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

        # Update memory bank
        if self.opt.use_clip_contrast:
            with torch.no_grad():
                B = self.embeddings_real_A.size(0)
                for i in range(B):
                    self.memory_bank_A_real.append(self.embeddings_real_A[i].clone().detach())
                    self.memory_bank_A_fake.append(self.embeddings_fake_B[i].clone().detach())
                    self.memory_bank_B_real.append(self.embeddings_real_B[i].clone().detach())
                    self.memory_bank_B_fake.append(self.embeddings_fake_A[i].clone().detach())

    def save_networks(self, epoch):
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

        if self.opt.use_clip_contrast:
            save_dir_A = os.path.join(self.save_dir, f'{epoch}_net_PLIP_A')
            if not os.path.exists(save_dir_A):
                os.makedirs(save_dir_A)
            self.netPLIP_A.save_pretrained(save_dir_A)
            save_dir_B = os.path.join(self.save_dir, f'{epoch}_net_PLIP_B')

    def load_networks(self, epoch):
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

        if self.opt.use_clip_contrast:
            load_dir_A = os.path.join(self.save_dir, f'{epoch}_net_PLIP_A')
            print(f'Loading PLIP_A from {load_dir_A}')
            self.netPLIP_A = CLIPModel.from_pretrained(load_dir_A).to(self.device)
