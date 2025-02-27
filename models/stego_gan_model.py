import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_stego as networks


class stegoganmodel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    Add the same forward pass to get rec_A as fake_A

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
            parser.add_argument('--lambda_reg', type=float, default=0.2)

            ## HYPER-PARAMETER FOR STEGO-GAN
            parser.add_argument('--lambda_consistency', type=float, default=3)
            parser.add_argument('--lambda_identity', type=float, default=0.01, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        parser.add_argument('--netG_A', type=str, default='resnet_9blocks_maskv1', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--netG_B', type=str, default='resnet_9blocks_maskv3', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--fusionblock', action='store_true', help='Extra blocks to fuse features')
        parser.add_argument('--mask_group', type=int, default=256, help='number of mask groups')
        parser.add_argument('--mask_detach', type=bool, default=False, help='if mask should be detached in training')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'consistency_B', 'regularization_G_B', 'consistency_feature']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'rec_A_clean', 'fake_B_clean', 'latent_fake_B_mask_upsampled']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'rec_B_clean', 'latent_real_B_mask_upsampled']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        print('the generator is:', opt.netG_A, opt.netG_B,  'the discriminator is:', opt.netD)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG_A, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, resnet_layer=opt.resnet_layer, fusionblock=opt.fusionblock) # add one channel as uncertainty
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG_B, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, out_dim=opt.mask_group, resnet_layer=opt.resnet_layer, fusionblock=opt.fusionblock)

        self.num_classes = 11

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids, num_classes=self.num_classes)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids, num_classes=self.num_classes)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCE = nn.CrossEntropyLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
        self.real_A = self.real_A[:, :, start:end, start:end]
        self.real_B = self.real_B[:, :, start:end, start:end]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_A, self.latent_real_B, self.latent_real_B_mask = self.netG_B(self.real_B)
        gaussian_noise = 0.01*torch.randn(self.fake_A.size())
        self.rec_B = self.netG_A(self.fake_A + gaussian_noise.cuda(), self.latent_real_B)
        self.rec_B_clean = self.netG_A(self.fake_A)
        
        self.fake_B_clean = self.netG_A(self.real_A)
        self.rec_A_clean, _, _ = self.netG_B(self.fake_B_clean)
        self.fake_B = self.netG_A(self.real_A, self.latent_real_B.detach())
        self.rec_A, self.latent_fake_B, self.latent_fake_B_mask = self.netG_B(self.fake_B)

        self.latent_real_B_mask_upsampled = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(self.latent_real_B_mask)
        self.latent_fake_B_mask_upsampled = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(self.latent_fake_B_mask)

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

        # Gradient penalty (R1 + R2)
        real_class_indices = real_labels.view(-1)
        preds_real_per_sample_sum = 0
        for pred_scale in preds_real:
            preds_real_per_sample_sum += pred_scale[torch.arange(pred_scale.size(0)), real_class_indices]

        grad_outputs_real = torch.ones_like(preds_real_per_sample_sum)
        gradients_real = torch.autograd.grad(
            outputs=preds_real_per_sample_sum,
            inputs=real,
            grad_outputs=grad_outputs_real,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradient_norms_real = gradients_real.view(gradients_real.size(0), -1).norm(2, dim=1)
        loss_gp_real = (gradient_norms_real ** 2).mean()

        fake_class_indices = fake_labels.view(-1)
        preds_fake_per_sample_sum = 0
        for pred_scale in preds_fake:
            preds_fake_per_sample_sum += pred_scale[torch.arange(pred_scale.size(0)), fake_class_indices]

        grad_outputs_fake = torch.ones_like(preds_fake_per_sample_sum)
        gradients_fake = torch.autograd.grad(
            outputs=preds_fake_per_sample_sum,
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
        fake_B = self.fake_B_pool.query(self.fake_B.detach())
        fake_labels = torch.full((fake_B.size(0),), self.num_classes-1, dtype=torch.long, device=self.device)
        real_labels = self.real_B_class.view(-1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, real_labels, fake_B, fake_labels)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        fake_labels = torch.full((fake_A.size(0),), self.num_classes-1, dtype=torch.long, device=self.device)
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
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _= self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        pred_fake_B = self.netD_A(self.fake_B)
        target_labels_B = self.real_A_class
        self.loss_G_A = self.calculate_disc_loss(pred_fake_B, target_labels_B)

        pred_fake_A = self.netD_B(self.fake_A)
        target_labels_A = self.real_B_class
        self.loss_G_B = self.calculate_disc_loss(pred_fake_A, target_labels_A)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A 
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Regularization loss
        lambda_reg = self.opt.lambda_reg
        self.loss_regularization_G_B = lambda_reg*torch.mean((self.latent_real_B_mask+1e-10)**0.5) + lambda_reg*torch.mean((self.latent_fake_B_mask+1e-10)**0.5) #default 0.2

        # Consistency loss of B
        lambda_consistency = self.opt.lambda_consistency

        self.loss_consistency_B = lambda_consistency*self.criterionCycle(self.rec_B_clean*(1-self.latent_real_B_mask_upsampled), \
                                                                    self.real_B*(1-self.latent_real_B_mask_upsampled))  + \
                                                                    lambda_consistency*self.criterionCycle(self.fake_B_clean*(1-self.latent_fake_B_mask_upsampled), \
                                                                    self.fake_B*(1-self.latent_fake_B_mask_upsampled))

        # Consistency loss of A
        self.loss_consistency_feature = lambda_consistency*self.criterionCycle(self.latent_fake_B, self.latent_real_B.detach())
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_regularization_G_B + self.loss_consistency_B
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
        self.optimizer_G.step()       # twice

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights