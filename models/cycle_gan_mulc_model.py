import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANMULCModel(BaseModel):
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

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

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
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCE = torch.nn.CrossEntropyLoss().to(self.device)
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
        self.real_A = self.real_A[:, :, start:end, start:end]
        self.real_B = self.real_B[:, :, start:end, start:end]

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

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
            self.netD_A, self.real_B, real_labels, fake_B, fake_labels
        )

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        fake_labels = torch.full((fake_A.size(0),), self.num_classes-1,
                                 dtype=torch.long, device=self.device)
        real_labels = self.real_A_class.view(-1)
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self.real_A, real_labels, fake_A, fake_labels
        )

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
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
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

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
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