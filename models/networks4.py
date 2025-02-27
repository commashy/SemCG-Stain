import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator

###############################################################################
# Helper Functions
###############################################################################


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample_res(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample_res(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, opt=opt)
    elif netG == 'smallstylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=2, opt=opt)
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst', nl_layer='relu')
    elif netG == 'vit-modnet':
        net = ViTModNetGenerator(input_shape= (3, 256, 256), output_shape=(3, 256, 256), features=384, n_heads=6, n_blocks=12, ffn_features=1536, embed_features=384, activ='gelu', norm='layer', modnet_features_list=[48, 96, 192, 384], modnet_activ='leakyrelu', modnet_norm=None, modnet_downsample='conv', modnet_upsample='upsample-conv', modnet_rezero=False, modnet_demod=True, rezero=True, activ_output='sigmoid', style_rezero=True, style_bias=True, n_ext=1)
    elif netG == 'attr2unet':
        net = R2AttU_Net(img_ch=3, output_ch=3, t=2)
    elif netG == 'unet':
        net = UNet(image_channels=3, n_channels=ngf)
    elif netG == 'unet2':
        net = CrossUNet(image_channels=3)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'mapping':
        net = MappingF(input_nc, gpu_ids=gpu_ids)
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None, num_classes=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'unet':
        net = UnetDiscriminator(input_nc, ndf)
    elif 'stylegan2' in netD:
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    elif 'basic2':
        net = NLayerDiscriminator2(input_nc, ndf, n_layers=2, norm_layer=norm_layer, no_antialias=no_antialias, num_classes=7)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        elif gan_mode == "hinge":
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        elif self.gan_mode == 'hinge':
            if target_is_real:
                minvalue = torch.min(prediction - 1, torch.zeros(prediction.shape).to(prediction.device))
                loss = -torch.mean(minvalue)
            else:
                minvalue = torch.min(-prediction - 1,torch.zeros(prediction.shape).to(prediction.device))
                loss = -torch.mean(minvalue)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)

class MappingF(nn.Module):
    def __init__(self, in_layer=4, gpu_ids=[], nc=256, patch_num=256, dim=64, init_type='normal', init_gain=0.02):
        # hard-coded code.
        super().__init__()
        self.init_type = init_type
        self.nc=nc
        self.dim=dim
        self.in_layer=in_layer
        self.patch_num = patch_num
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        avg = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(in_layer, dim, 3, stride=2)
        self.model = nn.Sequential(*[conv, nn.ReLU(), avg, nn.Flatten(), nn.Linear(dim,dim), nn.ReLU(), nn.Linear(dim, dim)])
        init_net(self.model, self.init_type, self.init_gain, self.gpu_ids)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(1, -1, self.patch_num, self.nc)
        x = self.model(x)
        x_norm = self.l2norm(x)
        return x_norm


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv2d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class G_Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon

##################################################################################
# Encoder and Decoders
##################################################################################


class E_adaIN(nn.Module):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        # style encoder
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def forward(self, image):
        style = self.enc_style(image)
        return style


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
        else:
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).view(x.size(0), -1)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x, nce_layers=[], encode_only=False):
        if len(nce_layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None

class Decoder_all(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [Upsample2(scale_factor=2), Conv2dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample2(scale_factor=2), Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)

##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class ResnetDecoder(nn.Module):
    """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based decoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if(no_antialias):
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetEncoder(nn.Module):
    """Resnet-based encoder that consists of a few downsampling + several Resnet blocks
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class NLayerDiscriminator2(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False, num_classes=7):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator2, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, num_classes, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(input)


class UnetDiscriminator(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    Arg:
        input_shape: Shape of the input.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, input_nc, num_feat=64, skip_connection=True):
        super(UnetDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.parametrizations.spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(input_nc, num_feat, kernel_size=3, stride=1, padding=1) # 64, 256, 256
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False)) # 128, 128, 128
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False)) # 256, 64, 64
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False)) # 512, 32, 32
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 16, 4, 2, 1, bias=False)) # 1024, 16, 16
        # upsample
        self.conv5 = norm(nn.ConvTranspose2d(num_feat * 16, num_feat * 8, 4, 2, 1, bias=False)) # 512, 32, 32
        self.conv6 = norm(nn.ConvTranspose2d(num_feat * 8, num_feat * 4, 4, 2, 1, bias=False)) # 256, 64, 64
        self.conv7 = norm(nn.ConvTranspose2d(num_feat * 4, num_feat * 2, 4, 2, 1, bias=False)) # 128, 128, 128
        self.conv8 = norm(nn.ConvTranspose2d(num_feat * 2, num_feat, 4, 2, 1, bias=False)) # 64, 256, 256
        # extra convolutions
        self.conv9 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)) # 64, 256, 256
        self.conv10 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)) # 64, 256, 256
        self.conv11 = nn.Conv2d(num_feat, 1, 3, 1, 1) # 1, 256, 256

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        # upsample
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x3
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x2
        x7 = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x7 = x7 + x1
        x8 = F.leaky_relu(self.conv8(x7), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x8 = x8 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv9(x8), negative_slope=0.2, inplace=True)    
        out = F.leaky_relu(self.conv10(out), negative_slope=0.2, inplace=True)
        out = self.conv11(out)

        return out, x4

class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)

# from .uvcgan.transformer import ExtendedPixelwiseViT
# from .uvcgan.modnet      import ModNet
# from .uvcgan.select      import get_activ_layer

# class ViTModNetGenerator(nn.Module):

#     def __init__(
#         self, features, n_heads, n_blocks, ffn_features, embed_features,
#         activ, norm, input_shape, output_shape, modnet_features_list,
#         modnet_activ,
#         modnet_norm       = None,
#         modnet_downsample = 'conv',
#         modnet_upsample   = 'upsample-conv',
#         modnet_rezero     = False,
#         modnet_demod      = True,
#         rezero            = True,
#         activ_output      = None,
#         style_rezero      = True,
#         style_bias        = True,
#         n_ext             = 1,
#         **kwargs
#     ):
#         # pylint: disable = too-many-locals
#         super().__init__(**kwargs)

#         assert input_shape == output_shape
#         image_shape = input_shape

#         self.image_shape = image_shape

#         mod_features = features * n_ext

#         self.net = ModNet(
#             modnet_features_list, modnet_activ, modnet_norm, image_shape,
#             modnet_downsample, modnet_upsample, mod_features, modnet_rezero,
#             modnet_demod, style_rezero, style_bias, return_mod = False
#         )

#         bottleneck = ExtendedPixelwiseViT(
#             features, n_heads, n_blocks, ffn_features, embed_features,
#             activ, norm,
#             image_shape = self.net.get_inner_shape(),
#             rezero      = rezero,
#             n_ext       = n_ext,
#         )

#         self.net.set_bottleneck(bottleneck)

#         self.output = get_activ_layer(activ_output)

#     def forward(self, x):
#         # x : (N, C, H, W)
#         result = self.net(x)
#         return self.output(result)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init

# class conv_block(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(conv_block,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )


#     def forward(self,x):
#         x = self.conv(x)
#         return x

# class up_conv(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(up_conv,self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
# 		    nn.BatchNorm2d(ch_out),
# 			nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.up(x)
#         return x

# class Recurrent_block(nn.Module):
#     def __init__(self,ch_out,t=2):
#         super(Recurrent_block,self).__init__()
#         self.t = t
#         self.ch_out = ch_out
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
# 		    nn.BatchNorm2d(ch_out),
# 			nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         for i in range(self.t):

#             if i==0:
#                 x1 = self.conv(x)
            
#             x1 = self.conv(x+x1)
#         return x1
        
# class RRCNN_block(nn.Module):
#     def __init__(self,ch_in,ch_out,t=2):
#         super(RRCNN_block,self).__init__()
#         self.RCNN = nn.Sequential(
#             Recurrent_block(ch_out,t=t),
#             Recurrent_block(ch_out,t=t)
#         )
#         self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

#     def forward(self,x):
#         x = self.Conv_1x1(x)
#         x1 = self.RCNN(x)
#         return x+x1


# class single_conv(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(single_conv,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.conv(x)
#         return x

# class Attention_block(nn.Module):
#     def __init__(self,F_g,F_l,F_int):
#         super(Attention_block,self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(F_int)
#             )
        
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self,g,x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1+x1)
#         psi = self.psi(psi)

#         return x*psi


# class U_Net(nn.Module):
#     def __init__(self,img_ch=3,output_ch=1):
#         super(U_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
#         self.Conv2 = conv_block(ch_in=64,ch_out=128)
#         self.Conv3 = conv_block(ch_in=128,ch_out=256)
#         self.Conv4 = conv_block(ch_in=256,ch_out=512)
#         self.Conv5 = conv_block(ch_in=512,ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)
        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


# class R2U_Net(nn.Module):
#     def __init__(self,img_ch=3,output_ch=1,t=2):
#         super(R2U_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

#         self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
#         self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
#         self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
#         self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.RRCNN1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.RRCNN2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.RRCNN3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.RRCNN4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.RRCNN5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_RRCNN5(d5)
        
#         d4 = self.Up4(d5)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_RRCNN4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_RRCNN3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_RRCNN2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1



# class AttU_Net(nn.Module):
#     def __init__(self,img_ch=3,output_ch=1):
#         super(AttU_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
#         self.Conv2 = conv_block(ch_in=64,ch_out=128)
#         self.Conv3 = conv_block(ch_in=128,ch_out=256)
#         self.Conv4 = conv_block(ch_in=256,ch_out=512)
#         self.Conv5 = conv_block(ch_in=512,ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4,x=x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3,x=x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2,x=x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


# class R2AttU_Net(nn.Module):
#     def __init__(self,img_ch=3,output_ch=3,t=2):
#         super(R2AttU_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

#         self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
#         self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
#         self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
#         self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

#         self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.RRCNN1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.RRCNN2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.RRCNN3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.RRCNN4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.RRCNN5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(g=d5,x=x4)
#         d5 = torch.cat((x4,d5),dim=1)
#         d5 = self.Up_RRCNN5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(g=d4,x=x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_RRCNN4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3,x=x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_RRCNN3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2,x=x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_RRCNN2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1


"""
DDPM unet
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn


class Swish(nn.Module):
    """
    ### Swish activation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)

from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x


# class ConvNeXtV2Block(nn.Module):
#     """ ConvNeXtV2 Block.
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#     """
#     def __init__(self, dim, drop_path=0.):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.grn = GRN(4 * dim)
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.grn(x)
#         x = self.pwconv2(x)
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x


class ResidualBlockwoc(nn.Module):
    """
    Residual block with image embedding
    """

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        # self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.SiLU()
        # self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))

        # self.norm2 = nn.GroupNorm(n_groups, in_channels)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.act2 = nn.SiLU()
        # self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        _ = emb
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        # h = self.act1(self.conv1(self.norm1(x)))
        # h = self.act2(self.conv2(self.norm2(h)))
        # h = self.conv1(self.act1(self.norm1(x)))
        # h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class ResidualBlock(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.emb_norm = nn.InstanceNorm2d(out_channels)
        self.emb_gamma = nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding='same')
        self.emb_beta = nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding='same')
        self.emb_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        emb = F.interpolate(emb, size=h.size()[2:], mode='nearest')
        emb = self.emb_act(emb)
        gamma = self.emb_gamma(emb)
        beta = self.emb_beta(emb)
        h = self.emb_norm(h)
        h = h * (1 + gamma) + beta
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        # self.norm = nn.GroupNorm(n_groups, n_channels)
        self.norm = nn.InstanceNorm2d(n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlockwoc(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


# class MiddleBlock(nn.Module):
#     """
#     ### Middle block

#     It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
#     This block is applied at the lowest resolution of the U-Net.
#     """

#     def __init__(self, n_channels: int, time_channels: int):
#         super().__init__()
#         self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
#         self.attn = AttentionBlock(n_channels)
#         self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

#     def forward(self, x: torch.Tensor, t: torch.Tensor):
#         x = self.res1(x, t)
#         x = self.attn(x)
#         x = self.res2(x, t)
#         return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, global_channels: int):
        super().__init__()
        self.res1 = ResidualBlockwoc(n_channels, n_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlockwoc(n_channels, n_channels)

    def forward(self, x: torch.Tensor, x_global: torch.Tensor):
        x = self.res1(x, x_global)
        x = self.attn(x)
        x = self.res2(x, x_global)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        # self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        self.conv = nn.Conv2d(n_channels, n_channels * 4, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        x = self.conv(x)
        return self.pixel_shuffle(x)
        # return self.conv(x)

class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        _ = t
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)

class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    Modified U-Net for image-to-image translation using image embeddings.
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 1, 2, 2),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 1, embed_dim: int = 256):
        """
        Args:
            image_channels: Number of channels in the input image.
            n_channels: Base number of channels for the U-Net.
            ch_mults: Multipliers for the number of channels at each resolution.
            is_attn: A list of booleans indicating where to use attention.
            n_blocks: Number of residual blocks at each resolution.
            embed_dim: The dimensionality of the image embedding.
        """
        super().__init__()

        self.image_channels = image_channels
        self.n_channels = n_channels

        # Image embedding layer (replaces the time embedding)
        self.emb_norm = nn.InstanceNorm2d(image_channels)
        self.emb_act = nn.SiLU()
        self.image_emb = nn.Conv2d(image_channels, n_channels, kernel_size=3, stride=3)
        # self.image_emb = ImageEmbedding1(64, embed_dim)

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Downsampling blocks
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, embed_dim, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, embed_dim)

        # Upsampling blocks
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, embed_dim, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, embed_dim, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        # self.norm = nn.GroupNorm(8, n_channels)
        # self.act = Swish()
        # self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

        # self.final_block1 = ResidualBlockwoc(in_channels, image_channels, n_groups=8)
        self.final_block1 = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.final_block2 = ResidualBlockwoc(image_channels, image_channels, n_groups=8)

        # Define a learnable embedding for when `full=False`
        # self.learned_emb = nn.Parameter(torch.randn(1, self.n_channels * 4))

    def forward(self, x_large: torch.Tensor, full=False):
        """
        Args:
            x_large: The larger input patch (e.g., 768x768).
        """
        # Step 1: Generate image embedding from the larger input
        if full:
            x_large = self.emb_norm(x_large)
            # emb = self.emb_act(x_large)
            emb = self.image_emb(x_large)
            emb = self.emb_act(emb)
        else:
            # emb = torch.zeros(x_large.size(0), self.n_channels * 4, device=x_large.device)  # Zero embedding
            fake_x_large = F.interpolate(x_large, scale_factor=3, mode='nearest')
            fake_x_large = self.emb_norm(fake_x_large)
            emb = self.image_emb(fake_x_large)
            emb = self.emb_act(emb)

        # Step 2: Extract the center 256x256 patch from the larger 768x768 input
        if full:
            center_size = 256
            large_size = x_large.shape[2]
            start = (large_size - center_size) // 2
            end = start + center_size
            x_center = x_large[:, :, start:end, start:end]
        else:
            x_center = x_large

        # Step 3: Project the center patch into feature space
        x = self.image_proj(x_center)

        # Save skip connections from the first half of the U-Net
        h = [x]

        # Apply downsampling blocks
        for m in self.down:
            x = m(x, emb)
            h.append(x)

        # Middle block
        x = self.middle(x, emb)

        # Apply upsampling blocks with skip connections
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, emb)
            else:
                # Get the skip connection from the first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, emb)

        # Final normalization and output
        # return self.final(self.act(self.norm(x)))
        return self.final_block1(x)


# class embed_fc(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(embed_fc, self).__init__()
#         self.fc = nn.Linear(in_channels, out_channels)
#     def forward(self, x, emb):
#         return x + self.fc(emb)[:, :, None, None]


# class ResnetGenerator2(nn.Module):
#     """Resnet-based generator with large view image embedding using custom residual blocks."""

#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
#                  n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, 
#                  embed_dim=256, use_image_embedding=True, opt=None):
#         """Construct a Resnet-based generator with optional large image view embedding.

#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#             embed_dim (int)     -- dimension of the image embedding
#             use_image_embedding (bool) -- whether to use large view image embedding
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator2, self).__init__()
#         self.opt = opt
#         self.use_image_embedding = use_image_embedding

#         if use_image_embedding:
#             self.image_embedding = ImageEmbedding2(input_nc, embed_dim)

#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         # Initial convolution
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]

#         # Downsampling layers
#         n_downsampling = 2
#         for i in range(n_downsampling):
#             mult = 2 ** i
#             if(no_antialias):
#                 model += [embed_fc(embed_dim, ngf * mult),
#                           nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                           norm_layer(ngf * mult * 2),
#                           nn.ReLU(True)]
#             else:
#                 model += [embed_fc(embed_dim, ngf * mult),
#                           nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
#                           norm_layer(ngf * mult * 2),
#                           nn.ReLU(True),
#                           Downsample(ngf * mult * 2)]

#         # Custom Residual blocks with embedding support
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):
#             # model += [ResidualBlock(ngf * mult, ngf * mult, embed_dim=embed_dim)]
#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

#         # Upsampling layers
#         for i in range(n_downsampling):
#             mult = 2 ** (n_downsampling - i)
#             if no_antialias_up:
#                 model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                              kernel_size=3, stride=2,
#                                              padding=1, output_padding=1,
#                                              bias=use_bias),
#                           norm_layer(int(ngf * mult / 2)),
#                           nn.ReLU(True)]
#             else:
#                 model += [Upsample(ngf * mult),
#                           nn.Conv2d(ngf * mult, int(ngf * mult / 2),
#                                     kernel_size=3, stride=1,
#                                     padding=1,
#                                     bias=use_bias),
#                           norm_layer(int(ngf * mult / 2)),
#                           nn.ReLU(True)]

#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]

#         self.model = nn.Sequential(*model)

#         # Define a learnable embedding for when full=False
#         self.learned_emb = nn.Parameter(torch.randn(1, ngf * 4))

#     def forward(self, input, layers=[], encode_only=False, full=False):
#         """Forward pass with optional large view image embedding.

#         Args:
#             input (torch.Tensor): The input image tensor.
#             layers (list): List of layers to return intermediate features from.
#             encode_only (bool): Whether to return only encoded features.
#             full (bool): Whether to condition the model with the large view image or learned embedding.
#         """
#         # Always condition on large view image or learned embedding
#         if full:
#             emb = self.image_embedding(input)

#             # Extract center region for conditioning
#             center_size = 256
#             large_size = input.shape[2]
#             start = (large_size - center_size) // 2
#             end = start + center_size
#             input = input[:, :, start:end, start:end]  # Adjust input to use center region
#         else:
#             emb = self.learned_emb.expand(input.size(0), -1)  # Use learned embedding for conditioning

#         # If layers are provided, we extract intermediate features
#         if -1 in layers:
#             layers.append(len(self.model))

#         feat = input
#         feats = []

#         for layer_id, layer in enumerate(self.model):
#             # If the layer is a ResidualBlock, pass the embedding
#             if isinstance(layer, embed_fc):
#                 feat = layer(feat, emb)
#             else:
#                 feat = layer(feat)

#             if layer_id in layers:
#                 feats.append(feat)

#             # Add a check to ensure we only access `layers[-1]` if layers is not empty
#             if len(layers) > 0 and layer_id == layers[-1] and encode_only:
#                 return feats  # Return intermediate features alone if requested

#         # If no layers provided, return the final output
#         if len(layers) == 0:
#             fake = feat  # Final output
#             return fake
#         else:
#             return feat, feats  # Return both final output and intermediate features if layers were specified

class CrossAttentionBlock(nn.Module):
    def __init__(self, n_channels, n_global_channels, n_heads=1, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = n_channels // n_heads
        self.n_heads = n_heads
        self.d_k = d_k

        # Linear projections
        self.query_proj = nn.Conv2d(n_channels, n_heads * d_k, kernel_size=1)
        self.key_proj = nn.Conv2d(n_global_channels, n_heads * d_k, kernel_size=1)
        self.value_proj = nn.Conv2d(n_global_channels, n_heads * d_k, kernel_size=1)
        self.output_proj = nn.Conv2d(n_heads * d_k, n_channels, kernel_size=1)

        self.scale = d_k ** -0.5

    def forward(self, x_local, x_global):
        batch_size, c, h, w = x_local.shape
        _, c_global, h_global, w_global = x_global.shape

        # Flatten spatial dimensions
        q = self.query_proj(x_local).view(batch_size, self.n_heads, self.d_k, -1)  # [B, H, D_k, N]
        k = self.key_proj(x_global).view(batch_size, self.n_heads, self.d_k, -1)   # [B, H, D_k, M]
        v = self.value_proj(x_global).view(batch_size, self.n_heads, self.d_k, -1) # [B, H, D_k, M]

        # Transpose for matmul
        q = q.permute(0, 1, 3, 2)  # [B, H, N, D_k]
        k = k.permute(0, 1, 2, 3)  # [B, H, D_k, M]
        v = v.permute(0, 1, 3, 2)  # [B, H, M, D_k]

        # Compute attention scores
        attn_scores = torch.matmul(q, k) * self.scale  # [B, H, N, M]
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [B, H, N, D_k]
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, h, w)  # [B, H * D_k, H_local, W_local]

        # Project back to original dimension
        output = self.output_proj(attn_output)

        # Residual connection
        output = output + x_local

        return output

class ResidualBlockWithCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, n_global_channels, n_heads=1, n_groups=32, dropout=0.1, attn=True):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = AttentionBlock(out_channels, n_heads=n_heads*2)
        else:
            self.attn = nn.Identity()
        self.cross_attn = CrossAttentionBlock(out_channels, n_global_channels, n_heads=n_heads)

    def forward(self, x, x_global):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))

        # h = self.attn(self.attn_norm1(h)) + h
        # h = self.cross_attn(self.attn_norm2(h), x_global) + h
        # h = self.FF(self.FF_norm(h)) + h
        if self.attn:
            h = self.attn(h) + h
        else:
            h = h
        h = self.cross_attn(h, x_global) + h

        return h + self.shortcut(x)

class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int, n_global_channels: int, n_heads=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.cross_attn = CrossAttentionBlock(out_channels, n_global_channels, n_heads=n_heads)

        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.act3 = nn.SiLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm4 = nn.InstanceNorm2d(out_channels)
        self.act4 = nn.SiLU()
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.dropout2 = nn.Dropout(dropout)
        

    def forward(self, x: torch.Tensor, x_global: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(x)))
        
        h = self.cross_attn(h, x_global) + h
        x = h + self.shortcut(x)

        h = self.conv3(self.act3(self.norm3(x)))
        h = self.conv4(self.dropout2(self.act4(self.norm4(h))))

        return h + self.shortcut2(x)

import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The number of channels (dimension) of the positional embedding.
                         Should be divisible by 4.
        """
        super(PositionalEncoding2D, self).__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4.")
        self.channels = channels

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Positional encoding tensor of the same shape.
        """
        batch_size, _, height, width = x.size()
        device = x.device
        pe = torch.zeros(batch_size, self.channels, height, width, device=device)
        channels = self.channels // 2

        div_term = torch.exp(
            torch.arange(0., channels, 2, device=device) * -(math.log(10000.0) / channels))
        pos_w = torch.arange(0., width, device=device).unsqueeze(1)
        pos_h = torch.arange(0., height, device=device).unsqueeze(1)

        pe_h = torch.zeros(channels, height, 1, device=device)
        pe_w = torch.zeros(channels, 1, width, device=device)

        pe_h[0::2, :, 0] = torch.sin(pos_h * div_term).transpose(0, 1)
        pe_h[1::2, :, 0] = torch.cos(pos_h * div_term).transpose(0, 1)

        pe_w[0::2, 0, :] = torch.sin(pos_w * div_term).transpose(0, 1)
        pe_w[1::2, 0, :] = torch.cos(pos_w * div_term).transpose(0, 1)

        pe[:, :channels, :, :] = pe_h.repeat(batch_size, 1, 1, width)
        pe[:, channels:, :, :] = pe_w.repeat(batch_size, 1, height, 1)
        return pe


# class CrossUNet(nn.Module):
#     def __init__(self, image_channels=3, n_channels=64,
#                  ch_mults=(1, 1, 2, 2),
#                  is_attn=(False, False, True, True),
#                  n_blocks=1, embed_dim=256):
#         super().__init__()

#         self.image_channels = image_channels
#         self.n_channels = n_channels

#         self.initial_conv = nn.Conv2d(image_channels, 4, kernel_size=3, padding=1)

#         # Positional Encoding for the large input
#         self.positional_encoding = PositionalEncoding2D(channels=4)

#         # Encoder for the large view
#         self.global_encoder = nn.Sequential(
#             nn.InstanceNorm2d(4),
#             nn.SiLU(),
#             nn.Conv2d(4, n_channels, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(n_channels),
#             nn.SiLU(),
#             nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(n_channels * 2),
#             nn.SiLU(),
#             nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(n_channels * 4),
#             nn.SiLU(),
#             nn.Conv2d(n_channels * 4, n_channels * 4, kernel_size=4, stride=2, padding=1),
#         )

#         # Project image into feature map
#         self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)

#         # Downsampling blocks
#         down = []
#         out_channels = in_channels = n_channels
#         for i in range(len(ch_mults)):
#             out_channels = in_channels * ch_mults[i]
#             for _ in range(n_blocks):
#                 down.append(ResidualBlockWithCrossAttention(
#                     in_channels, out_channels, n_channels * 4, n_heads=4, attn=is_attn[i]))
#                 in_channels = out_channels
#             if i < len(ch_mults) - 1:
#                 down.append(Downsample(in_channels))

#         self.down = nn.ModuleList(down)

#         # Middle block
#         self.middle = MiddleBlock(out_channels, out_channels, n_channels*4, n_heads=4)

#         # Upsampling blocks
#         up = []
#         in_channels = out_channels
#         for i in reversed(range(len(ch_mults))):
#             out_channels = in_channels
#             for _ in range(n_blocks):
#                 up.append(ResidualBlockWithCrossAttention(
#                     in_channels + out_channels, out_channels, n_channels * 4, n_heads=4, attn=is_attn[i]))
#             out_channels = in_channels // ch_mults[i]
#             up.append(ResidualBlockWithCrossAttention(
#                 in_channels + out_channels, out_channels, n_channels * 4, n_heads=4, attn=is_attn[i]))
#             in_channels = out_channels
#             if i > 0:
#                 up.append(Upsample(in_channels))

#         self.up = nn.ModuleList(up)

#         # Final convolution layer
#         self.out = nn.Sequential(
#             nn.InstanceNorm2d(in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, image_channels, 3, padding=1),
#         )

#     def forward(self, x_large, full=False):
#         # Step 1: Add positional embeddings to x_large
#         x_large_proj = self.initial_conv(x_large)
#         pe_large = self.positional_encoding(x_large_proj)
#         x_large_with_pe = x_large_proj + pe_large

#         # Generate image embedding from the larger input with positional embeddings
#         if full:
#             x_global = self.global_encoder(x_large_with_pe)
#             x_global = nn.AdaptiveAvgPool2d((8, 8))(x_global)
#         else:
#             fake_x_large = F.interpolate(x_large_with_pe, scale_factor=3, mode='nearest')
#             x_global = self.global_encoder(fake_x_large)
#             x_global = nn.AdaptiveAvgPool2d((8, 8))(x_global)

#         # # Step 1: Generate image embedding from the larger input
#         # if full:
#         #     x_global = self.global_encoder(x_large)
#         #     x_global = nn.AdaptiveAvgPool2d((8, 8))(x_global)
#         # else:
#         #     fake_x_large = F.interpolate(x_large, scale_factor=3, mode='nearest')
#         #     x_global = self.global_encoder(fake_x_large)
#         #     x_global = nn.AdaptiveAvgPool2d((8, 8))(x_global)

#         # Step 2: Extract the center 256x256 patch from the larger 768x768 input
#         if full:
#             center_size = 256
#             large_size = x_large.shape[2]
#             start = (large_size - center_size) // 2
#             end = start + center_size
#             x_center = x_large[:, :, start:end, start:end]
#         else:
#             x_center = x_large

#         # Project the center patch
#         x = self.image_proj(x_center)
#         h = [x]

#         # Downsampling blocks
#         for m in self.down:
#             if isinstance(m, Downsample):
#                 x = m(x, None)
#                 h.append(x)
#             else:
#                 x = m(x, x_global)
#                 h.append(x)

#         # Middle block
#         x = self.middle(x, x_global)

#         # Upsampling blocks
#         for m in self.up:
#             if isinstance(m, Upsample):
#                 x = m(x, None)
#             else:
#                 s = h.pop()
#                 x = torch.cat((x, s), dim=1)
#                 x = m(x, x_global)

#         # Final output
#         x = self.out(x)
#         return x
# # Attempt to import FlashAttention
# try:
#     from flash_attn import flash_attn_func
#     print("FlashAttention imported successfully!")
# except ImportError:
#     flash_attn_func = None
#     print("FlashAttention not found. Using standard attention.")

# class CrossAttention(nn.Module):
#     use_flash_attention: bool = True  # Class variable to control the use of FlashAttention

#     def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int = None):
#         super().__init__()
#         self.n_heads = n_heads
#         self.d_head = d_head
#         self.scale = d_head ** -0.5

#         # Query projection
#         self.to_q = nn.Linear(d_model, n_heads * d_head, bias=False)

#         # Key and value projections (conditioned if d_cond is provided)
#         kv_dim = d_cond if d_cond is not None else d_model
#         self.to_k = nn.Linear(kv_dim, n_heads * d_head, bias=False)
#         self.to_v = nn.Linear(kv_dim, n_heads * d_head, bias=False)

#         self.to_out = nn.Linear(n_heads * d_head, d_model)

#     def forward(self, x, cond=None):
#         b, n, _ = x.shape

#         q = self.to_q(x)  # [b, n, n_heads * d_head]
#         if cond is None:
#             # Self-attention
#             k = self.to_k(x)
#             v = self.to_v(x)
#         else:
#             # Cross-attention
#             k = self.to_k(cond)
#             v = self.to_v(cond)

#         # Reshape for multi-head attention
#         # q, k, v: [b, seq_len, n_heads, d_head]
#         q = q.view(b, -1, self.n_heads, self.d_head)
#         k = k.view(b, -1, self.n_heads, self.d_head)
#         v = v.view(b, -1, self.n_heads, self.d_head)

#         # Determine whether to use FlashAttention
#         if (
#             self.use_flash_attention
#             and flash_attn_func is not None
#             and self.d_head <= 128  # FlashAttention supports head_dim up to 128
#             and x.is_cuda  # FlashAttention requires CUDA
#             and q.dtype in (torch.float16, torch.bfloat16)  # FlashAttention supports float16 and bfloat16
#         ):
#             # Use FlashAttention
#             print("Using FlashAttention!")
#             return self.flash_attention(q, k, v)
#         else:
#             # Use standard attention
#             return self.normal_attention(q, k, v)

#     def flash_attention(self, q, k, v):
#         # q, k, v: [b, seq_len, n_heads, d_head]
#         # Ensure tensors are contiguous and have correct dtype
#         q, k, v = map(lambda t: t.contiguous(), (q, k, v))

#         # FlashAttention expects q, k, v of shape [b, seqlen, nheads, headdim]
#         attn_output = flash_attn_func(
#             q, k, v,
#             dropout_p=0.0,
#             softmax_scale=self.scale,
#             causal=False  # Set to True if using causal attention
#         )

#         # attn_output: [b, seqlen_q, n_heads, d_head]
#         # Reshape to [b, seqlen_q, n_heads * d_head]
#         attn_output = attn_output.reshape(q.shape[0], q.shape[1], -1)
#         return self.to_out(attn_output)

#     def normal_attention(self, q, k, v):
#         # q, k, v: [b, seq_len, n_heads, d_head]
#         b, seqlen_q, _, _ = q.shape
#         seqlen_k = k.shape[1]

#         # Transpose for batched matrix multiplication
#         # q, k: [b, n_heads, seq_len, d_head]
#         q = q.permute(0, 2, 1, 3)
#         k = k.permute(0, 2, 1, 3)
#         v = v.permute(0, 2, 1, 3)

#         # Scaled dot-product attention
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [b, n_heads, seqlen_q, seqlen_k]
#         attn_probs = F.softmax(attn_scores, dim=-1)

#         attn_output = torch.matmul(attn_probs, v)  # [b, n_heads, seqlen_q, d_head]

#         # Transpose back and reshape
#         attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
#         attn_output = attn_output.view(b, seqlen_q, -1)  # [b, seqlen_q, n_heads * d_head]

#         return self.to_out(attn_output)

# class FeedForward(nn.Module):
#     def __init__(self, d_model: int, d_mult: int = 4, dropout: float = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, d_model * d_mult * 2),
#             GeGLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * d_mult, d_model)
#         )

#     def forward(self, x: torch.Tensor):
#         return self.net(x)

# class GeGLU(nn.Module):
#     def forward(self, x):
#         x, gate = x.chunk(2, dim=-1)
#         return x * F.gelu(gate)

# class BasicTransformerBlock(nn.Module):
#     def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int = None, dropout: float = 0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(d_model)
#         self.attn1 = CrossAttention(d_model, n_heads, d_head)  # Self-attention

#         self.norm2 = nn.LayerNorm(d_model)
#         self.attn2 = CrossAttention(d_model, n_heads, d_head, d_cond=d_cond)  # Cross-attention

#         self.norm3 = nn.LayerNorm(d_model)
#         self.ff = FeedForward(d_model, dropout=dropout)

#     def forward(self, x, cond=None):
#         x = x + self.attn1(self.norm1(x))
#         x = x + self.attn2(self.norm2(x), cond=cond)
#         x = x + self.ff(self.norm3(x))
#         return x

# class SpatialTransformer(nn.Module):
#     def __init__(self, channels: int, n_heads: int, d_head: int, n_layers: int, d_cond: int = None):
#         super().__init__()
#         # self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
#         self.norm = nn.InstanceNorm2d(channels)
#         self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)

#         self.transformer_blocks = nn.ModuleList([
#             BasicTransformerBlock(channels, n_heads, d_head, d_cond=d_cond)
#             for _ in range(n_layers)
#         ])

#         self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

#     def forward(self, x, cond=None):
#         b, c, h, w = x.shape
#         x_in = x
#         x = self.norm(x)
#         x = self.proj_in(x)
#         x = x.view(b, c, h * w).permute(0, 2, 1)  # [b, hw, c]
#         for block in self.transformer_blocks:
#             x = block(x, cond)
#         x = x.permute(0, 2, 1).view(b, c, h, w)
#         x = self.proj_out(x)
#         return x + x_in

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(32, in_channels)
#         self.act1 = nn.SiLU()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

#         # self.norm2 = nn.GroupNorm(32, out_channels)
#         self.norm2 = nn.InstanceNorm2d(out_channels)
#         self.act2 = nn.SiLU()
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
#         else:
#             self.shortcut = nn.Identity()

#     def forward(self, x):
#         h = self.conv1(self.act1(self.norm1(x)))
#         h = self.conv2(self.act2(self.norm2(h)))
#         return h + self.shortcut(x)

# class TransformerBlock(nn.Module):
#     def __init__(self, channels, n_heads, d_head, n_layers, d_cond=None):
#         super().__init__()
#         self.transformer = SpatialTransformer(
#             channels, n_heads, d_head, n_layers, d_cond=d_cond
#         )

#     def forward(self, x, cond=None):
#         return self.transformer(x, cond)

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Any

# Attempt to import xformers
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
    print("xformers imported successfully!")
except ImportError:
    XFORMERS_IS_AVAILABLE = False
    print("xformers not found. Using standard attention.")

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int = None, dropout: float = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        # Query projection
        self.to_q = nn.Linear(d_model, n_heads * d_head, bias=False)

        # Key and value projections (conditioned if d_cond is provided)
        context_dim = d_cond if d_cond is not None else d_model
        self.to_k = nn.Linear(context_dim, n_heads * d_head, bias=False)
        self.to_v = nn.Linear(context_dim, n_heads * d_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_head, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond=None, mask=None):
        b, n, _ = x.shape

        q = self.to_q(x)  # [b, n, n_heads * d_head]
        context = cond if cond is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # [b, n_heads, n, d_head]
        k = k.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)  # [b, n_heads, seq_len_k, d_head]
        v = v.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)  # [b, n_heads, seq_len_k, d_head]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [b, n_heads, n, seq_len_k]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # [b, n_heads, n, d_head]

        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.n_heads * self.d_head)
        return self.to_out(attn_output)

class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int = None, dropout: float = 0.0):
        super().__init__()
        inner_dim = d_head * n_heads
        context_dim = d_cond if d_cond is not None else d_model

        self.n_heads = n_heads
        self.d_head = d_head

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None  # Can be used to specify xformers attention operation

    def forward(self, x, cond=None, mask=None):
        if not XFORMERS_IS_AVAILABLE:
            raise ImportError("xformers is not available, cannot use MemoryEfficientCrossAttention")

        q = self.to_q(x)
        context = cond if cond is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape

        # Reshape q, k, v for xformers
        q = q.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)  # [b, n_heads, seq_len_q, d_head]
        k = k.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)  # [b, n_heads, seq_len_k, d_head]
        v = v.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)  # [b, n_heads, seq_len_k, d_head]

        # Flatten the batch and heads
        q = q.reshape(b * self.n_heads, -1, self.d_head)
        k = k.reshape(b * self.n_heads, -1, self.d_head)
        v = v.reshape(b * self.n_heads, -1, self.d_head)

        # Use xformers memory efficient attention
        attn_output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # Reshape back to [b, n_heads, seq_len_q, d_head]
        attn_output = attn_output.view(b, self.n_heads, -1, self.d_head)

        # Transpose and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, -1, self.n_heads * self.d_head)

        return self.to_out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_mult: int = 4, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * d_mult * 2),
            GeGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "memory_efficient": MemoryEfficientCrossAttention
    }

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int = None, dropout: float = 0., self_attn: bool = False, cross: bool = True):
        super().__init__()
        attn_mode = "memory_efficient" if XFORMERS_IS_AVAILABLE else "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.cross = cross
        self.self_attn = self_attn
        if self.self_attn:
            self.norm1 = nn.LayerNorm(d_model)
            self.attn1 = attn_cls(d_model, n_heads, d_head, dropout=dropout)  # Self-attention

        if self.cross:
            self.norm2 = nn.LayerNorm(d_model)
            self.attn2 = attn_cls(d_model, n_heads, d_head, d_cond=d_cond, dropout=dropout)  # Cross-attention
        else:
            self.norm2 = nn.Identity()
            self.attn2 = nn.Identity()

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, cond=None):
        if self.self_attn:
            x = x + self.attn1(self.norm1(x))
        if self.cross:
            x = x + self.attn2(self.norm2(x), cond=cond)
        else:
            x = self.attn2(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, d_head: int, n_layers: int, d_cond: int = None, self_attn: bool = False, cross: bool = True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(channels, n_heads, d_head, d_cond=d_cond, self_attn=self_attn, cross=cross)
            for _ in range(n_layers)
        ])

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)  # [b, hw, c]
        for block in self.transformer_blocks:
            x = block(x, cond)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        x = self.proj_out(x)
        return x + x_in

class SPADE(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channels)
        
        nhidden = 128
        ks = 3
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(3, nhidden, kernel_size=ks, padding=pw),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, n_channels, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, n_channels, kernel_size=ks, padding=pw)

    def forward(self, x, emb):
        normalized = self.norm(x)

        emb = F.interpolate(emb, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(emb)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm1 = SPADE(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # self.norm2 = nn.GroupNorm(32, out_channels)
        self.norm2 = SPADE(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            # self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            self.shortcut = nn.Identity()

    def forward(self, x, cond=None):
        h = self.conv1(self.act1(self.norm1(x, cond)))
        h = self.conv2(self.act2(self.norm2(h, cond)))
        return h + self.shortcut(x)

class TransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, d_head, n_layers, d_cond=None, self_attn=False, cross=True):
        super().__init__()
        self.transformer = SpatialTransformer(
            channels, n_heads, d_head, n_layers, d_cond=d_cond, self_attn=self_attn, cross=cross
        )

    def forward(self, x, cond=None):
        return self.transformer(x, cond)

# Ensure that in your ResidualBlockWithOptionalTransformer, you are using the updated SpatialTransformer
class ResidualBlockWithOptionalTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, use_transformer=False,
                 n_heads=1, d_head=64, n_layers=1, d_cond=None, image_resolution=(16, 16), self_attn=False, cross=True):
        super().__init__()
        self.use_transformer = use_transformer
        self.res_block = ResidualBlock(in_channels, out_channels)
        # self.channel_attn = ChannelAttention(out_channels)
        self.res_block2 = ResidualBlock(out_channels, out_channels)
        if use_transformer:
            self.transformer_block = TransformerBlock(
                out_channels, n_heads, d_head, n_layers, d_cond=d_cond, self_attn=self_attn, cross=cross
            )
        else:
            self.transformer_block = None

    def forward(self, x, cond=None, cond2=None):
        h = self.res_block(x, cond2)
        # h = self.channel_attn(h) * h
        h = self.res_block2(h, cond2)
        if self.use_transformer:
            h = self.transformer_block(h, cond) + h
        return h

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Ensure intermediate_planes is at least 1
        intermediate_planes = max(in_planes // ratio, 1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, intermediate_planes, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(intermediate_planes, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, bias=False):
        super(SpatialAttention, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

# class ResidualBlockWithOptionalTransformer(nn.Module):
#     def __init__(self, in_channels, out_channels, use_transformer=False,
#                  n_heads=1, d_head=64, n_layers=1, d_cond=None, image_resolution=(16, 16)):
#         super().__init__()
#         self.use_transformer = use_transformer
#         self.res_block = ResidualBlock(in_channels, out_channels)
#         self.channel_attn = ChannelAttention(out_channels)
#         self.res_block2 = ResidualBlock(out_channels, out_channels)
#         if use_transformer:
#             self.transformer_block = TransformerBlock(
#                 out_channels, n_heads, d_head, n_layers, d_cond=d_cond
#             )
#         else:
#             self.transformer_block = None

#     def forward(self, x, cond=None):
#         h = self.res_block(x)
#         h = self.channel_attn(h) * h
#         h = self.res_block2(h)
#         if self.use_transformer:
#             h = self.transformer_block(h, cond) + h
#         return h

class MiddleBlock(nn.Module):
    def __init__(self, channels, n_heads, d_head, n_layers, d_cond):
        super().__init__()
        self.block1 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=True,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond, self_attn=True
        )
        self.block2 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=False,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond
        )
        self.block3 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=True,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond, self_attn=True
        )
        self.block4 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=False,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond
        )
        self.block5 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=True,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond, self_attn=True
        )
        self.block6 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=False,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond
        )

    def forward(self, x, cond=None, cond2=None):
        x = self.block1(x, cond=cond, cond2=cond2)
        x = self.block2(x, cond=cond, cond2=cond2)
        x = self.block3(x, cond=cond, cond2=cond2)
        x = self.block4(x, cond=cond, cond2=cond2)
        x = self.block5(x, cond=cond, cond2=cond2)
        x = self.block6(x, cond=cond, cond2=cond2)
        return x

class global_encoder(nn.Module):
    def __init__(self, in_channels, n_channels, out_channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=3, padding=1) # 6, 256x256
        self.CA1 = ChannelAttention(in_planes=in_channels * 2)
        self.SA1 = SpatialAttention()

        self.norm2 = nn.InstanceNorm2d(in_channels * 2)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=4, stride=2, padding=1) # 12, 128x128
        self.CA2 = ChannelAttention(in_planes=in_channels * 4)
        self.SA2 = SpatialAttention()

        self.norm3 = nn.InstanceNorm2d(in_channels * 4)
        self.act3 = nn.SiLU()
        self.conv3 = nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=4, stride=2, padding=1) # 24, 64x64
        self.CA3 = ChannelAttention(in_planes=in_channels * 4)
        self.SA3 = SpatialAttention()

        self.norm4 = nn.InstanceNorm2d(in_channels * 4)
        self.act4 = nn.SiLU()
        self.conv4 = nn.Conv2d(in_channels * 4, n_channels, kernel_size=4, stride=2, padding=1) # 64, 32x32
        self.CA4 = ChannelAttention(in_planes=n_channels)
        self.SA4 = SpatialAttention()

        # self.norm5 = nn.InstanceNorm2d(n_channels)
        # self.act5 = nn.SiLU()
        # self.conv5 = nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1) # 128, 16x16
        # self.CA5 = ChannelAttention(in_planes=n_channels * 2)
        # self.SA5 = SpatialAttention()

        # self.norm6 = nn.InstanceNorm2d(n_channels * 2)
        # self.act6 = nn.SiLU()
        # self.conv6 = nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2, padding=1) # 256, 8x8
        # self.CA6 = ChannelAttention(in_planes=n_channels * 4)
        # self.SA6 = SpatialAttention()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.SA1(self.CA1(h)) + h
        h = self.conv2(self.act2(self.norm2(h)))
        h = self.SA2(self.CA2(h)) + h
        h = self.conv3(self.act3(self.norm3(h)))
        h = self.SA3(self.CA3(h)) + h
        h = self.conv4(self.act4(self.norm4(h)))
        h = self.SA4(self.CA4(h)) + h
        # h = self.conv5(self.act5(self.norm5(h)))
        # h = self.SA5(self.CA5(h)) + h
        # h = self.conv6(self.act6(self.norm6(h)))
        # h = self.SA6(self.CA6(h)) + h
        return h

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.CA = ChannelAttention(channels)
        self.SA = SpatialAttention()

    def forward(self, x):
        return self.SA(self.CA(x)) + x

class global_encoder(nn.Module):
    def __init__(self, in_channels, n_channels):
        super().__init__()
        self.init_norm = nn.InstanceNorm2d(in_channels)
        self.init_act = nn.SiLU()
        self.init_conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=7, stride=3, padding=3) # 6, 256x256
        self.CBAM1 = CBAM(in_channels*2)

        self.norm1 = nn.InstanceNorm2d(in_channels*2)
        self.act1 = nn.SiLU()
        self.downsample1 = nn.Conv2d(in_channels*2, in_channels*4, kernel_size=7, stride=2, padding=3) # 12, 128x128
        self.CBAM2 = CBAM(in_channels*4)

        self.norm2 = nn.InstanceNorm2d(in_channels*4)
        self.act2 = nn.SiLU()
        self.downsample2 = nn.Conv2d(in_channels*4, in_channels*8, kernel_size=7, stride=2, padding=3) # 24, 64x64
        self.CBAM3 = CBAM(in_channels*8)

        self.norm3 = nn.InstanceNorm2d(in_channels*8)
        self.act3 = nn.SiLU()
        self.downsample3 = nn.Conv2d(in_channels*8, n_channels, kernel_size=7, stride=2, padding=3) # 64, 32x32
        self.CBAM4 = CBAM(n_channels)

        # self.norm4 = nn.InstanceNorm2d(n_channels)
        # self.act4 = nn.SiLU()
        # self.downsample4 = nn.Conv2d(n_channels, n_channels*2, kernel_size=7, stride=2, padding=3) # 128, 16x16
        # self.CBAM5 = CBAM(n_channels*2)

        # self.norm5 = nn.InstanceNorm2d(n_channels*2)
        # self.act5 = nn.SiLU()
        # self.downsample5 = nn.Conv2d(n_channels*2, n_channels*4, kernel_size=7, stride=2, padding=3) # 256, 8x8
        # self.CBAM6 = CBAM(n_channels*4)

    def forward(self, x):
        h = self.init_conv(self.init_act(self.init_norm(x)))
        h = self.CBAM1(h)

        h = self.downsample1(self.act1(self.norm1(h)))
        h = self.CBAM2(h)

        h = self.downsample2(self.act2(self.norm2(h)))
        h = self.CBAM3(h)

        h = self.downsample3(self.act3(self.norm3(h)))
        h = self.CBAM4(h)

        # h = self.downsample4(self.act4(self.norm4(h)))
        # h = self.CBAM5(h)

        # h = self.downsample5(self.act5(self.norm5(h)))
        # h = self.CBAM6(h)

        return h

# class CrossUNet(nn.Module):
#     def __init__(self, image_channels=3, n_channels=64,
#                  ch_mults=(1, 1, 2, 2),
#                  is_attn=(False, False, False, False), #changed
#                  self_attn=(False, False, False, False), #changed
#                  n_blocks=1, embed_dim=256):
#         super().__init__()

#         self.image_channels = image_channels
#         self.n_channels = n_channels
#         # self.initial_conv = nn.Conv2d(image_channels, 4, kernel_size=3, padding=1)

#         # Positional Encoding for the large input
#         # self.positional_encoding = PositionalEncoding2D(channels=4)

#         # Encoder for the large view
#         self.global_encoder = global_encoder(3, n_channels)

#         # Project image into feature map
#         self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)

#         self.image_resolution = (256, 256)
        
#         self.n_heads = 12

#         # Downsampling blocks
#         down = []
#         out_channels = in_channels = n_channels
#         for i in range(len(ch_mults)):
#             out_channels = in_channels * ch_mults[i]
#             for _ in range(n_blocks):
#                 use_transformer = is_attn[i]
#                 attn = self_attn[i]
#                 down.append(ResidualBlockWithOptionalTransformer(
#                     in_channels, out_channels, use_transformer=use_transformer,
#                     n_heads=self.n_heads, d_head=out_channels // 4, n_layers=1,
#                     d_cond=n_channels, image_resolution=self.image_resolution, self_attn=attn))
#                 in_channels = out_channels
#             if i < len(ch_mults) - 1:
#                 down.append(Downsample(in_channels))
#                 self.image_resolution = (self.image_resolution[0] // 2, self.image_resolution[1] // 2)

#         self.down = nn.ModuleList(down)

#         # Middle block
#         self.middle = MiddleBlock(
#             out_channels, n_heads=self.n_heads, d_head=out_channels // 4, n_layers=1, d_cond=n_channels
#         )

#         # Upsampling blocks
#         up = []
#         in_channels = out_channels
#         for i in reversed(range(len(ch_mults))):
#             out_channels = in_channels
#             for j in range(n_blocks + 1):  # Adjusted for matching dimensions
#                 use_transformer = is_attn[i]
#                 attn = self_attn[i]
#                 up.append(ResidualBlockWithOptionalTransformer(
#                     in_channels + out_channels, out_channels, use_transformer=use_transformer,
#                     n_heads=self.n_heads, d_head=out_channels // 4, n_layers=1,
#                     d_cond=n_channels, image_resolution=self.image_resolution, self_attn=attn))
#                 out_channels = in_channels // ch_mults[i]
#             in_channels = out_channels
#             if i > 0:
#                 up.append(UpSample(in_channels))
#                 self.image_resolution = (self.image_resolution[0] * 2, self.image_resolution[1] * 2)

#         self.up = nn.ModuleList(up)

#         # Final convolution layer
#         self.out = nn.Sequential(
#             nn.InstanceNorm2d(in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, image_channels, 3, padding=1),
#         )

#     def forward(self, x_large, full=False):
#         # Step 1: Add positional embeddings to x_large
#         # x_large_proj = self.initial_conv(x_large)
#         # pe_large = self.positional_encoding(x_large_proj)
#         # x_large_with_pe = x_large_proj + pe_large

#         # Generate image embedding from the larger input with positional embeddings
#         if full:
#             x_global = self.global_encoder(x_large)
#         else:
#             fake_x_large = F.interpolate(x_large, scale_factor=3, mode='nearest')
#             x_global = self.global_encoder(fake_x_large)

#         # Flatten x_global for conditioning
#         b, c, h, w = x_global.shape
#         x_global_flat = x_global.view(b, c, h * w).permute(0, 2, 1)  # [batch_size, n_cond, d_cond]

#         # Step 2: Extract the center 256x256 patch from the larger 768x768 input
#         if full:
#             center_size = 256
#             large_size = x_large.shape[2]
#             start = (large_size - center_size) // 2
#             end = start + center_size
#             x_center = x_large[:, :, start:end, start:end]
#         else:
#             x_center = x_large

#         # Project the center patch
#         x = self.image_proj(x_center)
#         h = [x]

#         # Downsampling blocks
#         for m in self.down:
#             if isinstance(m, Downsample):
#                 x = m(x, None)
#                 h.append(x)
#             else:
#                 if m.use_transformer:
#                     x = m(x, cond=x_global_flat)
#                 else:
#                     x = m(x, cond=x_global)
#                 h.append(x)

#         # Middle block
#         x = self.middle(x, cond=x_global_flat)

#         # Upsampling blocks
#         for m in self.up:
#             if isinstance(m, UpSample):
#                 x = m(x, None)
#             else:
#                 s = h.pop()
#                 x = torch.cat((x, s), dim=1)
#                 if m.use_transformer:
#                     x = m(x, cond=x_global_flat)
#                 else:
#                     x = m(x, cond=x_global)

#         # Final output
#         x = self.out(x)
#         return x

class CrossUNet(nn.Module):
    def __init__(self, image_channels=3, n_channels=64,
                 ch_mults=(1, 2, 4, 4),
                 is_attn=(False, False, False, False), #change
                 self_attn=(False, False, False, False), #changed
                 cross=(False, False, False, False), #changed
                 n_blocks=1, embed_dim=64):
        super().__init__()

        self.image_channels = image_channels
        self.n_channels = n_channels
        # self.initial_conv = nn.Conv2d(image_channels, 4, kernel_size=3, padding=1)

        # Positional Encoding for the large input
        # self.positional_encoding = PositionalEncoding2D(channels=4)

        # Encoder for the large view
        self.global_encoder = global_encoder(image_channels, embed_dim)
        self.intermidiate_channel = n_channels // 2

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)

        self.image_resolution = (256, 256)
        
        self.n_heads = 12

        # Downsampling blocks
        down = []
        out_channels = in_channels = n_channels
        for i in range(len(ch_mults)):
            out_channels = self.n_channels * ch_mults[i]
            for _ in range(n_blocks):
                use_transformer = is_attn[i]
                attn = self_attn[i]
                cross_attn = cross[i]
                down.append(ResidualBlockWithOptionalTransformer(
                    in_channels, out_channels, use_transformer=use_transformer,
                    n_heads=self.n_heads, d_head=out_channels // 4, n_layers=1,
                    d_cond=embed_dim, image_resolution=self.image_resolution, self_attn=attn, cross=cross_attn))
                in_channels = out_channels
            if i < len(ch_mults) - 1:
                down.append(Downsample(in_channels))
                self.image_resolution = (self.image_resolution[0] // 2, self.image_resolution[1] // 2)

        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels, n_heads=self.n_heads, d_head=out_channels // 4, n_layers=1, d_cond=embed_dim
        )

        # Upsampling blocks
        up = []
        in_channels = out_channels
        for i in reversed(range(len(ch_mults))):
            out_channels = in_channels
            for j in range(n_blocks + 1):  # Adjusted for matching dimensions
                use_transformer = is_attn[i]
                attn = self_attn[i]
                cross_attn = cross[i]
                if not i == 0: 
                    out_channels = self.n_channels * ch_mults[i-j]
                if i == 0:
                    out_channels = self.n_channels
                up.append(ResidualBlockWithOptionalTransformer(
                    in_channels + out_channels, out_channels, use_transformer=use_transformer,
                    n_heads=self.n_heads, d_head=out_channels // 4, n_layers=1,
                    d_cond=embed_dim, image_resolution=self.image_resolution, self_attn=attn, cross=cross_attn))
            in_channels = out_channels
            if i > 0:
                up.append(UpSample(in_channels))
                self.image_resolution = (self.image_resolution[0] * 2, self.image_resolution[1] * 2)

        self.up = nn.ModuleList(up)

        # Final convolution layer
        self.out = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, image_channels, 3, padding=1),
        )

    def forward(self, x_large, full=False):
        # Step 1: Add positional embeddings to x_large
        # x_large_proj = self.initial_conv(x_large)
        # pe_large = self.positional_encoding(x_large_proj)
        # x_large_with_pe = x_large_proj + pe_large

        # Generate image embedding from the larger input with positional embeddings
        if full:
            x_global = self.global_encoder(x_large)
        else:
            fake_x_large = F.interpolate(x_large, scale_factor=3, mode='nearest')
            x_global = self.global_encoder(fake_x_large)

        # Flatten x_global for conditioning
        b, c, h, w = x_global.shape
        x_global_flat = x_global.view(b, c, h * w).permute(0, 2, 1)  # [batch_size, n_cond, d_cond]

        # Step 2: Extract the center 256x256 patch from the larger 768x768 input
        if full:
            center_size = 256
            large_size = x_large.shape[2]
            start = (large_size - center_size) // 2
            end = start + center_size
            x_center = x_large[:, :, start:end, start:end]
            cond2 = F.interpolate(x_large, size=(256, 256), mode='nearest')
        else:
            x_center = x_large
            cond2 = x_large

        # Project the center patch
        x = self.image_proj(x_center)
        h = [x]

        # Downsampling blocks
        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x, None)
                h.append(x)
            else:
                if m.use_transformer:
                    x = m(x, cond=x_global_flat, cond2=cond2)
                else:
                    x = m(x, cond2=cond2)
                h.append(x)

        # Middle block
        x = self.middle(x, cond=x_global_flat, cond2=cond2)

        # Upsampling blocks
        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, None)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                if m.use_transformer:
                    x = m(x, cond=x_global_flat, cond2=cond2)
                else:
                    x = m(x, cond2=cond2)

        # Final output
        x = self.out(x)
        return x


# class CrossResnet(nn.Module):
#     def __init__(self, image_channels=3, n_channels=64,
#                  ch_mults=(2, 4), n_blocks=1, embed_dim=256):
#         super().__init__()

#         self.image_channels = image_channels
#         self.n_channels = n_channels
#         self.initial_conv = nn.Conv2d(image_channels, 4, kernel_size=3, padding=1)

#         # Positional Encoding for the large input
#         self.positional_encoding = PositionalEncoding2D(channels=4)

#         # Encoder for the large view
#         self.global_encoder = nn.Sequential(
#             nn.InstanceNorm2d(4),
#             nn.SiLU(),
#             nn.Conv2d(4, 8, kernel_size=4, stride=3, padding=1), # 768 -> 256
#             nn.InstanceNorm2d(16),
#             nn.SiLU(),
#             nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1), # 256 -> 128
#             nn.InstanceNorm2d(32),
#             nn.SiLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 128 -> 64
#             nn.InstanceNorm2d(n_channels),
#             nn.SiLU(),
#             nn.Conv2d(32, n_channels, kernel_size=4, stride=2, padding=1), # 64 -> 32
#             nn.InstanceNorm2d(n_channels * 2),
#             nn.SiLU(),
#             nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1), # 32 -> 16
#             nn.InstanceNorm2d(n_channels * 2),
#             nn.SiLU(),
#             nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2, padding=1), # 16 -> 8
#         )

#         # Project image into feature map
#         self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)

#         self.image_resolution = (256, 256)

#         # Downsampling blocks
#         down = []
#         out_channels = in_channels = n_channels
#         for i in range(len(ch_mults)):
#             out_channels = in_channels * ch_mults[i]
#             for _ in range(n_blocks):
#                 use_transformer = is_attn[i]
#                 down.append(ResidualBlockWithOptionalTransformer(
#                     in_channels, out_channels, use_transformer=use_transformer,
#                     n_heads=4, d_head=out_channels // 4, n_layers=1,
#                     d_cond=n_channels * 4, image_resolution=self.image_resolution))
#                 in_channels = out_channels
#             if i < len(ch_mults) - 1:
#                 down.append(Downsample(in_channels))
#                 self.image_resolution = (self.image_resolution[0] // 2, self.image_resolution[1] // 2)

#         self.down = nn.ModuleList(down)

#         # Middle block
#         self.middle = MiddleBlock(
#             out_channels, n_heads=4, d_head=out_channels // 4, n_layers=1, d_cond=n_channels * 4
#         )

#         # Upsampling blocks
#         up = []
#         in_channels = out_channels
#         for i in reversed(range(len(ch_mults))):
#             out_channels = in_channels
#             for j in range(n_blocks + 1):  # Adjusted for matching dimensions
#                 if j == 0:
#                     use_transformer = False
#                 else:
#                     use_transformer = is_attn[i]
#                 up.append(ResidualBlockWithOptionalTransformer(
#                     in_channels + out_channels, out_channels, use_transformer=use_transformer,
#                     n_heads=4, d_head=out_channels // 4, n_layers=1,
#                     d_cond=n_channels * 4, image_resolution=self.image_resolution))
#                 out_channels = in_channels // ch_mults[i]
#             in_channels = out_channels
#             if i > 0:
#                 up.append(UpSample(in_channels))
#                 self.image_resolution = (self.image_resolution[0] * 2, self.image_resolution[1] * 2)

#         self.up = nn.ModuleList(up)

#         # Final convolution layer
#         self.out = nn.Sequential(
#             nn.InstanceNorm2d(in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, image_channels, 3, padding=1),
#         )

#     def forward(self, x_large, full=False):
#         # Step 1: Add positional embeddings to x_large
#         x_large_proj = self.initial_conv(x_large)
#         pe_large = self.positional_encoding(x_large_proj)
#         x_large_with_pe = x_large_proj + pe_large

#         # Generate image embedding from the larger input with positional embeddings
#         if full:
#             x_global = self.global_encoder(x_large_with_pe)
#         else:
#             fake_x_large = F.interpolate(x_large_with_pe, scale_factor=3, mode='nearest')
#             x_global = self.global_encoder(fake_x_large)

#         # Flatten x_global for conditioning
#         b, c, h, w = x_global.shape
#         x_global_flat = x_global.view(b, c, h * w).permute(0, 2, 1)  # [batch_size, n_cond, d_cond]

#         # Step 2: Extract the center 256x256 patch from the larger 768x768 input
#         if full:
#             center_size = 256
#             large_size = x_large.shape[2]
#             start = (large_size - center_size) // 2
#             end = start + center_size
#             x_center = x_large[:, :, start:end, start:end]
#         else:
#             x_center = x_large

#         # Project the center patch
#         x = self.image_proj(x_center)
#         h = [x]

#         # Downsampling blocks
#         for m in self.down:
#             if isinstance(m, Downsample):
#                 x = m(x, None)
#                 h.append(x)
#             else:
#                 if m.use_transformer:
#                     x = m(x, cond=x_global_flat)
#                 else:
#                     x = m(x, cond=x_global)
#                 h.append(x)

#         # Middle block
#         x = self.middle(x, cond=x_global_flat)

#         # Upsampling blocks
#         for m in self.up:
#             if isinstance(m, UpSample):
#                 x = m(x, None)
#             else:
#                 s = h.pop()
#                 x = torch.cat((x, s), dim=1)
#                 if m.use_transformer:
#                     x = m(x, cond=x_global_flat)
#                 else:
#                     x = m(x, cond=x_global)

#         # Final output
#         x = self.out(x)
#         return x
