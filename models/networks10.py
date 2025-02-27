import torch
torch.manual_seed(523)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator
from torch.nn.utils import spectral_norm

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


class Downsample(nn.Module):
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


class Upsample(nn.Module):
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
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
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
    elif netD == 'stylegan2':
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    elif netD == 'basic2':
        net = NLayerDiscriminator2(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias, num_classes=num_classes)
    elif netD == 'basic3':
        net = NLayerDiscriminator3(input_nc, ndf, n_layers=3, num_classes=num_classes)
    elif netD == 'multi':
        net = MultiScaleDiscriminator(input_nc, ndf, num_classes=num_classes)
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

        # sequence += [nn.Conv2d(ndf * nf_mult, num_classes, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

        ####
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ndf * nf_mult, num_classes)

    def forward(self, input):
        """Standard forward."""
        x = self.model(input)

        x = self.global_pool(x)  # Shape: (N, ndf * nf_mult, 1, 1)
        x = x.view(x.size(0), -1)  # Shape: (N, ndf * nf_mult)
        output = self.fc(x)  # Shape: (N, num_classes)

        return output


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

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UnetDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.parametrizations.spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        # Apply convolution
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="zeros", stride=1)

    def forward(self, x: torch.Tensor):
        # Downsample
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        # Apply convolution
        return self.conv(x)

class DownsampleAA(nn.Module):
    def __init__(self, n_channels, filt_size=3, stride=2, pad_type='zero'):
        super().__init__()

        self.down = Downsample(
            channels=n_channels,
            pad_type=pad_type,
            filt_size=filt_size,
            stride=stride,
            pad_off=0
        )
    def forward(self, x):
        return self.down(x)

class UpSampleAA(nn.Module):
    def __init__(self, n_channels, filt_size=4, stride=2, pad_type='reflect'):
        super().__init__()

        self.up = Upsample(
            channels=n_channels,
            pad_type=pad_type,
            filt_size=filt_size,
            stride=stride
        )
    def forward(self, x):
        return self.up(x)

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
        # self.norm = nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.norm1 = SAN(in_channels)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect')

        # self.norm2 = SAN(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode='reflect')

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        
        return h + self.shortcut(x)

class TransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, d_head, n_layers, d_cond=None, self_attn=False, cross=True):
        super().__init__()
        self.transformer = SpatialTransformer(
            channels, n_heads, d_head, n_layers, d_cond=d_cond, self_attn=self_attn, cross=cross
        )

    def forward(self, x, cond=None):
        return self.transformer(x, cond)

class ResidualBlockWithOptionalTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, use_transformer=False,
                 n_heads=1, d_head=64, n_layers=1, d_cond=None, self_attn=False, cross=True):
        super().__init__()
        self.use_transformer = use_transformer

        self.res_block1 = ResidualBlock(in_channels, out_channels)
        if use_transformer:
            self.transformer_block1 = TransformerBlock(
                out_channels, n_heads, d_head, n_layers, d_cond=d_cond, self_attn=self_attn, cross=cross
            )
        else:
            self.transformer_block = None

    def forward(self, x, emb=None):
        h = self.res_block1(x)
        if self.use_transformer:
            h = self.transformer_block1(h, emb) + h

        return h

class MiddleBlock(nn.Module):
    def __init__(self, channels, n_heads, d_head, n_layers, d_cond, cross):
        super().__init__()
        self.block1 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=True,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond, self_attn=True, cross=cross
        )
        self.block2 = ResidualBlockWithOptionalTransformer(
            channels, channels, use_transformer=False,
            n_heads=n_heads, d_head=d_head, n_layers=n_layers,
            d_cond=d_cond
        )

    def forward(self, x, emb=None):
        x = self.block1(x, emb)
        x = self.block2(x)

        return x

class Adapter(torch.nn.Module):
    def __init__(self, cross_attention_dim=512, clip_embeddings_dim=512, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.act = torch.nn.GELU()
        self.linear = torch.nn.Linear(self.clip_extra_context_tokens * cross_attention_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.linear(self.act(self.proj(embeds)))
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class CrossUNet(nn.Module):
    """
    Modified architecture:
      1) No saving x right after image_proj.
      2) Store skip connections *before* each downsampling at each scale.
      3) Decoder: each scale first concatenates with skip, then runs (n_blocks+1) blocks, then upsamples.

    Explanation of key attributes:
      - self.n_blocks: number of residual/transformer blocks per scale in the down path.
      - self.down: nn.ModuleList of blocks and downsampling modules.
      - self.middle: bottleneck MiddleBlock.
      - self.up: nn.ModuleList of blocks that mirror self.down in reverse, plus upsampling modules.
      - self.out: final 3×3 convolution to produce the output image.
    """
    def __init__(self, image_channels=3, n_channels=64,
                 ch_mults=(1, 1, 2, 2, 4, 4),
                 is_attn=(False, False, False, True, True, True),
                 self_attn=(False, False, False, True, True, True),
                 cross=(False, False, False, True, True, True),
                 n_blocks=1, embed_dim=512, clip_dim=512):
        super().__init__()

        self.image_channels = image_channels
        self.n_channels = n_channels
        self.n_blocks = n_blocks  # Store n_blocks for use in forward()
        self.cross = cross[5]

        # Cross-attention embedding adapter
        if self.cross:
            self.adapter = Adapter(
                cross_attention_dim=embed_dim,
                clip_embeddings_dim=clip_dim,
                clip_extra_context_tokens=4
            )

        # Initial projection: image_channels -> n_channels
        self.image_proj = nn.Conv2d(
            in_channels=image_channels,
            out_channels=n_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )

        self.n_heads = 8
        self.num_scales = len(ch_mults)

        # -----------------------------------------------------------------
        # Construct Downsampling path (self.down)
        # -----------------------------------------------------------------
        down_modules = []
        in_ch = n_channels

        for i in range(self.num_scales):
            out_ch = n_channels * ch_mults[i]

            # For each scale, apply n_blocks of ResidualBlockWithOptionalTransformer
            for _ in range(self.n_blocks):
                use_transformer = is_attn[i]
                attn = self_attn[i]
                cross_attn = cross[i]
                down_modules.append(
                    ResidualBlockWithOptionalTransformer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        use_transformer=use_transformer,
                        n_heads=self.n_heads,
                        d_head=out_ch // 4,
                        n_layers=1,
                        d_cond=embed_dim,
                        self_attn=attn,
                        cross=cross_attn
                    )
                )
                in_ch = out_ch

            # Store skip AFTER these blocks. Then if not last scale, downsample
            if i < self.num_scales - 1:
                # down_modules.append(DownsampleAA(in_ch))
                down_modules.append(DownSample(in_ch))

        self.down = nn.ModuleList(down_modules)

        # -----------------------------------------------------------------
        # Middle (bottleneck) block
        # -----------------------------------------------------------------
        self.middle = MiddleBlock(
            channels=in_ch,
            n_heads=self.n_heads,
            d_head=in_ch // 4,
            n_layers=1,
            d_cond=embed_dim,
            cross=self.cross
        )

        # -----------------------------------------------------------------
        # Construct Upsampling path (self.up)
        # -----------------------------------------------------------------
        up_modules = []
        # 'in_ch' at this point is whatever the encoder ended with
        for i in reversed(range(self.num_scales)):
            out_ch = n_channels * ch_mults[i]

            # We do (n_blocks + 1) blocks at each scale in the up path
            for _ in range(self.n_blocks + 1):
                use_transformer = is_attn[i]
                attn = self_attn[i]
                cross_attn = cross[i]
                up_modules.append(
                    ResidualBlockWithOptionalTransformer(
                        in_channels=in_ch + out_ch,  # concat skip + current
                        out_channels=out_ch,
                        use_transformer=use_transformer,
                        n_heads=self.n_heads,
                        d_head=out_ch // 4,
                        n_layers=1,
                        d_cond=embed_dim,
                        self_attn=attn,
                        cross=cross_attn
                    )
                )
                in_ch = out_ch  # update for next block

            # Upsample if not the top scale
            if i > 0:
                # up_modules.append(UpSampleAA(in_ch))
                up_modules.append(UpSample(in_ch))

        self.up = nn.ModuleList(up_modules)

        # Final output
        self.out = nn.Sequential(
            nn.InstanceNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, image_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x, emb=None):
        """
        Overall flow:
          1) image_proj -> x
          2) Down path (self.down):
             - For each scale i, apply n_blocks, store skip, downsample if not last
          3) Middle (bottleneck)
          4) Up path (self.up):
             - For each scale i in reverse, pop skip, concat, (n_blocks+1) blocks, upsample
          5) Final output
        """
        # 1) Process cross-attention embedding if provided
        if self.cross:
            emb = self.adapter(emb)

        # 2) Initial conv
        x = self.image_proj(x)  # shape: [B, n_channels, H, W]

        # --------------------
        # Down Path
        # --------------------
        skips = []
        d_idx = 0  # index into self.down

        for i in range(self.num_scales):
            # Apply n_blocks at scale i
            for _ in range(self.n_blocks):
                block = self.down[d_idx]
                d_idx += 1
                x = block(x, emb)
            # Save skip
            skips.append(x)

            # Downsample if not last scale
            if i < self.num_scales - 1:
                downsample = self.down[d_idx]
                d_idx += 1
                x = downsample(x)

        # x now at bottleneck
        # --------------------
        # Middle
        # --------------------
        x = self.middle(x, emb)

        # --------------------
        # Up Path
        # --------------------
        u_idx = 0  # index into self.up

        for i in reversed(range(self.num_scales)):
            skip = skips[i]
            # (n_blocks + 1) blocks at scale i
            for _ in range(self.n_blocks + 1):
                # Concat skip
                x = torch.cat([x, skip], dim=1)
                block = self.up[u_idx]
                u_idx += 1
                x = block(x, emb)

            # Upsample if not top scale
            if i > 0:
                upsample = self.up[u_idx]
                u_idx += 1
                x = upsample(x)

        # --------------------
        # Final output
        # --------------------
        x = self.out(x)
        return x

# Discriminator remains the same with spectral normalization
class NLayerDiscriminator3(nn.Module):
    """Defines a PatchGAN discriminator with spectral normalization"""
    def __init__(self, input_nc, ndf=64, n_layers=3, num_classes=9):
        """Construct a PatchGAN discriminator with spectral normalization"""
        super(NLayerDiscriminator3, self).__init__()

        kw = 4
        padw = 1
        dim = ndf
        anti_alias = False

        # Initial convolution with spectral normalization
        if anti_alias:
            sequence = [
                nn.Conv2d(input_nc, dim, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect'),
                nn.LeakyReLU(0.2, True),
                Downsample(dim)
            ]

            # Downsampling layers with instance normalization
            for n in range(1, n_layers):

                sequence += [
                    nn.Conv2d(dim, dim*2, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect'),
                    nn.InstanceNorm2d(dim*2),
                    nn.LeakyReLU(0.2, True),
                    Downsample(dim*2)
                ]
                dim *= 2

            # Final convolution layers
            sequence += [
                nn.Conv2d(dim, dim*2, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect'),
                nn.InstanceNorm2d(dim*2),
                nn.LeakyReLU(0.2, True)
            ]
            dim *= 2

            self.model = nn.Sequential(*sequence)

            map1 = [
                nn.Conv2d(dim, dim, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect'),
                nn.InstanceNorm2d(dim),
                nn.LeakyReLU(0.2, True),
            ]

            self.map1 = nn.Sequential(*map1) # prediction map
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(dim, num_classes)

            self.map2 = nn.Conv2d(dim, 1, kernel_size=kw, stride=1, padding=padw) # real/fake map
        else:
            sequence = [
                spectral_norm(nn.Conv2d(input_nc, dim, kernel_size=kw, stride=2, padding=padw, padding_mode='reflect')),
                nn.LeakyReLU(0.2, True)
            ]

            # Downsampling layers with spectral normalization
            for n in range(1, n_layers):

                sequence += [
                    spectral_norm(nn.Conv2d(dim, dim*2, kernel_size=kw, stride=2, padding=padw, padding_mode='reflect')),
                    nn.LeakyReLU(0.2, True)
                ]
                dim *= 2

            # Final convolution layers with spectral normalization
            sequence += [
                spectral_norm(nn.Conv2d(dim, dim*2, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect')),
                nn.LeakyReLU(0.2, True)
            ]
            dim *= 2

            self.model = nn.Sequential(*sequence)

            map1 = [
                spectral_norm(nn.Conv2d(dim, dim, kernel_size=kw, stride=1, padding=padw, padding_mode='reflect')),
                nn.LeakyReLU(0.2, True),
            ]

            self.map1 = nn.Sequential(*map1) # prediction map
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = spectral_norm(nn.Linear(dim, num_classes))
            # self.map1 = nn.Conv2d(dim, num_classes, kernel_size=kw, stride=1, padding=padw) # prediction map
            self.map2 = spectral_norm(nn.Conv2d(dim, 1, kernel_size=kw, stride=1, padding=padw)) # real/fake map

    def forward(self, input):
        x = self.model(input)

        x_map1 = self.global_pool(self.map1(x))
        x_map1 = x_map1.view(x.size(0), -1)
        pred1 = self.fc(x_map1)

        pred2 = self.map2(x)

        return pred1, pred2

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_classes=9, num_scales=3, norm='none', activation='lrelu', pad_type='reflect'):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.padding = nn.ReflectionPad2d(1)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=0, count_include_pad=False)

        self.discriminators = nn.ModuleList()
        for _ in range(self.num_scales):
            discriminator = NLayerDiscriminator3(input_nc, ndf, n_layers=3, num_classes=num_classes)
            self.discriminators.append(discriminator)

    def forward(self, x):
        img_outputs = []
        for discriminator in self.discriminators:
            img_out = discriminator(x)
            img_outputs.append(img_out)
            x = self.downsample(self.padding(x))
        return img_outputs