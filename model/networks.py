import torch
import functools
from torch import nn
from torch.nn import init


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_layer=nn.BatchNorm2d, padding_type='reflect', use_dropout=True, use_bias=True):
        """Initialize the Resnet block

        Construct a convolutional block.

        :param dim: the number of channels in the conv layer.
        :param norm_layer: normalization layer
        :param padding_type: the name of padding layer: reflect | replicate | zero
        :param use_dropout: if use dropout layers.
        :param use_bias: if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))


        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, padding_type, use_dropout, use_bias):
        """Initialize the Resnet block

        Construct a convolutional block.

        :param dim: the number of channels in the conv layer.
        :param norm_layer: normalization layer
        :param padding_type: the name of padding layer: reflect | replicate | zero
        :param use_dropout: if use dropout layers.
        :param use_bias: if the conv layer uses bias or not
        :return: conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        # First Reflection Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # First conv layer
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        # First dropout layer
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # First Reflection Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 use_dropout=False, n_blocks=9):
        """Construct a Resnet-based generator

        :param input_nc: the number of channels in input images
        :param output_nc: the number of channels in output images
        :param ngf: the number of filters in the last conv layer
        :param norm_layer: normalization layer
        :param padding_type: the name of padding layer: reflect | replicate | zero
        :param use_dropout: if use dropout layers
        :param n_blocks: the number of ResNet blocks
        """

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        # use_bias is set false if BatchNorm2d is used as norm layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # First Conv layer
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # 2 Down sampling layers (2nd and 3rd)
        n_down_sampling = 2
        for i in range(n_down_sampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_down_sampling

        # n_blocks resnet layers
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, padding_type=padding_type, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        # Add up sampling Layers
        for i in range(n_down_sampling):
            mult = 2 ** (n_down_sampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        # use_bias is set false if BatchNorm2d is used as norm layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        # First Conv Layer
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        # Next n conv layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        # last-1 Conv Layer
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Last conv layer
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode: str, target_real_label=1.0, target_fake_label=0.0):
        """

        :param gan_mode: the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
        :param target_real_label: label for a real image. Default: 1.0
        :param target_fake_label: label of a fake image. Default: 0.0

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
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        :param prediction: the prediction from a discriminator
        :param target_is_real: if the ground truth label is for real images or fake images
        :return: A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grunt truth labels.

        :param prediction: typically the prediction output from a discriminator
        :param target_is_real: if the ground truth label is for real images or fake images
        :return: the calculated loss.
        """
        _loss = 0.0
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            _loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                _loss = -prediction.mean()
            else:
                _loss = prediction.mean()
        return _loss


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


def init_weights(net, init_type='normal', init_gain=0.02):
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
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights

    :param net: the network to be initialized
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :return: an initialized network.
    """

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def build_G(input_nc=3, output_nc=3, ngf=64, norm='batch', padding_type='reflect', use_dropout=True, n_blocks=9,
            init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    :param input_nc: the number of channels in input images
    :param output_nc: the number of channels in output images
    :param ngf: the number of filters in the last conv layer
    :param norm: the name of normalization layers used in the network: batch | instance | none
    :param padding_type: the name of padding layer: reflect | replicate | zero
    :param use_dropout: if use dropout layers.
    :param n_blocks: the number of ResNet blocks
    :param init_type: the name of our initialization method.
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :return: an initialized Generator network.
    """

    norm_layer = get_norm_layer(norm)
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, padding_type, use_dropout, n_blocks)
    return init_net(net, init_type, init_gain, gpu_ids)


def build_D(input_nc=3, ndf=64, n_layers=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    :param input_nc: the number of channels in input images
    :param ndf: the number of filters in the first conv layer
    :param n_layers: the number of conv layers in the discriminator
    :param norm: the name of normalization layers used in the network: batch | instance | none
    :param init_type: the name of our initialization method.
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :return: an initialized discriminator network.
    """
    norm_layer = get_norm_layer(norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)
