import torch.nn as nn


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_layer=nn.BatchNorm2d, reflect_padding=True, use_dropout=True, use_bias=True):
        """Initialize the Resnet block

        Construct a convolutional block.

        :param dim: (int)               -- the number of channels in the conv layer.
        :param norm_layer:              -- normalization layer
        :param reflect_padding: (bool)  -- add ReflectionPad2d or not
        :param use_dropout: (bool)      -- if use dropout layers.
        :param use_bias: (bool)         -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, reflect_padding, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, reflect_padding, use_dropout, use_bias):
        """Initialize the Resnet block

        Construct a convolutional block.

        :param dim: (int)               -- the number of channels in the conv layer.
        :param norm_layer:              -- normalization layer
        :param reflect_padding: (bool)  -- add ReflectionPad2d or not
        :param use_dropout: (bool)      -- if use dropout layers.
        :param use_bias: (bool)         -- if the conv layer uses bias or not
        :return: conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """

        conv_block = []

        # First Reflection Padding
        p = 0
        if reflect_padding:
            conv_block += [nn.ReflectionPad2d(1)]
        else:
            p = 1

        # First conv layer
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        # First dropout layer
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # First Reflection Padding
        p = 0
        if reflect_padding:
            conv_block += [nn.ReflectionPad2d(1)]
        else:
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    Link: https://github.com/jcjohnson/fast-neural-style
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 reflect_padding=True):
        """Construct a Resnet-based generator

        :param input_nc: (int)          -- the number of channels in input images
        :param output_nc: (int)         -- the number of channels in output images
        :param ngf: (int)               -- the number of filters in the last conv layer
        :param norm_layer:              -- normalization layer
        :param use_dropout: (bool)      -- if use dropout layers
        :param n_blocks: (int)          -- the number of ResNet blocks
        :param reflect_padding: (bool)  -- ReflectionPad2d or not
        """

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        # use_bias is set false if BatchNorm2d is used as norm layer
        use_bias = norm_layer != nn.BatchNorm2d

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
            model += [
                ResnetBlock(ngf * mult, norm_layer=norm_layer, reflect_padding=reflect_padding, use_dropout=use_dropout,
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
        """Construct a NLayerDiscriminator discriminator

        :param input_nc: (int)  -- the number of channels in input images
        :param ndf: (int)       -- the number of filters in the last conv layer
        :param n_layers: (int)  -- the number of conv layers in the discriminator
        :param norm_layer:      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        # use_bias is set false if BatchNorm2d is used as norm layer
        use_bias = norm_layer != nn.BatchNorm2d

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

    def forward(self, x):
        """Standard forward."""
        return self.model(x)
