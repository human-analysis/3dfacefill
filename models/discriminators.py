import torch
import torch.nn as nn
from .spectral_norm import SpectralNorm

__all__ = ['Discriminator', 'Discriminator_new', 'Discriminator_local', 'Discriminator_localglobal', 'PyramidGAN_large']

class ConvNormUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=0, n_group=4, bias=True):
        super(ConvNormUnit, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.GroupNorm(out_c//n_group, out_c)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.norm(self.conv2d(x)))

class SpectralConvUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=0, n_group=4, bias=True):
        super(SpectralConvUnit, self).__init__()
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.norm = nn.GroupNorm(out_c//n_group, out_c)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.conv2d(x))

class SpectralConvUnit_old(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=0, n_group=4, bias=True):
        super(SpectralConvUnit_old, self).__init__()
        self.conv2d = SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # self.norm = nn.GroupNorm(out_c//n_group, out_c)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.conv2d(x))

class ConvUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=0, bias=True):
        super(ConvUnit, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.act(self.conv2d(x))

class OutConv(nn.Module):
    def __init__(self, in_c, spectral=False):
        super(OutConv, self).__init__()
        if spectral:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0))
        else:
            self.conv = nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class OutConv_old(nn.Module):
    def __init__(self, in_c, spectral=False):
        super(OutConv_old, self).__init__()
        if spectral:
            self.conv = SpectralNorm(nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0))
        else:
            self.conv = nn.Conv2d(in_c, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    ######PatchGAN Discriminator#######
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.nc = args.nchannels
        self.n_group = args.n_groupnorm
        self.n_layers = args.n_disc_layers
        self.layers = []
        self.ndf = args.ndf*2

        self.l1 = ConvUnit(self.nc, self.ndf, kernel_size=4, stride=2, padding=1)

        c_in = self.ndf
        for i in range(self.n_layers-1):
            c_out = min(c_in*2, 512)
            self.layers.append(ConvNormUnit(c_in, c_out, kernel_size=4, stride=2, padding=1, n_group=self.n_group))
            c_in = c_out
        self.layers = nn.ModuleList(self.layers)

        c_out = min(c_in*2, 512)
        self.prefinal = ConvNormUnit(c_in, c_out, kernel_size=3, stride=1, padding=1, n_group=self.n_group)
        self.final = nn.Conv2d(c_out, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        temp = self.l1(input)

        for i in range(self.n_layers-1):
            temp = self.layers[i](temp)

        temp = self.prefinal(temp)
        out = self.final(temp)

        return out

class Discriminator_new(nn.Module):
    ######PatchGAN Discriminator#######
    def __init__(self, args, in_c=None, final_act=None):
        super(Discriminator_new, self).__init__()
        self.args = args
        if in_c is not None:
            self.nc = in_c
        else:
            self.nc = args.nchannels
        self.n_group = args.n_groupnorm
        self.n_layers = args.n_disc_layers
        self.layers = []
        self.ndf = args.ndf
        if final_act is not None:
            self.final_act = getattr(nn, final_act)()

        self.l1 = ConvUnit(self.nc, self.ndf, kernel_size=4, stride=2, padding=1)

        c_in = self.ndf
        for i in range(self.n_layers-1):
            c_out = min(c_in*2, 256)
            self.layers.append(ConvNormUnit(c_in, c_out, kernel_size=4, stride=2, padding=1, n_group=self.n_group))
            self.layers.append(ConvNormUnit(c_out, c_out, kernel_size=3, stride=1, padding=1, n_group=self.n_group))
            c_in = c_out
        self.layers = nn.Sequential(*self.layers)

        c_out = min(c_in*2, 256)
        self.prefinal = ConvNormUnit(c_in, c_out, kernel_size=3, stride=1, padding=1, n_group=self.n_group)
        self.final = nn.Conv2d(c_out, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        temp = self.l1(input)

        temp = self.layers(temp)
        temp = self.prefinal(temp)
        out = self.final(temp)

        if hasattr(self, 'final_act'):
            out = self.final_act(out)
        return out

class Discriminator_local(nn.Module):
    ######PatchGAN Discriminator#######
    def __init__(self, args, in_c=None, final_act=None, gan_type='patch'):
        super(Discriminator_local, self).__init__()
        self.args = args
        if in_c is not None:
            self.nc = in_c
        else:
            self.nc = args.nchannels
        self.n_group = args.n_groupnorm
        self.n_layers = args.n_disc_layers
        self.layers = []
        self.ndf = args.ndf//2
        self.type = gan_type

        self.l1 = SpectralConvUnit(self.nc, self.ndf, kernel_size=4, stride=2, padding=1)

        c_in = self.ndf
        for i in range(self.n_layers-1):
            c_out = min(c_in*2, 512)
            self.layers.append(SpectralConvUnit(c_in, c_out, kernel_size=4, stride=2, padding=1, n_group=self.n_group))
            c_in = c_out
        self.layers = nn.ModuleList(self.layers)

        # c_out = min(c_in*2, 512)
        # self.prefinal = SpectralConvUnit(c_in, c_out, kernel_size=3, stride=1, padding=1, n_group=self.n_group)
        self.final = OutConv(c_out, spectral=True)

    def forward(self, input,):
        temp = self.l1(input)

        for i in range(self.n_layers-1):
            temp = self.layers[i](temp)

        # temp = self.prefinal(temp)
        if self.type == 'full':
            temp = temp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        out = self.final(temp)  #.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        return out

class Discriminator_localglobal(nn.Module):
    ######PatchGAN Discriminator#######
    def __init__(self, args, in_c=None, final_act=None):
        super(Discriminator_localglobal, self).__init__()
        self.args = args
        if in_c is not None:
            self.nc = in_c
        else:
            self.nc = args.nchannels
        self.n_group = args.n_groupnorm
        self.n_layers = args.n_disc_layers
        self.layers = []
        self.outlayers = []
        self.ndf = args.ndf

        self.l1 = SpectralConvUnit(self.nc, self.ndf, kernel_size=4, stride=2, padding=1)

        c_in = self.ndf
        for i in range(self.n_layers-1):
            c_out = min(c_in*2, 512)
            self.layers.append(SpectralConvUnit(c_in, c_out, kernel_size=4, stride=2, padding=1, n_group=self.n_group))
            self.outlayers.append(OutConv(c_out, spectral=True))
            c_in = c_out
        self.layers = nn.ModuleList(self.layers)
        self.outlayers = nn.ModuleList(self.outlayers)

    def forward(self, input):
        temp = self.l1(input)

        output = []
        for i in range(self.n_layers-1):
            temp = self.layers[i](temp)
            output.append(self.outlayers[i](temp))

        return output

class PyramidGAN_large(nn.Module):
    ######PatchGAN Discriminator#######
    def __init__(self, args, in_c=None, final_act=None):
        super(PyramidGAN_large, self).__init__()
        self.args = args
        if in_c is not None:
            self.nc = in_c
        else:
            self.nc = args.nchannels
        self.n_group = args.n_groupnorm
        self.n_layers = args.n_disc_layers
        self.layers = []
        self.outlayers = []
        self.ndf = args.ndf

        self.l1 = SpectralConvUnit(self.nc, self.ndf, kernel_size=4, stride=2, padding=1)

        c_in = self.ndf
        for i in range(self.n_layers-1):
            c_out = min(c_in*2, 512)
            self.layers.append(nn.Sequential(SpectralConvUnit(c_in, c_out, kernel_size=3, stride=1, padding=1, n_group=self.n_group),
                                             SpectralConvUnit(c_out, c_out, kernel_size=3, stride=2, padding=1, n_group=self.n_group)))
            self.outlayers.append(OutConv(c_out, spectral=True))
            c_in = c_out
        self.layers = nn.ModuleList(self.layers)
        self.outlayers = nn.ModuleList(self.outlayers)

    def forward(self, input):
        temp = self.l1(input)

        output = []
        for i in range(self.n_layers-1):
            temp = self.layers[i](temp)
            output.append(self.outlayers[i](temp))

        return output


