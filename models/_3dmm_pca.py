# 3dmm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

# __all__ = ['Encoder', 'ShapeDecoder', 'AlbedoDecoder', 'AutoEncoder']

class ConvUnit(nn.Module):
    def __init__(self, in_c, out_c, n_groupf=4, kernel_size=3, stride=1, padding=1, bias=True, norm=True):
        super(ConvUnit, self).__init__()
        self.norm = norm
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # self.conv2d = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if self.norm:
            self.norm_layer = nn.GroupNorm(out_c//n_groupf, out_c)
        self.elu = nn.ELU()

    def forward(self, x):
        temp = self.conv2d(x)
        if self.norm:
            temp = self.norm_layer(temp)
        return self.elu(temp)

class ConvNoActUnit(nn.Module):
    def __init__(self, in_c, out_c, n_groupf=4, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvNoActUnit, self).__init__()
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # self.conv2d = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.GroupNorm(out_c//n_groupf, out_c)

    def forward(self, x):
        return self.norm(self.conv2d(x))

class DeconvUnit(nn.Module):
    """docstring for DeconvUnit"""
    def __init__(self, in_c, out_c, n_groupf=4, kernel_size=4, stride=2, padding=1, bias=True, upsample=True):
        super(DeconvUnit, self).__init__()
        if upsample:
            if type(kernel_size) is tuple:
                factor = (kernel_size[0]/stride, kernel_size[1]/stride)
            else:
                factor = kernel_size / stride
            deconv2d = [nn.Upsample(scale_factor=factor), nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=bias))]
            self.deconv2d = nn.Sequential(*deconv2d)
        else:
            self.deconv2d = nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        # self.deconv2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.GroupNorm(out_c//n_groupf, out_c)
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(self.norm(self.deconv2d(x)))


class Encoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.nc = args.nchannels            # 3
        self.ngf = args.ngf                 # 32
        self.ngfc = args.ngfc               # 512
        self.mdim = args.mdim               # 6
        self.ildim = args.ildim             # 27
        self.n_group = args.n_groupnorm     # nc_out/4
        self.use_conf = args.use_conf

        self.k0_1 = ConvUnit(self.nc, self.ngf*1, self.n_group, kernel_size=7, stride=2, padding=3, bias=False)     # 112
        self.k0_2 = ConvUnit(self.ngf*1, self.ngf*2, self.n_group, bias=False)

        self.k1_0 = ConvUnit(self.ngf*2, self.ngf*2, self.n_group, stride=2, bias=False)                            # 56
        self.k1_1 = ConvUnit(self.ngf*2, self.ngf*3, self.n_group, bias=False)
        self.k1_2 = ConvUnit(self.ngf*3, self.ngf*4, self.n_group, bias=False)

        self.k2_0 = ConvUnit(self.ngf*4, self.ngf*4, self.n_group, stride=2, bias=False)                            # 28
        self.k2_1 = ConvUnit(self.ngf*4, self.ngf*6, self.n_group, bias=False)
        self.k2_2 = ConvUnit(self.ngf*6, self.ngf*8, self.n_group, bias=False)

        self.k3_0 = ConvUnit(self.ngf*8, self.ngf*8, self.n_group, stride=2, bias=False)                            # 14
        self.k3_1 = ConvUnit(self.ngf*8, self.ngf*8, self.n_group, bias=False)
        self.k3_2 = ConvUnit(self.ngf*8, self.ngf*8, self.n_group, bias=False)

        self.k4_0 = ConvUnit(self.ngf*8, self.ngf*16, self.n_group, stride=2, bias=False)                            # 7
        self.k4_1 = ConvUnit(self.ngf*16, self.ngf*16, self.n_group, bias=False)

        # M
        self.k5_m = ConvUnit(self.ngf*16, self.ngf*5, self.n_group)
        self.k6_m = nn.Linear(self.ngf*5, self.mdim)
        self.act_m = nn.Tanh()

        # IL
        self.k5_il = ConvUnit(self.ngf*16, self.ngf*5, self.n_group)
        self.k6_il = nn.Linear(self.ngf*5, self.ildim)

        # Shape
        self.k5_shape = ConvUnit(self.ngf*16, self.ngfc, self.n_group)
        self.k6_shape = ConvUnit(self.ngfc, self.ngfc, self.n_group)
        self.k7_shape = nn.Linear(self.ngfc, 199+29)

        # Albedo
        self.k5_tex = ConvUnit(self.ngf*16, int(self.ngfc), self.n_group)

        # Confidence
        # if self.use_conf:
        self.k5_conf = ConvUnit(self.ngf*16, int(self.ngfc), self.n_group)

    def forward(self, input):
        temp = self.k0_1(input)
        temp = self.k0_2(temp)

        temp = self.k1_0(temp)
        temp = self.k1_1(temp)
        temp = self.k1_2(temp)

        temp = self.k2_0(temp)
        temp = self.k2_1(temp)
        temp = self.k2_2(temp)

        temp = self.k3_0(temp)
        temp = self.k3_1(temp)
        temp = self.k3_2(temp)

        temp = self.k4_0(temp)
        temp = self.k4_1(temp)

        # M
        m_temp = self.k5_m(temp)
        _shape = m_temp.shape[-1]
        m_temp = nn.functional.avg_pool2d(m_temp, kernel_size=_shape, stride=1)
        m_temp = m_temp.view(-1, self.ngf*5)
        m_temp = self.k6_m(m_temp)
        m_out = self.act_m(m_temp[:,1:])
        scale_out = 0.5*(self.act_m(m_temp[:,0])+1)*2e-3

        # IL
        il_temp = self.k5_il(temp)
        il_temp = nn.functional.avg_pool2d(il_temp, kernel_size=_shape, stride=1)
        il_temp = il_temp.view(-1, self.ngf*5)
        il_out = self.k6_il(il_temp)

        # Shape
        shape_temp = self.k5_shape(temp)
        shape_temp = self.k6_shape(shape_temp)
        shape_temp = nn.functional.avg_pool2d(shape_temp, kernel_size=_shape, stride=1)
        shape_temp = shape_temp.view(-1, int(self.ngfc))
        shape_out = self.k7_shape(shape_temp)

        # Albedo
        tex_temp = self.k5_tex(temp) # change back to self.k5_tex
        tex_temp = nn.functional.avg_pool2d(tex_temp, kernel_size=_shape, stride=1)
        tex_out = tex_temp.view(-1, int(self.ngfc))

        # Confidence
        if self.use_conf:
            conf_temp = self.k5_conf(temp)
            conf_temp = nn.functional.avg_pool2d(conf_temp, kernel_size=_shape, stride=1)
            conf_out = conf_temp.view(-1, int(self.ngfc))
            return shape_out, tex_out, conf_out, scale_out, m_out, il_out

        return shape_out, tex_out, scale_out, m_out, il_out


# Texture Decoder
class AlbedoDecoder(nn.Module):
    """docstring for TextureDecoder"""
    def __init__(self, args):
        super(AlbedoDecoder, self).__init__()
        self.args = args
        self.texture_size = args.texture_size
        self.s_h = int(self.texture_size[0])
        self.s_w = int(self.texture_size[1])
        self.s32_h = int(self.s_h/32)
        self.s32_w = int(self.s_w/32)
        self.ngfc = args.ngfc
        self.ngf = args.ngf
        self.nc = args.nchannels
        self.n_groupf = args.n_groupnorm

        self.h6_1 = DeconvUnit(self.ngfc, self.ngfc, self.n_groupf, kernel_size=(3,4), stride=1, padding=0)
        self.h6_0 = ConvUnit(self.ngfc, self.ngfc//2, norm=False)

        self.h5_2 = DeconvUnit(self.ngfc//2, self.ngfc//2, self.n_groupf, stride=2)
        self.h5_1 = ConvUnit(self.ngfc//2, self.ngfc//4)
        self.h5_0 = ConvUnit(self.ngfc//4, self.ngfc//4)

        self.h4_2 = DeconvUnit(self.ngfc//4, self.ngf*5, self.n_groupf, stride=2)
        self.h4_1 = ConvUnit(self.ngf*5, self.ngf*3, self.n_groupf)
        self.h4_0 = ConvUnit(self.ngf*3, self.ngf*4, self.n_groupf)

        self.h3_2 = DeconvUnit(self.ngf*4, self.ngf*4, self.n_groupf, stride=2)
        self.h3_1 = ConvUnit(self.ngf*4, self.ngf*2, self.n_groupf)
        self.h3_0 = ConvUnit(self.ngf*2, self.ngf*3, self.n_groupf)

        self.h2_2 = DeconvUnit(self.ngf*3, self.ngf*3, self.n_groupf, stride=2)
        self.h2_1 = ConvUnit(self.ngf*3, self.ngf*2, self.n_groupf)
        self.h2_0 = ConvUnit(self.ngf*2, self.ngf*2, self.n_groupf)

        self.h1_2 = DeconvUnit(self.ngf*2, self.ngf*2, self.n_groupf, stride=2)
        self.h1_1 = ConvUnit(self.ngf*2, self.ngf, self.n_groupf)
        self.h1_0 = ConvUnit(self.ngf, self.ngf, self.n_groupf)

        self.h0_2 = DeconvUnit(self.ngf, self.ngf, self.n_groupf, stride=2)
        self.h0_1 = ConvUnit(self.ngf, self.ngf//2, self.n_groupf)
        self.h0_0 = ConvUnit(self.ngf//2, self.ngf//2, self.n_groupf)

        self.final = nn.Conv2d(self.ngf//2, self.nc, kernel_size=1, stride=1, padding=0)
        self.final_act = nn.Tanh()

    def forward(self, input):
        temp = input.view(-1, self.ngfc, 1, 1)
        temp = self.h6_1(temp)
        temp = self.h6_0(temp)

        temp = self.h5_2(temp)
        temp = self.h5_1(temp)
        temp = self.h5_0(temp)

        temp = self.h4_2(temp)
        temp = self.h4_1(temp)
        temp = self.h4_0(temp)

        temp = self.h3_2(temp)
        temp = self.h3_1(temp)
        temp = self.h3_0(temp)

        temp = self.h2_2(temp)
        temp = self.h2_1(temp)
        temp = self.h2_0(temp)

        temp = self.h1_2(temp)
        temp = self.h1_1(temp)
        temp = self.h1_0(temp)

        temp = self.h0_2(temp)
        temp = self.h0_1(temp)
        temp = self.h0_0(temp)
        out = self.final_act(self.final(temp))

        return out

class ShapeDecoder(nn.Module):
    def __init__(self, args):
        super(ShapeDecoder, self).__init__()


# Symmetry Decoder
class ConfidenceDecoder(nn.Module):
    """docstring for TextureDecoder"""
    def __init__(self, args):
        super(ConfidenceDecoder, self).__init__()
        self.args = args
        self.texture_size = args.texture_size
        self.s_h = int(self.texture_size[0])
        self.s_w = int(self.texture_size[1])
        self.s32_h = int(self.s_h/32)
        self.s32_w = int(self.s_w/32)
        self.ngfc = args.ngfc
        self.ngf = args.ngf
        self.nc = args.nchannels
        self.n_groupf = args.n_groupnorm

        self.h6 = DeconvUnit(self.ngfc, self.ngfc//2, self.n_groupf, kernel_size=(3,4), stride=1, padding=0)
        self.h5 = DeconvUnit(self.ngfc//2, self.ngfc//4, self.n_groupf, stride=2)
        self.h4 = DeconvUnit(self.ngfc//4, self.ngf*5, self.n_groupf, stride=2)
        self.h3 = DeconvUnit(self.ngf*5, self.ngf*4, self.n_groupf, stride=2)
        self.h2 = DeconvUnit(self.ngf*4, self.ngf*3, self.n_groupf, stride=2)
        self.h1 = DeconvUnit(self.ngf*3, self.ngf*2, self.n_groupf, stride=2)
        self.h0 = DeconvUnit(self.ngf*2, self.ngf, self.n_groupf, stride=2)
        self.final = nn.Conv2d(self.ngf, 1, kernel_size=1, stride=1, padding=0)
        self.final_act = nn.Softplus()

    def forward(self, input):
        temp = input.view(-1, self.ngfc, 1, 1)
        temp = self.h6(temp)
        temp = self.h5(temp)
        temp = self.h4(temp)
        temp = self.h3(temp)
        temp = self.h2(temp)
        temp = self.h1(temp)
        temp = self.h0(temp)
        out = self.final(temp)
        out = torch.clamp(out, min=-10)

        return out


# Autoencoder model
class AutoEncoder(nn.Module):
    """docstring for AutoEncoder"""
    def __init__(self, args, in_channels=3, out_channels=1):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.in_c = in_channels            # 3
        self.out_c = out_channels
        self.ngf = args.ngf                 # 32
        self.n_groupf = args.n_groupnorm     # nc_out/4
        self.z = args.enc_z
        self.s_h = int(args.resolution_high/32)
        self.s_w = int(args.resolution_wide/32)

        self.he0_0 = ConvUnit(self.in_c, self.ngf*1, self.n_groupf, kernel_size=7, stride=2, padding=3, bias=False)         # 112
        self.he0_1 = ConvUnit(self.ngf*1, self.ngf*1, self.n_groupf, bias=False)

        self.he1_0 = ConvUnit(self.ngf*1, self.ngf*2, self.n_groupf, stride=2, bias=False)                                  # 56
        self.he1_1 = ConvUnit(self.ngf*2, self.ngf*2, self.n_groupf, bias=False)

        self.he2_0 = ConvUnit(self.ngf*2, self.ngf*4, self.n_groupf, stride=2, bias=False)                                  # 28
        self.he2_1 = ConvUnit(self.ngf*4, self.ngf*4, self.n_groupf, bias=False)

        self.he3_0 = ConvUnit(self.ngf*4, self.ngf*8, self.n_groupf, stride=2, bias=False)                                  # 14
        self.he3_1 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, bias=False)

        self.he4_0 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, stride=2, bias=False)                                  # 7
        self.he4_1 = ConvUnit(self.ngf*8, self.ngf*16, self.n_groupf, bias=False)
        self.he5_lin = nn.Linear(self.ngf*16, self.z)                                                                       # Nxz

        self.hd5_lin = nn.Linear(self.z, self.ngf*8*self.s_h*self.s_w)                                                      # Nx256x7x7
        self.hd5_0 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, norm=False)

        self.hd4_1 = DeconvUnit(self.ngf*8, self.ngf*8, self.n_groupf, stride=2)                                            # 14
        self.hd4_0 = ConvUnit(self.ngf*8, self.ngf*4, self.n_groupf)

        self.hd3_1 = DeconvUnit(self.ngf*4, self.ngf*4, self.n_groupf, stride=2)                                            # 28
        self.hd3_0 = ConvUnit(self.ngf*4, self.ngf*4, self.n_groupf)

        self.hd2_1 = DeconvUnit(self.ngf*4, self.ngf*2, self.n_groupf, stride=2)                                            # 56
        self.hd2_0 = ConvUnit(self.ngf*2, self.ngf*2, self.n_groupf)

        self.hd1_1 = DeconvUnit(self.ngf*2, self.ngf*1, self.n_groupf, stride=2)                                            # 112
        self.hd1_0 = ConvUnit(self.ngf*1, self.ngf*1, self.n_groupf)

        self.hd0_1 = DeconvUnit(self.ngf*1, self.ngf//2, self.n_groupf, stride=2)                                           # 224
        self.hd0_0 = ConvUnit(self.ngf//2, self.ngf//2, self.n_groupf)

        self.hd_final = nn.Conv2d(self.ngf//2, self.out_c, kernel_size=3, stride=1, padding=1)
        self.hd_act = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        x = self.he0_1(self.he0_0(x))
        x = self.he1_1(self.he1_0(x))
        x = self.he2_1(self.he2_0(x))
        x = self.he3_1(self.he3_0(x))
        x = self.he4_1(self.he4_0(x))

        _shape = x.shape[-1]
        x = F.avg_pool2d(x, kernel_size=_shape, stride=1)
        x = x.view(-1, self.ngf*16)
        x = F.elu(self.he5_lin(x))

        return x

    def decoder(self, x):
        batch_size = x.shape[0]
        x = F.elu(self.hd5_lin(x))
        x = x.reshape(batch_size, -1, self.s_h, self.s_w)
        x = self.hd5_0(x)

        x = self.hd4_0(self.hd4_1(x))
        x = self.hd3_0(self.hd3_1(x))
        x = self.hd2_0(self.hd2_1(x))
        x = self.hd1_0(self.hd1_1(x))
        x = self.hd0_0(self.hd0_1(x))

        x = self.hd_act(self.hd_final(x))

        return x


class FCN8_VGG(nn.Module):
    """docstring for AutoEncoder"""
    def __init__(self, args, in_channels=3, out_channels=1):
        super(FCN8_VGG, self).__init__()
        self.args = args
        self.in_c = in_channels            # 3
        self.out_c = out_channels
        self.ngf = args.ngf                 # 32
        self.n_groupf = args.n_groupnorm     # nc_out/4
        self.z = args.enc_z
        self.s_h = int(args.resolution_high/32)
        self.s_w = int(args.resolution_wide/32)

        self.e1_1 = ConvUnit(self.in_c, self.ngf*2, self.n_groupf, bias=False)
        self.e1_2 = ConvUnit(self.ngf*2, self.ngf*2, self.n_groupf, bias=False)                                             # 224
        self.e1_pool = nn.MaxPool2d(2, 2)                                                                                   # 112

        self.e2_1 = ConvUnit(self.ngf*2, self.ngf*4, self.n_groupf, bias=False)
        self.e2_2 = ConvUnit(self.ngf*4, self.ngf*4, self.n_groupf, bias=False)                                             # 112
        self.e2_pool = nn.MaxPool2d(2, 2)                                                                                   # 56

        self.e3_1 = ConvUnit(self.ngf*4, self.ngf*8, self.n_groupf, bias=False)
        self.e3_2 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, bias=False)
        self.e3_3 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, bias=False)                                             # 56
        self.e3_pool = nn.MaxPool2d(2, 2)                                                                                   # 28

        self.d3 = nn.Conv2d(self.ngf*8, self.out_c, kernel_size=1, stride=1)                                                # 28

        self.e4_1 = ConvUnit(self.ngf*8, self.ngf*16, self.n_groupf, bias=False)
        self.e4_2 = ConvUnit(self.ngf*16, self.ngf*16, self.n_groupf, bias=False)
        self.e4_3 = ConvUnit(self.ngf*16, self.ngf*16, self.n_groupf, bias=False)                                           # 28
        self.e4_pool = nn.MaxPool2d(2, 2)                                                                                   # 14                                                                                  # 7

        self.d4_1 = ConvUnit(self.ngf*16, self.ngf*32, self.n_groupf, bias=False)
        self.d4_2 = ConvUnit(self.ngf*32, self.ngf*32, self.n_groupf, bias=False)
        self.d4 = nn.Conv2d(self.ngf*32, self.out_c, kernel_size=1, stride=1)                                               # 14
        self.d4_up = DeconvUnit(self.out_c, self.out_c, n_groupf=2, kernel_size=2, stride=2, padding=0)                     # 28

        self.final_upsample = nn.Upsample(size=224, mode='bilinear')
        self.final_act = nn.Softmax2d()

    def forward(self, x):
        x = self.e1_1(x)
        x = self.e1_2(x)
        x = self.e1_pool(x)

        x = self.e2_1(x)
        x = self.e2_2(x)
        x = self.e2_pool(x)

        x = self.e3_1(x)
        x = self.e3_2(x)
        x = self.e3_3(x)
        x = self.e3_pool(x)
        d3 = self.d3(x)

        x = self.e4_1(x)
        x = self.e4_2(x)
        x = self.e4_3(x)
        x = self.e4_pool(x)

        x = self.d4_1(x)
        x = self.d4_2(x)
        x = self.d4(x)
        d4 = self.d4_up(x)

        out = d3 + d4
        out = self.final_upsample(out)
        out = self.final_act(out)

        return out

class FCN4_VGG(nn.Module):
    """docstring for AutoEncoder"""
    def __init__(self, args, in_channels=3, out_channels=1):
        super(FCN4_VGG, self).__init__()
        self.args = args
        self.in_c = in_channels            # 3
        self.out_c = out_channels
        self.ngf = args.ngf                 # 32
        self.n_groupf = args.n_groupnorm     # nc_out/4
        self.z = args.enc_z
        self.s_h = int(args.resolution_high/32)
        self.s_w = int(args.resolution_wide/32)

        self.e1_1 = ConvUnit(self.in_c, self.ngf*2, self.n_groupf, bias=False)
        self.e1_2 = ConvUnit(self.ngf*2, self.ngf*2, self.n_groupf, bias=False)                                             # 224
        self.e1_pool = nn.MaxPool2d(2, 2)                                                                                   # 112

        self.e2_1 = ConvUnit(self.ngf*2, self.ngf*4, self.n_groupf, bias=False)
        self.e2_2 = ConvUnit(self.ngf*4, self.ngf*4, self.n_groupf, bias=False)                                             # 112
        self.e2_pool = nn.MaxPool2d(2, 2)                                                                                   # 56
        self.d2 = nn.Conv2d(self.ngf*4, self.out_c, kernel_size=1, stride=1)                                                # 56

        self.e3_1 = ConvUnit(self.ngf*4, self.ngf*8, self.n_groupf, bias=False)
        self.e3_2 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, bias=False)
        self.e3_3 = ConvUnit(self.ngf*8, self.ngf*8, self.n_groupf, bias=False)                                             # 56
        self.e3_pool = nn.MaxPool2d(2, 2)                                                                                   # 28
        self.d3 = nn.Conv2d(self.ngf*8, self.out_c, kernel_size=1, stride=1)                                                # 28
        self.d3_up = DeconvUnit(self.out_c, self.out_c, n_groupf=2, kernel_size=2, stride=2, padding=0)                     # 56

        self.e4_1 = ConvUnit(self.ngf*8, self.ngf*16, self.n_groupf, bias=False)
        self.e4_2 = ConvUnit(self.ngf*16, self.ngf*16, self.n_groupf, bias=False)
        self.e4_3 = ConvUnit(self.ngf*16, self.ngf*16, self.n_groupf, bias=False)                                           # 28
        self.e4_pool = nn.MaxPool2d(2, 2)                                                                                   # 14                                                                                  # 7

        self.d4_1 = ConvUnit(self.ngf*16, self.ngf*32, self.n_groupf, bias=False)
        self.d4_2 = ConvUnit(self.ngf*32, self.ngf*32, self.n_groupf, bias=False)
        self.d4 = nn.Conv2d(self.ngf*32, self.out_c, kernel_size=1, stride=1)                                               # 14
        self.d4_up = DeconvUnit(self.out_c, self.out_c, n_groupf=2, kernel_size=2, stride=2, padding=0)                     # 28

        self.final_upsample = nn.Upsample(size=224, mode='bilinear')
        self.final_act = nn.Softmax2d()

    def forward(self, x):
        x = self.e1_1(x)
        x = self.e1_2(x)
        x = self.e1_pool(x)

        x = self.e2_1(x)
        x = self.e2_2(x)
        x = self.e2_pool(x)
        d2 = self.d2(x)

        x = self.e3_1(x)
        x = self.e3_2(x)
        x = self.e3_3(x)
        x = self.e3_pool(x)
        d3 = self.d3(x)

        x = self.e4_1(x)
        x = self.e4_2(x)
        x = self.e4_3(x)
        x = self.e4_pool(x)

        x = self.d4_1(x)
        x = self.d4_2(x)
        x = self.d4(x)
        d4 = self.d4_up(x)

        out1 = d3+d4
        out1 = self.d3_up(out1)

        out2 = out1 + d2
        out = self.final_upsample(out2)
        out = self.final_act(out)

        return out

