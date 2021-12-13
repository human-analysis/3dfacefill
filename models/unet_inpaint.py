import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['UnetSelfAttnConf32']

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out, attention
        else:
            return out

class ConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, act=True):
        super(ConvUnit, self).__init__()

        self.activate = act
        padding = dilation
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.norm = nn.GroupNorm(out_ch//4, out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activate:
            return self.act(out)
        else:
            return out

class UpConvUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, dilation=1, act=True, bilinear=False):
        super(UpConvUnit, self).__init__()

        self.activate = act

        if bilinear:
            self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),  nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)))
        else:
            padding=dilation
            self.conv = nn.utils.spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))

        self.norm = nn.GroupNorm(out_ch//4, out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activate:
            return self.act(out)
        else:
            return out

# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super(DoubleConv, self).__init__()
        self.stride = stride
        padding = dilation
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding, dilation=dilation)),
            nn.GroupNorm(out_ch//4, out_ch),
            nn.ELU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=padding, dilation=dilation)),
            nn.GroupNorm(out_ch//4, out_ch))
        self.act = nn.ELU(inplace=True)

        self.res_conv = None
        self.res_norm = None
        if in_ch != out_ch or stride != 1:
            self.res_conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0))
            self.res_norm = nn.GroupNorm(out_ch//4, out_ch)

    def forward(self, x):
        residual = x

        x = self.conv(x)

        if self.res_conv is not None:
            residual = self.res_norm(self.res_conv(residual))

        x += residual
        x = self.act(x)

        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(Down, self).__init__()
        self.mpconv = DoubleConv(in_ch, out_ch, stride=2, dilation=dilation)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.utils.spectral_norm(nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2))

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UpSingle(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False, act=False, bilinear=False):
        super(UpSingle, self).__init__()

        if bilinear:
            layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)),
                nn.GroupNorm(out_ch//4, out_ch),
                nn.ELU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1))]
        else:
            layers = [nn.utils.spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)),
                nn.GroupNorm(out_ch//4, out_ch),
                nn.ELU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1))]

        if norm:
            layers.append(nn.GroupNorm(out_ch//4, out_ch))
        if act:
            layers.append(nn.ELU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x):
        x = self.conv(x)
        return x

class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, dir=None, bilinear=False, act=True, norm=True):
        super(GatedConv, self).__init__()
        if dir == 'down':
            self.feat_conv = Down(in_ch, out_ch)
            self.gate_conv = ConvUnit(in_ch, out_ch, stride=2, act=False)
        elif dir == 'up':
            self.feat_conv = Up(in_ch, out_ch, bilinear=bilinear)
            self.gate_conv = UpConvUnit(in_ch//2, out_ch, act=False, bilinear=False)
        elif dir == 'upsingle':
            self.feat_conv = UpSingle(in_ch, out_ch, norm=True, act=True)#, bilinear=bilinear)
            self.gate_conv = UpConvUnit(in_ch, out_ch, act=False, bilinear=False)
        elif dir is None:
            self.feat_conv = DoubleConv(in_ch, out_ch)
            self.gate_conv = ConvUnit(in_ch, out_ch, act=False)

        self.act = nn.Sigmoid()

    def forward(self, x, x_skip=None):
        g = self.act(self.gate_conv(x))

        if x_skip is not None:
            f = self.feat_conv(x, x_skip)
        else:
            f = self.feat_conv(x)
        return f, g


class UnetSelfAttnConf32(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, final_act='Softmax2d', dropout_p=0, use_attn=True):
        super(UnetSelfAttnConf32, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  out_channels
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        self.use_attn =use_attn

        self.down1 = GatedConv(in_channels, 32, 'down')
        self.down1_flip = GatedConv(in_channels, 32, 'down')
        self.down2 = GatedConv(64, 64, 'down')
        self.down3 = GatedConv(64, 128, 'down')
        self.down4 = GatedConv(128, 256, 'down')
        self.down5 = GatedConv(256, 512, 'down')
        self.interm = GatedConv(512, 256)
        self.up1 = GatedConv(512, 128, 'up', bilinear=bilinear)
        self.up2 = GatedConv(256, 64, 'up', bilinear=bilinear)
        if self.use_attn:
            self.attn = Self_Attn(64, 'relu', with_attn=True)
        self.up3 = GatedConv(128, 64, 'up', bilinear=bilinear)
        self.up4 = GatedConv(128, 64, 'up', bilinear=bilinear)
        self.up5 = GatedConv(64, 32, 'upsingle', bilinear=bilinear)
        self.outc = OutConv(32, out_channels+1)

    def forward(self, x, mask=None):
        if mask is not None:
            x = torch.cat((x, mask), dim=1)

        f1, g1 = self.down1(x)
        f1_flip, g1_flip = self.down1_flip(x.flip(dims=(3,)))
        x1 = torch.cat((f1*g1, f1_flip*g1_flip), dim=1)

        f2, g2 = self.down2(x1)
        x2 = f2*g2

        f3, g3 = self.down3(x2)
        x3 = f3*g3

        f4, g4 = self.down4(x3)
        x4 = f4*g4

        f5, g5 = self.down5(x4)
        x5 = f5*g5

        f6, g6 = self.interm(x5)
        x6 = f6*g6

        f, g = self.up1(x6, x4)
        x = f*g

        f, g = self.up2(x, x3)
        x = f*g

        # attn = torch.ones_like(x)[:,0:1]
        if not self.use_attn:
            attn = torch.ones_like(x)[:,0:1]
        else:
            x, attn = self.attn(x)

        f, g = self.up3(x, x2)
        x = f*g

        f, g = self.up4(x, x1)
        x = f*g

        f, g = self.up5(x)
        x = f*g

        x = self.outc(x)
        out = x[:,:-1].clamp(-1, 1)
        conf = x[:,-1].unsqueeze(1)

        return out, (torch.cat((g1, g1_flip), 1), g2, attn), conf
