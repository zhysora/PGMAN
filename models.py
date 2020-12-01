import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import numpy as np
from functions import *
from torch.nn.functional import interpolate

############## items
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, withBN=True, Norm='BN'):
        super(ResBlock, self).__init__()
        self.basic = []
        self.basic.append(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding))
        if withBN:
            if Norm is 'BN':
                self.basic.append(nn.BatchNorm2d(out_channel))
            elif Norm is 'IN':
                self.basic.append(nn.InstanceNorm2d(out_channel))
        self.basic.append(nn.ReLU(True))
        self.basic.append(nn.Conv2d(out_channel,out_channel,kernel_size,stride,padding))
        if withBN:
            if Norm is 'BN':
                self.basic.append(nn.BatchNorm2d(out_channel))
            elif Norm is 'IN':
                self.basic.append(nn.InstanceNorm2d(out_channel))
        self.basic = nn.Sequential(*self.basic)
    
    def forward(self, x):
        return self.basic(x) + x

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
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
        return out,attention

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

## if n=0, then use pixelGAN (rf=1)
## else rf is 16 if n=1
##            34 if n=2
##            70 if n=3
##            142 if n=4
##            286 if n=5
##            574 if n=6
class Pixel_Discriminator(nn.Module):
    def __init__(self, in_channels, ndf, withBN, Norm='BN'):
        super(Pixel_Discriminator, self).__init__()
        self.netD = []
        self.netD.append( nn.Conv2d(in_channels, ndf, 1, 1) ) # 256 * 256 * 64
        self.netD.append( nn.LeakyReLU(0.2, True) ) 
        self.netD.append( nn.Conv2d(ndf, ndf * 2, 1, 1) ) # 256 * 256 * 128
        if withBN:
            if Norm is 'BN':
                self.netD.append( nn.BatchNorm2d(ndf * 2) )
            elif Norm is 'IN':
                self.netD.append( nn.InstanceNorm2d(ndf * 2) )
        self.netD.append( nn.LeakyReLU(0.2, True) )
        self.netD.append( nn.Conv2d(ndf * 2, 1, 1) ) # 256 * 256 * 128
        self.netD = nn.Sequential( *self.netD )
        
    def forward(self, x):
        return self.netD(x)

class Patch_Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, n_layers=3, withBN=True, Norm='BN'):
        super(Patch_Discriminator, self).__init__()
        self.netD = []
        self.netD.append( nn.Conv2d(in_channels, ndf, 4, 2, 1) ) # 256 * 256 * 64
        self.netD.append( nn.LeakyReLU(0.2, True) )

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.netD.append( nn.Conv2d( ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1) )
            if withBN:
                if Norm is 'BN':
                    self.netD.append( nn.BatchNorm2d(ndf * nf_mult) )
                elif Norm is 'IN':
                    self.netD.append( nn.InstanceNorm2d(ndf * nf_mult) )
            self.netD.append( nn.LeakyReLU(0.2, True) )

        # N * N * (ndf*M)
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.netD.append( nn.Conv2d( ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1) ) # (N-1) * (N-1) * (ndf*M*2)
        if withBN:
            if Norm is 'BN':
                self.netD.append( nn.BatchNorm2d(ndf * nf_mult) )
            elif Norm is 'IN':
                self.netD.append( nn.InstanceNorm2d(ndf * nf_mult) )
        self.netD.append( nn.LeakyReLU(0.2, True) )

        self.netD.append( nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1) ) # (N-2) * (N-2) * 1
        self.netD = nn.Sequential( *self.netD )

    def forward(self, x):
        return self.netD(x)

############## PGMAN
class PGMAN_Generator(nn.Module):
    def __init__(self, withBN=True, high_pass=False, res_layer=3, Norm='BN'):
        super(PGMAN_Generator, self).__init__()
        self.high_pass = high_pass

        self.extractor_lr = []
        self.extractor_lr.append( nn.Conv2d(4, 32, 7, 1, 3) ) # 32 x 64 x 64
        if withBN:
            if Norm is 'BN':
                self.extractor_lr.append( nn.BatchNorm2d(32) )
            elif Norm is 'IN':
                self.extractor_lr.append( nn.InstanceNorm2d(32) )
        self.extractor_lr.append( nn.ReLU() )   
        self.extractor_lr.append( nn.Conv2d(32, 64, 3, 1, 1) ) # 64 x 64 x 64
        if withBN:
            if Norm is 'BN':
                self.extractor_lr.append( nn.BatchNorm2d(64) )
            elif Norm is 'IN':
                self.extractor_lr.append( nn.InstanceNorm2d(64) )
        self.extractor_lr.append( nn.ReLU() )
        self.extractor_lr.append( nn.Conv2d(64, 128, 3, 1, 1) ) # 128 x 64 x 64
        if withBN:
            if Norm is 'BN':
                self.extractor_lr.append( nn.BatchNorm2d(128) )
            elif Norm is 'IN':
                self.extractor_lr.append( nn.InstanceNorm2d(128) )
        self.extractor_lr = nn.Sequential( *self.extractor_lr )

        self.extractor_pan = []
        self.extractor_pan.append( nn.Conv2d(1, 32, 7, 1, 3) ) # 32 x 256 x 256
        if withBN:
            if Norm is 'BN':
                self.extractor_pan.append( nn.BatchNorm2d(32) )
            elif Norm is 'IN':
                self.extractor_pan.append( nn.InstanceNorm2d(32) )
        self.extractor_pan.append( nn.ReLU() )
        self.extractor_pan.append( nn.Conv2d(32, 64, 3, 2, 1) ) # 64 x 128 x 128
        if withBN:
            if Norm is 'BN':
                self.extractor_pan.append( nn.BatchNorm2d(64) )
            elif Norm is 'IN':
                self.extractor_pan.append( nn.InstanceNorm2d(64) )
        self.extractor_pan.append( nn.ReLU() )
        self.extractor_pan.append( nn.Conv2d(64, 128, 3, 2, 1) ) # 128 x 64 x 64 
        if withBN:
            if Norm is 'BN':
                self.extractor_pan.append( nn.BatchNorm2d(128) )
            elif Norm is 'IN':
                self.extractor_pan.append( nn.InstanceNorm2d(128) )
        self.extractor_pan = nn.Sequential( *self.extractor_pan )

        self.res = []
        for _ in range(res_layer):
            self.res.append( nn.ReLU() )
            self.res.append( ResBlock(256, 256, 3, 1, 1, withBN, Norm) ) # 256 x 64 x 64 
        self.res.append( nn.ReLU() )  
        self.res.append( nn.ConvTranspose2d(256, 128, 2, 2) ) # 128 x 128 x 128
        if withBN:
            if Norm is 'BN':
                self.res.append( nn.BatchNorm2d(128) )
            elif Norm is 'IN':
                self.res.append( nn.InstanceNorm2d(128) )
        self.res.append( nn.ReLU() )  
        self.res.append( nn.ConvTranspose2d(128, 64, 2, 2) ) # 64 x 256 x 256 
        if withBN:
            if Norm is 'BN':
                self.res.append( nn.BatchNorm2d(64) )
            elif Norm is 'IN':
                self.res.append( nn.InstanceNorm2d(64) )
        self.res.append( nn.ReLU() )  
        self.res.append( nn.Conv2d(64, 4, 7, 1, 3) ) # 4 x 256 x 256
        self.res = nn.Sequential( *self.res )

    def forward(self, pan, lr_u, lr):
        if self.high_pass:
            ms_hp = get_edge(lr)
            pan_hp = get_edge(pan)
            lr_feat = self.extractor_lr(ms_hp)
            pan_feat = self.extractor_pan(pan_hp)
            res = self.res( torch.cat((lr_feat, pan_feat), dim=1) ) + lr_u
        else:
            lr_feat = self.extractor_lr(lr)
            pan_feat = self.extractor_pan(pan)
            res = self.res( torch.cat((lr_feat, pan_feat), dim=1) )
        return res