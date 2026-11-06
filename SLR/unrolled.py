import torch
import torch.nn as nn
from torch.nn import init
import os
import torch.nn.functional as F
from utils import ifft2c, fft2c, r2c, c2r
import numpy as np
#from layer import *
from torch.autograd import Variable
import torch.autograd as autograd
from spectral_norm import SpectralNorm

conv_size = 64
filter_size = 3


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.05):  #init_type='xavier', gain=1.0
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

#def r2c(x):
#    #out = torch.complex(x[:, 0:12, ...], x[:, 12:24, ...]) #torch.unsqueeze(, 1)
#    re, im = torch.chunk(x,2,1)
#    out = re +1j*im
#    return out
#def c2r(x):
#    out = torch.cat([torch.real(x), torch.imag(x)], dim=1)
#    return out

class SLRH(BaseNetwork):
    def __init__(self,LayerNo=10,conv_size=64, nch=30, filter_size=3, pad=1, bias=False):
        super(SLRH, self).__init__()
        self.LayerNo = LayerNo
        Layers = []

        for i in range(LayerNo):
            Layers.append(SLRBlocknH(conv_size=conv_size, nch=nch, filter_size=filter_size, pad=pad, bias=bias))

        self.fcs = nn.ModuleList(Layers)

    def forward(self, x, mask, atb):
        for i in range(self.LayerNo):           
            x = self.fcs[i](x, mask, atb)            
        return x

class SLRBlocknH(BaseNetwork):
    def __init__(self, conv_size=64, nch=24, filter_size=3, pad=1, bias=False):
        super(SLRBlocknH, self).__init__()
#
        self.lam = nn.Parameter(torch.Tensor([.4]))
        self.mu = nn.Parameter(torch.Tensor([.1]))
#        self.tau = nn.Parameter(torch.Tensor([.5]))
##        
        self.conv1 = SpectralNorm(nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv2 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv3 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv4 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv5 = SpectralNorm(nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias))
##
        self.conv6 = nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv7 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv8 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv9 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv10 = nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias)
#        
        self.init_weights()
        
    def forward(self, y, mask, atb):
                  
        x = self.conv1(y)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = y + x        
         
        t = r2c(y)
        t = ifft2c(t)             
        t = c2r(t)              
        z = self.conv6(t)
        z = F.relu(z)
        z = self.conv7(z)
        z = F.relu(z)
        z = self.conv8(z)
        z = F.relu(z) 
        z = self.conv9(z)
        z = F.relu(z)
        z = self.conv10(z)  
        z = t + z  
        z = r2c(z)
        z = fft2c(z) # torch.fft.fft2(z, norm="ortho")  #  
        z = c2r(z)
               

        y = self.lam * x + self.mu * z

        y = (1 - mask) * y + mask * atb

        return y



#
#class SLR1(torch.nn.Module):
#    def __init__(self,LayerNo=10,conv_size=64, nch=30, filter_size=3, pad=1, bias=False):
#        super(SLR1, self).__init__()
#        self.LayerNo = LayerNo
#        Layers = []
#
#        for i in range(LayerNo):
#            Layers.append(SLR1Blockn(conv_size=conv_size, nch=nch, filter_size=filter_size, pad=pad, bias=bias))
#
#        self.fcs = nn.ModuleList(Layers)
#
#    def forward(self, x, mask, atb):
#        for i in range(self.LayerNo):           
#            x = self.fcs[i](x, mask, atb)            
#        return x
#
#
#class SLR1Blockn(torch.nn.Module):
#    def __init__(self, conv_size=64, nch=24, filter_size=3, pad=1, bias=False):
#        super(SLR1Blockn, self).__init__()
##
#        self.lam = nn.Parameter(torch.Tensor([.5]))
##        self.mu = nn.Parameter(torch.Tensor([.5]))
##        self.tau = nn.Parameter(torch.Tensor([.5]))
#        
#        self.conv1 = SpectralNorm(nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
#        self.conv2 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
#        self.conv3 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
#        self.conv4 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
#        self.conv5 = SpectralNorm(nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias))
#### 
##        self.conv6 = SpectralNorm(nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
##        self.conv7 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
##        self.conv8 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
##        self.conv9 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
##        self.conv10 = SpectralNorm(nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias))
###        
##        self.conv1 = nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad, bias=bias)
##        self.conv2 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
##        self.conv3 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
##        self.conv4 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
##        self.conv5 = nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias)
###
#        self.conv6 = nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv7 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv8 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv9 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv10 = nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias)
##    
#    def forward(self, y, mask, atb):
#                  
#        x = self.conv1(y)
#        x = F.relu(x)
#        x = self.conv2(x)
#        x = F.relu(x)
#        x = self.conv3(x)
#        x = F.relu(x)
#        x = self.conv4(x)
#        x = F.relu(x)
#        x = self.conv5(x)
##        x = 0.6 * y + 0.4 * x        
#        lam = torch.clamp(self.lam, 0.01, 0.98)
#        x = lam * y + (.99 - lam) * x
#         
#        t = r2c(y)
#        t = ifft2c(t)  # torch.fft.ifft2(y, norm="ortho")   #              
#        t = c2r(t)              
#        z = self.conv6(t)
#        z = F.relu(z)
#        z = self.conv7(z)
#        z = F.relu(z)
#        z = self.conv8(z)
#        z = F.relu(z) 
#        z = self.conv9(z)
#        z = F.relu(z)
#        z = self.conv10(z)  
#      
##        z = z + self.mu*y
#        z = 0.6 * t + 0.4 * z 
#
#        z = r2c(z)
#        z = fft2c(z) # torch.fft.fft2(z, norm="ortho")  #  
#        z = c2r(z)
#        
#        y = 0.9*x + 0.1*z
##        y = x + z
#       # print('mask', mask.shape)
#        y = (1 - mask) * y + atb
#        #print('lam', lam)
#        return y
#


class SLRK(torch.nn.Module):
    def __init__(self,LayerNo=10,conv_size=64, nch=24, filter_size=3, pad=1, bias=False):
        super(SLRK, self).__init__()
        self.LayerNo = LayerNo
        Layers = []

        for i in range(LayerNo):
            Layers.append(SLRBlocknK(conv_size=conv_size, nch=nch, filter_size=filter_size, pad=pad, bias=bias))

        self.fcs = nn.ModuleList(Layers)

    def forward(self, x, mask, atb):
        for i in range(self.LayerNo):           
            x = self.fcs[i](x, mask, atb)            
        return x   
        
        
class SLRBlocknK(torch.nn.Module):
    def __init__(self, conv_size=64, nch=24, filter_size=3, pad=1, bias=False):
        super(SLRBlocknK, self).__init__()


        self.lam = nn.Parameter(torch.Tensor([.1]))
        self.mu = nn.Parameter(torch.Tensor([.4]))

#        self.conv1 = nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv2 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv3 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv4 = nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias)
#        self.conv5 = nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias)
        self.conv1 = SpectralNorm(nn.Conv2d(nch, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv2 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv3 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv4 = SpectralNorm(nn.Conv2d(conv_size, conv_size, kernel_size=filter_size, padding=pad,bias=bias))
        self.conv5 = SpectralNorm(nn.Conv2d(conv_size, nch, kernel_size=filter_size, padding=pad,bias=bias))
        
    def forward(self, y, mask, atb):
    
        x = self.conv1(y)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        
        y = self.lam * x + self.mu * y

        y = (1 - mask) * y + mask * atb

        return y     
        
class SLRL(torch.nn.Module):
    def __init__(self,LayerNo=10,conv_size=64, nch=24, filter_size=3, pad=1, bias=False):
        super(SLRL, self).__init__()
        self.LayerNo = LayerNo
        Layers = []

        for i in range(LayerNo):
            Layers.append(SLRBlocknL(conv_size=conv_size, nch=nch, filter_size=filter_size, pad=pad, bias=bias))

        self.fcs = nn.ModuleList(Layers)

    def forward(self, x, mask, atb):
        for i in range(self.LayerNo):           
            x = self.fcs[i](x, mask, atb)            
        return x

class SLRBlocknL(torch.nn.Module):
    def __init__(self,conv_size=64,nch=24, filter_size=3, pad=1, bias=False):
        super(SLRBlocknL, self).__init__()

        self.lam = nn.Parameter(torch.Tensor([.1]))
        self.mu = nn.Parameter(torch.Tensor([.4]))
        
        self.conv_1 = nn.Parameter(
            nn.init.xavier_normal_(torch.Tensor(conv_size, nch, filter_size, filter_size), gain=1))
        self.conv_2 = nn.Parameter(
            nn.init.xavier_normal_(torch.Tensor(conv_size, conv_size, filter_size, filter_size), gain=1))
        self.conv_3 = nn.Parameter(
            nn.init.xavier_normal_(torch.Tensor(conv_size, conv_size, filter_size, filter_size), gain=1))
#        self.conv_4 = nn.Parameter(
#            nn.init.xavier_normal_(torch.Tensor(conv_size, conv_size, filter_size, filter_size), gain=1))
#        self.conv_5 = nn.Parameter(
#            nn.init.xavier_normal_(torch.Tensor(nch, conv_size, filter_size, filter_size), gain=1))
        

    def forward(self, y, mask, atb):
        x = F.conv2d(y, self.conv_1, padding=1)
        x = F.conv2d(x, self.conv_2, padding=1)
        x = F.conv2d(x, self.conv_3, padding=1)
#        x = F.conv2d(x, self.conv_4, padding=1)      
#        x = F.conv2d(x, self.conv_5, padding=1)
        
        x = F.relu(x)

 #       x = F.conv_transpose2d(x, self.conv_5, padding=1)
#        x = F.relu(x)
 #       x = F.conv_transpose2d(x, self.conv_4, padding=1)
#        x = F.relu(x)
        x = F.conv_transpose2d(x, self.conv_3, padding=1)
        x = F.conv_transpose2d(x, self.conv_2, padding=1)
        x = F.conv_transpose2d(x, self.conv_1, padding=1)

        y = self.lam* x   + self.mu*y      
        y = (1 - mask) * y + atb                                     
       
        return y