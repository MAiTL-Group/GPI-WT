import torch
import numpy as np
import os
sqrt = np.sqrt
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift
import torch.fft as FFT

def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)

def ifft2c(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def fft2c(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x =  torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def r2c(x):
    re, im = torch.chunk(x,2,1)
    x = torch.complex(re,im)
    return x


def c2r(x):
    x = torch.cat([torch.real(x),torch.imag(x)],1)
    return x


#
#def sos(x):
#    #print(x.dtype)
#    #xr, xi = torch.chunk(x,2,1)
#   # if x.dtype == 'torch.complex64':
#    xr = torch.real(x)
#    xi = torch.imag(x)
#    x = torch.pow(torch.abs(xr),2)+torch.pow(torch.abs(xi),2)
#    x = torch.sum(x, dim=1)
#    x = torch.pow(x,0.5)
#    x = torch.unsqueeze(x,1)
#    return x

def sos(x):
    #xr, xi = torch.chunk(x,2,1)
   # if x.dtype == 'torch.complex64':
    xr = np.real(x)
    xi = np.imag(x)
    x = np.power(np.abs(xr),2)+np.power(np.abs(xi),2)
    x = np.sum(x, 1)
    x = np.power(x,0.5)
    #x = np.unsqueeze(x,1)
    return x
    
    

def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)

    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) / np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3) / np.math.sqrt(ny)
    return x


def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) * np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3) * np.math.sqrt(ny)
    return x
    
def normalize(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)  

def myPSNR(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed     images"""
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    return psnr