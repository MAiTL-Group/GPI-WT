import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
#import tensorflow as tf
import logging
sqrt = np.sqrt
import torch.nn.functional as F
import torchvision.transforms as T
#import sigpy as sp
#from icecream import ic
from tqdm import tqdm
from scipy.linalg import null_space, svd
#from optimal_thresh import optht
#import sigpy as sp
#import sigpy.mri.app as MR
#from torch.utils.dlpack import to_dlpack, from_dlpack
#from cupy import from_dlpack as cu_from_dlpack
#import pytorch_wavelets as wavelets

def crop(img, cropc, cropx, cropy):
    if img.dim() == 4:
        nb, c, x, y = img.shape
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, startc:startc + cropc, startx:startx + cropx, starty: starty + cropy]
    elif img.dim() == 3:
        c, x, y = img.shape
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[startc:startc + cropc, startx:startx + cropx, starty: starty + cropy]   
    return cimg

def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sn, sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = torch.zeros((sn, sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            # res[:, count, :] = torch.reshape(
            #     im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz), order='F')
            res[:, :, count, :] = torch.reshape(
                im[:, x:sx-wx+x+1, y:sy-wy+y+1, :], (sn, sh, sz))
            count += 1
    return res

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''
    tmp = im2row(data, kernel_size)
    tsn, tsx, tsy, tsz = tmp.shape[:]
    #A = torch.reshape(tmp, (tsx, tsy*tsz))
    A = reshape_fortran(tmp, (tsn, tsx, tsy*tsz))
    #print('a', A, A.shape, A.dtype) #torch.Size([1, 400, 375]) torch.complex64
    #print(A.T.shape)
    return torch.matmul(A.permute(0,2,1).conj(), A)

def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):
    kx, ky = kernel_size[:]
    if sampling is None:
        sampling = torch.ones((*kernel_size, ncoils))
    dummyK = torch.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), coil] = 1
    #print('dummyK', dummyK.shape) dummyK torch.Size([5, 5, 15])
    idxY = torch.where(dummyK)
    #print('idxY', idxY) #idxY (tensor([2]), tensor([2]), tensor([0]))
    
    
    idxY_flat = torch.sort((torch.arange(dummyK.numel()).reshape(ncoils, ky, kx))[idxY[2], idxY[1], idxY[0]])[0]
    #idxY_flat = np.sort(np.ravel_multi_index(idxY, dummyK.shape, order='F'))
    #print('idxY_flat', idxY_flat)
    sampling[idxY] = 0
    idxA = torch.where(sampling)
    #print('idxA', idxA)
    idxA_flat = torch.sort( (torch.arange(sampling.numel()).reshape(sampling.shape[-1],
                      sampling.shape[-2],sampling.shape[-3]))[idxA[2], idxA[1], idxA[0]] )[0]
    #idxA_flat = np.sort(np.ravel_multi_index(idxA, sampling.shape, order='F'))
    #print('idxA_flat', idxA_flat[0])
    #print('AtA', AtA[0,0,0], AtA.shape)

    Aty = AtA[:, :, idxY_flat]
    Aty = Aty[:, idxA_flat]

    AtA0 = AtA[:, idxA_flat, :]
    AtA0 = AtA0[:, :, idxA_flat]

    #print('sampling', sampling.shape)
    kernel = torch.zeros(AtA.shape[0], sampling.numel(), dtype=AtA0.dtype)
    lamda = torch.norm(AtA0,dim=(1,2))/AtA0.shape[1]*lamda
    #print('lamda', lamda, lamda.shape)
    rawkernel = torch.linalg.solve(AtA0 + torch.eye(AtA0.shape[1])*lamda, Aty)
    #print('rawkernel', rawkernel[0,0], rawkernel.shape) #awkernel tensor([-0.0098-0.0095j]) torch.Size([1, 374, 1])
    #print('kernel', kernel.shape)
    kernel[:, idxA_flat] = torch.squeeze(rawkernel, dim=2) #rawkernel.squeeze()
    kernel = reshape_fortran(kernel, (AtA.shape[0],  *sampling.shape)) #torch.reshape(kernel, sampling.shape, order='F')
  
    return kernel, rawkernel

#def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):
#
#    kx, ky = kernel_size[:]
#    if sampling is None:
#        sampling = np.ones((*kernel_size, ncoils))
#    dummyK = np.zeros((kx, ky, ncoils))
#    dummyK[int(kx/2), int(ky/2), coil] = 1
#
#    idxY = np.where(dummyK)
#    idxY_flat = np.sort(
#        np.ravel_multi_index(idxY, dummyK.shape, order='F'))
#    sampling[idxY] = 0
#    idxA = np.where(sampling)
#    idxA_flat = np.sort(
#        np.ravel_multi_index(idxA, sampling.shape, order='F'))
#
#    Aty = AtA[:, idxY_flat]
#    Aty = Aty[idxA_flat]
#
#    AtA0 = AtA[idxA_flat, :]
#    AtA0 = AtA0[:, idxA_flat]
#
#    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)
#    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda
#    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0])*lamda, Aty) # fast 1s
#
#    kernel[idxA_flat] = rawkernel.squeeze()
#    kernel = np.reshape(kernel, sampling.shape, order='F')
#
#    return(kernel, rawkernel)
#

def spirit_calibrate(acs, kSize, lamda=0.001, filtering=False, verbose=True): # lamda=0.01
    nbatch, nCoil = acs.shape[0], acs.shape[-1]
    AtA = dat2AtA(acs,kSize)
#    if filtering: # singular value threshing
#        if verbose:
#            ic('prefiltering w/ opth')
#        U,s,Vh = svd(AtA, full_matrices=False)
#        k = optht(AtA, sv=s, sigma=None)
#        if verbose:
#            print('{}/{} kernels used'.format(k, len(s)))
#        AtA= (U[:, :k] * s[:k] ).dot( Vh[:k,:])
    #print('ata', AtA.shape, AtA.dtype)  #ata torch.Size([1, 375, 375]) torch.complex64 
    spirit_kernel = torch.zeros(nbatch, nCoil, nCoil, *kSize, dtype=torch.complex64)
    for c in range(nCoil): #tqdm(range(nCoil)):
        tmp, _ = calibrate_single_coil(AtA,kernel_size=kSize,ncoils=nCoil,coil=c,lamda=lamda)
        spirit_kernel[:,c,:,:,:] = tmp.permute(0,3,1,2) #np.transpose(tmp,[2,0,1])
        
    spirit_kernel = spirit_kernel.permute(0,3,4,2,1)   #np.transpose(spirit_kernel,[2,3,1,0]) # Now same as matlab!
#    GOP = np.transpose(spirit_kernel[::-1,::-1],[3,2,0,1])
#    GOP = GOP.copy()
#    for n in range(nCoil):
#        GOP[n,n,kSize[0]//2,kSize[1]//2] = -1  
    return spirit_kernel


def complex_kernel_forward(filter, i):
    filter = torch.squeeze(filter[i])
    filter_real = torch.real(filter)
    filter_img = torch.imag(filter)
    kernel_real = torch.cat([filter_real, -filter_img], 1)
    kernel_imag = torch.cat([filter_img, filter_real], 1)
    kernel_complex = torch.cat([kernel_real, kernel_imag], 0)
    return kernel_complex


def conv2(x1, x2):
    return F.conv2d(x1, x2, padding='same')
    
def ksp2float(ksp, i):
    kdata = torch.squeeze(ksp[i])  # nb,nc,nx,ny
    if len(kdata.shape) == 3:
        kdata = torch.unsqueeze(kdata, 0)

    kdata_float = torch.cat([torch.real(kdata), torch.imag(kdata)], 1)
    return kdata_float

def spirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc, nx, ny
    """
    nb = ksp.shape[0]

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)

    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))

    re, im = torch.chunk(res_i, 2, 1)
    res = torch.complex(re, im) - ksp
    return res
    
def adjspirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc_s, nx, ny
    """

    nb = kernel.shape[0]
    filter = torch.permute(kernel, (0, 2, 1, 3, 4))
    filter = torch.conj(filter.flip(dims=[-2, -1]))

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)

    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))

    re, im = torch.chunk(res_i, 2, 1)

    res = torch.complex(re, im) - ksp

    return res
