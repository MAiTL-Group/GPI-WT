import time
import numpy as np
from scipy.io import loadmat
#from skimage.measure import compare_psnr, compare_ssim
import h5py
import os
import matplotlib.pyplot as plt
from utils import IFFT2c, FFT2c
import scipy.io as sio
from os.path import join
from natsort import ns, natsorted
from scipy.ndimage import zoom
import glob
import itertools
from torch.utils.data import Dataset,DataLoader
import torch
from utils_spirit import  spirit_crop, spirit_calibrate, spirit, adjspirit

def get_dataset(maskpath, mode, ny, nx, nc):
    
    dataset = FastMRIKneeDataSet(maskpath, mode, ny, nx, nc)
#    dataset = T1MRIKneeDataSet(maskpath, mode, ny, nx, nc)
  
    if mode == 'training':
        data = DataLoader(dataset, batch_size=1, shuffle=True, 
                            num_workers=0, pin_memory=True)
    elif mode == 'training28P':
        data = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)                            
    elif mode == 'testing28P':
        data = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)
    else:
        data = DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=True)

    print(mode, "data loaded")    
    return data


def get_all_files(folder, pattern='*'):
    files = glob.glob(os.path.join(folder, '*'))  # 获取文件夹中所有文件
    sorted_files = natsorted(files)  # 对文件名按照自然排序排序
    return sorted_files


class FastMRIKneeDataSet(Dataset):
    def __init__(self, maskpath, mode, ny, nx, nc):
        super(FastMRIKneeDataSet, self).__init__()
        #self.config = config
        self.ny, self.nx, self.nc = ny, nx, nc

        if mode == 'training':
            self.kspace_dir = '/data0/chentao/data/fastMRI/multicoil_train/kspace/'
            self.maps_dir = '/data0/chentao/data/fastMRI/multicoil_train/maps/'
            self.file_list = get_all_files(self.kspace_dir)[:200]
        elif mode == 'test':
            self.kspace_dir = '/data0/chentao/data/fastMRI/multicoil_test/kspace/'
            self.maps_dir = '/data0/chentao/data/fastMRI/multicoil_test/maps/'
            self.file_list = get_all_files(self.kspace_dir)[:5]
               
        elif mode == 'training28P':
            self.kspace_dir = '/data0/huayu/Aluochen/csmdataset/T1_data_33/'
            self.maps_dir = '/data0/huayu/Aluochen/csmdataset/Output_maps_33/'
            self.file_list = get_all_files(self.kspace_dir)[:28]
            
        elif mode == 'testing28P':
            self.kspace_dir = '/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/T1_data_3/'
            self.maps_dir = '/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/Output_maps_3/'
            self.file_list = get_all_files(self.kspace_dir)[:3]
        else:
            raise NotImplementedError

        self.mode = mode
        
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(
                self.kspace_dir, os.path.basename(file)))
            with h5py.File(os.path.join(self.kspace_dir, os.path.basename(file)), 'r') as data:
                if self.mode != 'sample':
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 5)
                else:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

        C = sio.loadmat(maskpath)    
        self.mask = C['mask'][:]


    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 被试者编号
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # 被试者扫描的帧数编号
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.kspace_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            if self.mode != 'sample':
                slice_idx = slice_idx + 5  # 去掉前5帧

            ksp_idx = data['kspace'][slice_idx] # 15x640x368
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = crop(IFFT2c(ksp_idx), cropy=self.ny, cropx=self.nx)
            ksp_idx = FFT2c(ksp_idx)

        nImg, *_ = ksp_idx.shape
#    
        M = np.tile(self.mask,[nImg,self.nc*2,1,1])
        mask = self.mask.astype(np.complex64)
        mask = np.tile(mask, [nImg, self.nc, 1, 1])  
        #print('mask', mask.shape, ksp_idx.shape) 
        #print('Undersampling')

        ksp, orgk, minv = generateUndersampled(ksp_idx, mask)
        
        cailb =  spirit_crop(np.transpose(ksp.squeeze(),(1,2,0)), 24, 24, self.nc)  
        kernel = spirit_calibrate(cailb, (7,7), lamda=0.16, filtering=False, verbose=True) #(5,5,12,12)
        #print('kernel', kernel.shape, kernel.dtype) #kernel (5, 5, 30, 30) complex128 (7, 7, 32, 32) complex128
        kernel =np.transpose(kernel,(3,2,0,1)).astype(np.complex64)
         
        
#        ksp = IFFT2c(ksp)    
#        orgk = IFFT2c(orgk) 
        ksp = c2r(ksp)
        orgk = c2r(orgk)

        ksp = np.asarray(np.squeeze(ksp, 0))
        orgk = np.asarray(np.squeeze(orgk, 0))
        M = np.asarray(np.squeeze(M, 0))
        mask = np.asarray(np.squeeze(mask, 0))
        kernel = np.asarray(kernel)
        return ksp, orgk, mask, kernel

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))


#class FastMRIKneeDataSet(Dataset):
#    def __init__(self, maskpath, mode):
#        super(FastMRIKneeDataSet, self).__init__()
#        #self.config = config
#
#        if mode == 'training':
#            self.kspace_dir = '/data0/chentao/data/fastMRI/multicoil_train/kspace/'
#            self.maps_dir = '/data0/chentao/data/fastMRI/multicoil_train/maps/'
#            self.file_list = get_all_files(self.kspace_dir)[:5]
#   
#        elif mode == 'val':
#            self.kspace_dir = '/data0/taofeng/kneetest/T1data/'
#            self.maps_dir = '/data0/taofeng/kneetest/csm/'
#        elif mode == 'test':
#            self.kspace_dir = '/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/T1_data_3/'
#            self.maps_dir = '/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/Output_maps_3/'
#            self.file_list = get_all_files(self.kspace_dir)[:5]
#        elif mode == 'datashift':
#            self.kspace_dir = '/data0/chentao/data/fastMRI_brain/brain_T2/'
#            self.maps_dir = '/data0/chentao/data/fastMRI_brain/output_maps/'
#        else:
#            raise NotImplementedError
#
#        self.mode = mode
#        
#        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
#        for idx, file in enumerate(self.file_list):
#            print('Input file:', os.path.join(
#                self.kspace_dir, os.path.basename(file)))
#            with h5py.File(os.path.join(self.kspace_dir, os.path.basename(file)), 'r') as data:
#                if self.mode != 'sample':
#                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 5)
#                else:
#                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])
#
#        # Create cumulative index for mapping
#        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'
#
#        C = sio.loadmat(maskpath)    
#        self.mask = C['mask'][:]
##       mask = np.transpose(mask, [1, 0])  
#
#    def __getitem__(self, idx):
#        # Convert to numerical
#        if torch.is_tensor(idx):
#            idx = idx.tolist()
#
#        # 被试者编号
#        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
#        print(idx, scan_idx)
#        # 被试者扫描的帧数编号
#        slice_idx = int(idx) if scan_idx == 0 else \
#            int(idx - self.slice_mapper[scan_idx] +
#                self.num_slices[scan_idx] - 1)
#
##        # Load maps for specific scan and slice
##        maps_file = os.path.join(self.maps_dir,
##                                 os.path.basename(self.file_list[scan_idx]))
##        with h5py.File(maps_file, 'r') as data:
##            # 去掉前5帧
##            if self.mode != 'sample':
##                slice_idx = slice_idx + 5
##            maps_idx = data['s_maps'][slice_idx]
##            maps_idx = np.expand_dims(maps_idx, 0)
##            maps_idx = crop(maps_idx, cropx=368, cropy=368)
##            maps_idx = np.squeeze(maps_idx, 0)
##            maps = np.asarray(maps_idx)
#
#        # Load raw data for specific scan and slice
#        raw_file = os.path.join(self.kspace_dir,
#                                os.path.basename(self.file_list[scan_idx]))
#        with h5py.File(raw_file, 'r') as data:
#            if self.mode != 'sample':
#                slice_idx = slice_idx + 5  # 去掉前5帧
#            print(slice_idx)
#            ksp_idx = data['kspace'][slice_idx] # 15x640x368
#            ksp_idx = np.expand_dims(ksp_idx, 0)
#            ksp_idx = crop(IFFT2c(ksp_idx), cropx=320, cropy=320)
#            ksp_idx = FFT2c(ksp_idx)
#            
#            #if self.config.data.normalize_type == 'minmax':
#                #img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
#                #img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
#                #ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
#            #elif self.config.data.normalize_type == 'std':
##            minv = np.std(ksp_idx)*2
##            ksp_idx = ksp_idx /  minv
##            ksp_idx = np.squeeze(ksp_idx, 0)
##            kspace = np.asarray(ksp_idx)
#        #print('ksp_idx', ksp_idx.shape) #ksp_idx (1, 15, 320, 320)
#
#        nImg, *_ = ksp_idx.shape
##    
#        M = np.tile(self.mask,[nImg,30,1,1])
#        mask = self.mask.astype(np.complex64)
#        mask = np.tile(mask, [nImg, 15, 1, 1])   
#
#        #print('Undersampling')
#
#        #print('org', org.shape, org.dtype)
#        ksp, orgk, minv = generateUndersampled(ksp_idx, mask)
##        ksp = IFFT2c(ksp)    
##        orgk = IFFT2c(orgk) 
#        ksp = c2r(ksp)
#        orgk = c2r(orgk)
#        #print('orgk', orgk.shape, orgk.dtype)orgk (1, 30, 320, 320) float64
#
#        ksp = np.asarray(np.squeeze(ksp, 0))
#        orgk = np.asarray(np.squeeze(orgk, 0))
#        M = np.asarray(np.squeeze(M, 0))
#        return ksp, orgk, M
#
#    def __len__(self):
#        # Total number of slices from all scans
#        return int(np.sum(self.num_slices))

def getData3(maskpath):
    print('Reading data')
    data_dir = r"/data0/chenluo/data/csmdataset/T1_data_34/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 28): #28
        checklist.append(datalist[i])
        datadir = data_dir + datalist[i]        
        #datadir_csm = datadir.replace('T1_data_33', 'Output_maps_33')
        A = h5py.File( datadir )
        orgk = A['kspace']
        orgk = orgk[5:, :, :, :]
        #orgk = crop(orgk, cropx=332)
        orgk = IFFT2c(orgk)
        orgk = crop(orgk, cropy=320, cropx=300)
#        print('orgk', orgk.shape, orgk.dtype)
        A.close()        
        if i == 0:
            org = orgk
        else:
            org = np.concatenate([org, orgk], 0)
#        B = h5py.File( datadir_csm )
#        csmk = B['s_maps']
#        csmk = csmk[5:, :, :, :]
#        csmk = crop(csmk, cropx=image_size2, cropy=image_size1)
#        B.close()
#        if i == 0:
#            csm = csmk
#        else:
#            csm = np.concatenate([csm, csmk], 0)
        print(datalist[i])
    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype            
    org = FFT2c(org)
    print('Data prepared!')             
######  org is multi-channel kspace #####
    
    nImg, *_ = org.shape
    
    C = sio.loadmat(maskpath)    
    mask = C['mask'][:]
#    mask = np.transpose(mask, [1, 0])   

    M = np.tile(mask,[nImg,30,1,1])
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 15, 1, 1])   

    print('Undersampling')

    print('org', org.shape, org.dtype)
    ksp, orgk, minv = generateUndersampled(org, mask)
#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 
#
#    kernel = []
#    for n in range(nImg): #tqdm(range(nCoil)):
#        cailb =  spirit_crop(np.transpose(ksp[n,:,:,:],(1,2,0)), 24, 24, 15)  
#        kernelk = spirit_calibrate(cailb, (7,7), lamda=0.16, filtering=False, verbose=True) #(5,5,12,12)
#        #print('kernel', kernelk.shape, kernelk.dtype) #kernel (5, 5, 30, 30) complex128 (7, 7, 32, 32) complex128
#        kernelk = np.transpose(kernelk,(3,2,0,1)).astype(np.complex64)        
#        kernel.append(np.expand_dims(kernelk, 0)) #org[i] = temp2#*W #* minv
#
#    kernel = np.concatenate(kernel, axis=0)  
       
         
    ksp = c2r(ksp)
    orgk = c2r(orgk)
 
    toc()
    print('Data prepared!')
   

    return ksp, orgk, M#, kernel



def getData4(maskpath):
    print('Reading data')
    data_dir = r"/data0/chenluo/data/T1data/T1_train/kspace/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 23): #23
        checklist.append(datalist[i])
        datadir = data_dir + datalist[i]        
        #datadir_csm = datadir.replace('T1_data_33', 'Output_maps_33')
        A = h5py.File( datadir )
        orgk = A['kspace']
        orgk = orgk[:-105, :, :, :]
        #orgk = crop(orgk, cropx=332)
        orgk = IFFT2c(orgk)
        orgk = crop(orgk, cropy=172, cropx=232)
#        print('orgk', orgk.shape, orgk.dtype)
        A.close()        
        if i == 0:
            org = orgk
        else:
            org = np.concatenate([org, orgk], 0)
#        B = h5py.File( datadir_csm )
#        csmk = B['s_maps']
#        csmk = csmk[5:, :, :, :]
#        csmk = crop(csmk, cropx=image_size2, cropy=image_size1)
#        B.close()
#        if i == 0:
#            csm = csmk
#        else:
#            csm = np.concatenate([csm, csmk], 0)
        print(datalist[i])
    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype            
    org = FFT2c(org)
    print('Data prepared!')             
######  org is multi-channel kspace #####
    
    nImg, *_ = org.shape
    
    C = sio.loadmat(maskpath)    
    mask = C['mask'][:]
#    mask = np.transpose(mask, [1, 0])   

    M = np.tile(mask,[nImg,64,1,1])
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 32, 1, 1])   

    print('Undersampling')

    print('org', org.shape, org.dtype)
    ksp, orgk, minv = generateUndersampled(org, mask)
    
    kernel = []
    for n in range(nImg): #tqdm(range(nCoil)):
        cailb =  spirit_crop(np.transpose(ksp[n],(1,2,0)), 24, 24, 32)  
        kernelk = spirit_calibrate(cailb, (7,7), lamda=0.16, filtering=False, verbose=True) #(5,5,12,12)
        #print('kernel', kernelk.shape, kernelk.dtype) #kernel (5, 5, 30, 30) complex128 (7, 7, 32, 32) complex128
        kernelk = np.transpose(kernelk,(3,2,0,1)).astype(np.complex64)        
        kernel.append(np.expand_dims(kernelk, 0)) #org[i] = temp2#*W #* minv

    kernel = np.concatenate(kernel, axis=0)  
        
#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 
    ksp = c2r(ksp)
    orgk = c2r(orgk)
 
    toc()
    print('Data prepared!')
   

    return ksp, orgk, M, kernel    
    


class T1MRIKneeDataSet(Dataset):
    def __init__(self, maskpath, mode, ny, nx, nc):
        super(T1MRIKneeDataSet, self).__init__()
        
        if mode == 'training':
            self.kspace_dir = '/data0/zhuoxu/data/XJTU2/T1_train/kspace/'
            self.maps_dir = '/data0/zhuoxu/data/XJTU2/T1_train/maps/'
            self.file_list = get_all_files(self.kspace_dir)[:]
        elif mode == 'val':
            self.kspace_dir = '/data0/taofeng/kneetest/T1data/'
            self.maps_dir = '/data0/taofeng/kneetest/csm/'
            self.file_list = get_all_files(self.kspace_dir)
        elif mode == 'testing':
            self.kspace_dir = '/data0/zhuoxu/data/XJTU2/T1_test/kspace/'
            self.maps_dir = '//data0/zhuoxu/data/XJTU2/T1_test/maps/'
            self.file_list = get_all_files(self.kspace_dir)[:2]
        elif mode == 'datashift':
            self.kspace_dir = '/data0/chentao/data/fastMRI_brain/brain_T2/'
            self.maps_dir = '/data0/chentao/data/fastMRI_brain/output_maps/'
            self.file_list = get_all_files(self.kspace_dir)
        else:
            raise NotImplementedError

        self.mode = mode
        self.ny = ny
        self.nx = nx
        self.nc = nc
        
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(self.kspace_dir, os.path.basename(file)))
            with h5py.File(os.path.join(self.kspace_dir, os.path.basename(file)), 'r') as data:
                if self.mode != 'sample':
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 106)
                else:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'
        C = sio.loadmat(maskpath)    
        self.mask = C['mask'][:]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        raw_file = os.path.join(self.kspace_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            if self.mode != 'sample':
                slice_idx = slice_idx + 5  # Adjust slice index

            ksp_idx = data['kspace'][slice_idx]
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = crop(IFFT2c(ksp_idx), cropy=self.ny, cropx=self.nx)
            ksp_idx = FFT2c(ksp_idx)

        nImg, *_ = ksp_idx.shape
        M = np.tile(self.mask, [nImg, self.nc * 2, 1, 1])
        mask = self.mask.astype(np.complex64)
        mask = np.tile(mask, [nImg, self.nc, 1, 1])

        ksp, orgk, minv = generateUndersampled(ksp_idx, mask)    
        #print('ksp,kernel', ksp.shape)
        cailb =  spirit_crop(np.transpose(ksp.squeeze(),(1,2,0)), 14, 14, self.nc)  
        kernel = spirit_calibrate(cailb, (7,7), lamda=0.16, filtering=False, verbose=True) #(5,5,12,12)
        #print('kernel', kernel.shape, kernel.dtype) #kernel (5, 5, 30, 30) complex128 (7, 7, 32, 32) complex128
        kernel =np.transpose(kernel,(3,2,0,1)).astype(np.complex64)
        #kernel = r2c(c2r(kernel).astype(np.float32)) 
        #print('ksp,kernel', ksp.shape, kernel.shape) #ksp,kernel (1, 32, 172, 232) (32, 32, 7, 7)


        ksp = c2r(ksp)
        orgk = c2r(orgk)

        ksp = np.asarray(np.squeeze(ksp, 0))
        orgk = np.asarray(np.squeeze(orgk, 0))
        M = np.asarray(np.squeeze(M, 0)) 
        kernel = np.asarray(kernel)               
        
        return ksp, orgk, M, kernel

    def __len__(self):
        return int(np.sum(self.num_slices))


#def getData3(maskpath):
#    print('Reading data')
#    data_dir = r"/data0/chentao/data/fastMRI/multicoil_train/kspace/"
#    
#    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
#    checklist = []
#    for i in range(0, 200): #28
#        checklist.append(datalist[i])
#        datadir = data_dir + datalist[i]        
#        #datadir_csm = datadir.replace('kspace', 'maps')
#        A = h5py.File( datadir )
#        orgk = A['kspace']
#        #print('orgk', orgk.shape, orgk.dtype)
#        orgk = orgk[5:, :, :, :]
#        #orgk = crop(orgk, cropx=320)
#        orgk = IFFT2c(orgk)
#        orgk = crop(orgk, cropx=320, cropy=320)
#        #print('orgk', orgk.shape, orgk.dtype)
#        A.close()        
#        if i == 0:
#            org = orgk
#        else:
#            org = np.concatenate([org, orgk], 0)
##        B = h5py.File( datadir_csm )
##        csmk = B['s_maps']
##        csmk = csmk[5:, :, :, :]
##        csmk = crop(csmk, cropx=image_size2, cropy=image_size1)
##        B.close()
##        if i == 0:
##            csm = csmk
##        else:
##            csm = np.concatenate([csm, csmk], 0)
#        print(datalist[i])
#    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype   
#    #org = zoom(org, (1, 1, 128 / 320, 128 / 320), mode='nearest')         
#    org = FFT2c(org)
#    print('Data prepared!')             
#######  org is multi-channel kspace #####
#    
#    nImg, *_ = org.shape
#    
#    C = sio.loadmat(maskpath)    
#    mask = C['mask'][:]
##    mask = np.transpose(mask, [1, 0])   
#
#    M = np.tile(mask,[nImg,30,1,1])
#    mask = mask.astype(np.complex64)
#    mask = np.tile(mask, [nImg, 15, 1, 1])   
#
#    print('Undersampling')
#
#    print('org', org.shape, org.dtype)
#    ksp, orgk, minv = generateUndersampled(org, mask)
#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 
#    ksp = c2r(ksp)
#    orgk = c2r(orgk)
# 
#    toc()
#    print('Data prepared!')
#   
#
#    return ksp, orgk, M


    
def getTestingData3(maskpath):
    print('Reading data')
    data_dir = r"/data0/chenluo/data/csmdataset/T1_data_3/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 3): #28
        checklist.append(datalist[i])
        datadir = data_dir + datalist[i]        
        #datadir_csm = datadir.replace('T1_data_33', 'Output_maps_33')
        A = h5py.File( datadir )
        orgk = A['kspace']
        orgk = orgk[5:, :, :, :]
        #orgk = crop(orgk, cropx=332)
        orgk = IFFT2c(orgk)
        orgk = crop(orgk, cropy=320, cropx=300)
#        print('orgk', orgk.shape, orgk.dtype)
        A.close()        
        if i == 0:
            org = orgk
        else:
            org = np.concatenate([org, orgk], 0)
#        B = h5py.File( datadir_csm )
#        csmk = B['s_maps']
#        csmk = csmk[5:, :, :, :]
#        csmk = crop(csmk, cropx=image_size2, cropy=image_size1)
#        B.close()
#        if i == 0:
#            csm = csmk
#        else:
#            csm = np.concatenate([csm, csmk], 0)
        print(datalist[i])
    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype            
    org = FFT2c(org)
    print('Data prepared!')             
######  org is multi-channel kspace #####
    
    nImg, *_ = org.shape
    
    C = sio.loadmat(maskpath)    
    mask = C['mask'][:]
#    mask = np.transpose(mask, [1, 0])   

    M = np.tile(mask,[nImg,30,1,1])
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 15, 1, 1])   

    print('Undersampling')

    print('org', org.shape, org.dtype)
    ksp, orgk, minv = generateUndersampled(org, mask)
#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 

#    kernel = []
#    for n in range(nImg): #tqdm(range(nCoil)):
#        cailb =  spirit_crop(np.transpose(ksp[n,:,:,:],(1,2,0)), 24, 24, 15)  
#        kernelk = spirit_calibrate(cailb, (7,7), lamda=0.16, filtering=False, verbose=True) #(5,5,12,12)
#        #print('kernel', kernelk.shape, kernelk.dtype) #kernel (5, 5, 30, 30) complex128 (7, 7, 32, 32) complex128
#        kernelk = np.transpose(kernelk,(3,2,0,1)).astype(np.complex64)        
#        kernel.append(np.expand_dims(kernelk, 0)) #org[i] = temp2#*W #* minv
#
#    kernel = np.concatenate(kernel, axis=0)  
       
         
    ksp = c2r(ksp)
    orgk = c2r(orgk)
 
    toc()
    print('Data prepared!')
   

    return ksp, orgk, M#, kernel
  
    
    
def getTestingData4(maskpath):
    print('Reading data')
    data_dir = r"/data0/chenluo/data/T1data/T1_test/kspace/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 3): #23
        checklist.append(datalist[i])
        datadir = data_dir + datalist[i]        
        #datadir_csm = datadir.replace('T1_data_33', 'Output_maps_33')
        A = h5py.File( datadir )
        orgk = A['kspace']
        orgk = orgk[:-105, :, :, :]
        #orgk = crop(orgk, cropx=332)
        orgk = IFFT2c(orgk)
        orgk = crop(orgk, cropy=172, cropx=232)
#        print('orgk', orgk.shape, orgk.dtype)
        A.close()        
        if i == 0:
            org = orgk
        else:
            org = np.concatenate([org, orgk], 0)
#        B = h5py.File( datadir_csm )
#        csmk = B['s_maps']
#        csmk = csmk[5:, :, :, :]
#        csmk = crop(csmk, cropx=image_size2, cropy=image_size1)
#        B.close()
#        if i == 0:
#            csm = csmk
#        else:
#            csm = np.concatenate([csm, csmk], 0)
        print(datalist[i])
    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype            
    org = FFT2c(org)
    print('Data prepared!')             
######  org is multi-channel kspace #####
    
    nImg, *_ = org.shape
    
    C = sio.loadmat(maskpath)    
    mask = C['mask'][:]
#    mask = np.transpose(mask, [1, 0])   

    M = np.tile(mask,[nImg,64,1,1])
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 32, 1, 1])   

    print('Undersampling')

    print('org', org.shape, org.dtype)
    ksp, orgk, minv = generateUndersampled(org, mask)
    
#    kernel = []
#    for n in range(nImg): #tqdm(range(nCoil)):
#        cailb =  spirit_crop(np.transpose(ksp[n],(1,2,0)), 24, 24, 32)  
#        kernelk = spirit_calibrate(cailb, (7,7), lamda=0.16, filtering=False, verbose=True) #(5,5,12,12)
#        #print('kernel', kernelk.shape, kernelk.dtype) #kernel (5, 5, 30, 30) complex128 (7, 7, 32, 32) complex128
#        kernelk = np.transpose(kernelk,(3,2,0,1)).astype(np.complex64)        
#        kernel.append(np.expand_dims(kernelk, 0)) #org[i] = temp2#*W #* minv
#
#    kernel = np.concatenate(kernel, axis=0)  
        
#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 
    ksp = c2r(ksp)
    orgk = c2r(orgk)
 
    toc()
    print('Data prepared!')
   

    return ksp, orgk, mask#, kernel    
    



def getData1(maskpath):
    print('Reading data')
    tic()
#    A = loadmat('/data0/huayu/Aluochen/mask_256_6x/brain_modl_trn.mat')
#    Org = A['trnOrg'][:]
#    csm = A['trnCsm'][:]
#    Org = np.transpose(Org, [2, 1, 0])
#    Org = Org[:, np.newaxis, :, :]
#    csm = np.transpose(csm, [3, 2, 1, 0])
#    org = FFT2c(Org * csm)
#    print('org, csm', org.shape, org.dtype, csm.shape, csm.dtype)

    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/256x232brain_ksp.h5"
    A = h5py.File(data_dir)
    org = A['kspace'][:]
    A.close()
#    data_dir_csm = r"/data2/luochen/model/SSDU/csm-est/256x232brain_csm.h5"
#    B = h5py.File(data_dir_csm)
#    sens_maps = B['s_maps'][:]
#    B.close()
    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype
 
    nImg, *_ = org.shape
    
    C = h5py.File(maskpath)    
    mask = C['mask'][:]
    mask1 = C['mask1'][:]
    C.close()
    
    mask = np.transpose(mask, [1, 0])
    mask1 = np.transpose(mask1, [1, 0])    
    M = np.tile(mask,[nImg,24,1,1])
    M1 = np.tile(mask1,[nImg,24,1,1])    
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 12, 1, 1])
    mask1 = mask1.astype(np.complex64)
    mask1 = np.tile(mask1, [nImg, 12, 1, 1])
        
    toc()
    print('Undersampling')
    tic()
    print('org', org.shape, org.dtype)
    ksp, orgk, minv = generateUndersampled(org, mask)
    _, atb1, _ = generateUndersampled(org, mask1)

#    atb1 = c2r(atb1)
#    atb = c2r(atb)
#    ksp = c2r(ksp)
#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 
    ksp = c2r(ksp)
    orgk = c2r(orgk)
    toc()
    print('Data prepared!')

    return ksp,orgk,atb1,M,M1

#def getData1(mask):
#    print('Reading data')
#    tic()
#    org=np.load('/data/31836012/data/trn_data_90im_4_subjects.npy')
#    org = org[:nImg,:,:,:]
#
#    B = h5py.File('/data/31836012/data/poisson_256x232_acc6_split.mat')
#    
#    mask = B['mask'][:]
#    mask1 = B['mask1'][:]
#    mask2 = B['mask2'][:]
#
#    mask = np.transpose(mask, [1, 0])
#    mask1 = np.transpose(mask1, [1, 0])
#    mask2 = np.transpose(mask2, [1, 0])
#    B.close()
#    M = mask
#    M1 = mask1
#    M2 = mask2
#
#    mask = mask.astype(np.complex64)
#    mask = np.tile(mask, [nImg, 12, 1, 1])
#    mask1 = mask1.astype(np.complex64)
#    mask1 = np.tile(mask1, [nImg, 12, 1, 1])
#    mask2 = mask2.astype(np.complex64)
#    mask2 = np.tile(mask2, [nImg, 12, 1, 1])
#    
#    M = np.tile(M,[nImg,24,1,1])
#    M1 = np.tile(M1,[nImg,24,1,1])
#    M2 = np.tile(M2,[nImg,24,1,1])
#    toc()
#    print('Undersampling')
#    tic()
#    ksp,orgk, _ = generateUndersampled(org, mask)
#    _,atb1, _ = generateUndersampled(org, mask1)
#    _,atb2, _ = generateUndersampled(org, mask2)
#
#    atb1=c2r(atb1)
#    atb2=c2r(atb2)
#    orgk=c2r(orgk)
#    ksp = c2r(ksp)
#    toc()
#    print('Data prepared!')
#    return ksp,orgk,atb1,atb2,M,M1,M2

def getTestingData1(maskpath):
    tic()  
    
    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/256x232brain_ksp_test.h5"
    data_dir_csm = '/data0/huayu/Aluochen/MyPaper1.1/data/256x232brain_csmfull_test.mat'
    A = h5py.File(data_dir)
    B = loadmat(data_dir_csm)
    org = A['kspace'][:]
    A.close()
    csm = B['csm_3d'][:]
    csm = np.transpose(csm,[3,2,1,0])

#    A = loadmat('/data0/huayu/Aluochen/mask_256_6x/brain_modl_tst.mat')
#    Org = A['tstOrg'][:]
#    csm = A['tstCsm'][:]
#    Org = np.transpose(Org, [2, 1, 0])
#    Org = Org[:, np.newaxis, :, :]
#    csm = np.transpose(csm, [3, 2, 1, 0])
#    org = FFT2c(Org * csm)

    print('org, csm', org.shape, org.dtype, csm.shape, csm.dtype)
    nSlices, *_ = org.shape
    
    B = h5py.File(maskpath)
    mask = B['mask'][:]
    mask = np.transpose(mask, [1, 0])
    B.close()   
    
    M = np.tile(mask,[nSlices,24,1,1])
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nSlices, 12, 1, 1])   
    
    toc()
    print('Undersampling')
    tic()

    orgk, atb, minv = generateUndersampled(org, mask) 
#    atb = IFFT2c(atb)
#    orgk = IFFT2c(orgk)   
    atb = c2r(atb)
    orgk = c2r(orgk)

    toc()
    print('Data prepared!')
    return csm,orgk,atb,M #,minv

#def getTestingData1(maskpath):
#    tic()
#
#    A = loadmat('/data/31836012/data/brain_model_test.mat')
#    org = A['org'][:]
#    org = org[52:52+nImg, :, :, :]#org[52:52+nImg, :, :, :]
#
#    C = h5py.File('/data/31836012/data/poisson_256x232_acc6.mat')
#    mask = C['mask'][:]
#    mask = np.transpose(mask,[1,0])
#    C.close()
#    M = mask
#    mask = mask.astype(np.complex64)
#    mask = np.tile(mask, [nImg, 12, 1, 1])
#    M = np.tile(M, [nImg, 24, 1, 1])
#    toc()
#    print('Undersampling')
#    tic()
#    orgk, atb, minv = generateUndersampled(org, mask)
#    atb = c2r(atb)
#    orgk = c2r(orgk)
#    toc()
#    print('Data prepared!')
#    return org,orgk,atb,M,minv



######################################################## 
#def getData2(maskpath):
#    print('Reading data')
#    data_dir = r"/data0/huayu/Aluochen/csmdataset/T1_data_33/"
#    
#    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
#    checklist = []
#    for i in range(0, 28): #28
#        checklist.append(datalist[i])
#        datadir = data_dir + datalist[i]        
#        datadir_csm = datadir.replace('T1_data_33', 'Output_maps_33')
#        A = h5py.File( datadir )
#        orgk = A['kspace']
#        orgk = orgk[5:, :, :, :]
#        orgk = IFFT2c(orgk)
#        orgk = crop(orgk, cropx=368, cropy=368)
##        print('orgk', orgk.shape, orgk.dtype)
#        A.close()        
#        if i == 0:
#            org = orgk
#        else:
#            org = np.concatenate([org, orgk], 0)
##        B = h5py.File( datadir_csm )
##        csmk = B['s_maps']
##        csmk = csmk[5:, :, :, :]
##        csmk = crop(csmk, cropx=image_size2, cropy=image_size1)
##        B.close()
##        if i == 0:
##            csm = csmk
##        else:
##            csm = np.concatenate([csm, csmk], 0)
#        print(datalist[i])
#    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype            
#    org = FFT2c(org)
#    print('Data prepared!')             
#######  org is multi-channel kspace #####
#    
#    nImg, *_ = org.shape
#    
#    C = sio.loadmat(maskpath)    
#    mask = C['mask'][:]
##    mask = np.transpose(mask, [1, 0])   
#
#    M = np.tile(mask,[nImg,30,1,1])
#    mask = mask.astype(np.complex64)
#    mask = np.tile(mask, [nImg, 15, 1, 1])   
#
#    print('Undersampling')
#
#    print('org', org.shape, org.dtype)
#    ksp, orgk, minv = generateUndersampled(org, mask)
##    ksp = IFFT2c(ksp)    
##    orgk = IFFT2c(orgk) 
#    ksp = c2r(ksp)
#    orgk = c2r(orgk)
#
#    toc()
#    print('Data prepared!')
#
#    return ksp,orgk,M
#
#def getTestingData2(maskpath):
#    print('Reading data')
#    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/T1_data_3/"
#    
#    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
#    checklist = []
#    for i in range(0, 1): #3
#        checklist.append(datalist[i])
#        datadir = data_dir + datalist[i]        
#        datadir_csm = datadir.replace('T1_data_3', 'Output_maps_3')
#
#        A = h5py.File( datadir )
#        orgk = A['kspace']
#        orgk = orgk[5:, :, :, :]
#        orgk = IFFT2c(orgk)
#        orgk = crop(orgk, cropx=368, cropy=368)
#        A.close()
#        if i == 0:
#            org = orgk
#        else:
#            org = np.concatenate([org, orgk], 0)
#
#        B = h5py.File( datadir_csm )
#        csmk = B['s_maps']
#        csmk = csmk[5:, :, :, :]
#        csmk = crop(csmk, cropx=368, cropy=368)
#        B.close()
#        if i == 0:
#            csm = csmk
#        else:
#            csm = np.concatenate([csm, csmk], 0)
#        print(datalist[i])
#    print('Data prepared!')           
#    org = FFT2c(org)
#    print('org', org.shape )
#    nImg, *_ = org.shape
#
#    C = sio.loadmat(maskpath)    
#    mask = C['mask'][:]
#  
#    M = np.tile(mask,[nImg,30,1,1])
#    mask = mask.astype(np.complex64)
#    mask = np.tile(mask, [nImg, 15, 1, 1])   
#
#    print('Undersampling')
#
#    orgk, atb, minv = generateUndersampled(org, mask) 
##    atb = IFFT2c(atb)
##    orgk = IFFT2c(orgk)   
#    atb = c2r(atb)
#    orgk = c2r(orgk)
#
#    toc()
#    print('Data prepared!')
#    return csm,orgk,atb,M #,minv
  
def getData2(maskpath):
    print('Reading data')
    tic()
    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_ksp/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 28): #28 ,31
        checklist.append(datalist[i])
#        dir_data = data_dir + datalist[i]
        A = h5py.File(data_dir + datalist[i])
        orgk = A['kspace'][:]
        A.close()
        if i==0:
            org = orgk
        else:
            org = np.concatenate([org, orgk], 0)  
        print(datalist[i])
    print('Data prepared!')
######  org是多通道的kspace #####
    print('org', org.shape, org.dtype) #, sens_maps.shape, sens_maps.dtype
    nImg, *_ = org.shape
    C = h5py.File(maskpath) 
    mask = C['mask'][:]
    mask1 = C['mask1'][:]
    mask = np.transpose(mask, [1, 0])
    mask1 = np.transpose(mask1, [1, 0])
    C.close()
    
    M = mask
    M1 = mask1
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 15, 1, 1])
    mask1 = mask1.astype(np.complex64)
    mask1 = np.tile(mask1, [nImg, 15, 1, 1])
        
    M = np.tile(M,[nImg,30,1,1])
    M1 = np.tile(M1,[nImg,30,1,1])

    toc()
    print('Undersampling')
    tic()
    print('org', org.shape, org.dtype)
    ksp, orgk, minv = generateUndersampled(org, mask)
    _, atb1, _ = generateUndersampled(org, mask1)


#    ksp = IFFT2c(ksp)    
#    orgk = IFFT2c(orgk) 
    ksp = c2r(ksp)
    orgk = c2r(orgk)

    toc()
    print('Data prepared!')

    return ksp,orgk,atb1,M,M1


def getTestingData2(maskpath):
    tic()
    print('Reading data')
    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/T1_data_3/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 1): #3
        checklist.append(datalist[i])
        datadir = data_dir + datalist[i]        
        datadir_csm = datadir.replace('T1_data_3', 'Output_maps_3')

        A = h5py.File( datadir )
        orgk = A['kspace']
        orgk = orgk[5:, :, :, :]
        orgk = IFFT2c(orgk)
        orgk = crop(orgk, cropx=368, cropy=368)
        A.close()
        if i == 0:
            org = orgk
        else:
            org = np.concatenate([org, orgk], 0)

        B = h5py.File( datadir_csm )
        csmk = B['s_maps']
        csmk = csmk[5:, :, :, :]
        csmk = crop(csmk, cropx=368, cropy=368)
        B.close()
        if i == 0:
            csm = csmk
        else:
            csm = np.concatenate([csm, csmk], 0)
        print(datalist[i])
    print('Data prepared!')             
    org = FFT2c(org)
    print('org', org.shape )
    nImg, *_ = org.shape

    C = h5py.File(maskpath)    
    mask = C['mask'][:]
    mask = np.transpose(mask, [1, 0])
    C.close()   

    M = np.tile(mask,[nImg,30,1,1])
    mask = mask.astype(np.complex64)
    mask = np.tile(mask, [nImg, 15, 1, 1])   
    
    toc()
    print('Undersampling')
    tic()

    orgk, atb, minv = generateUndersampled(org, mask) 
#    atb = IFFT2c(atb)
#    orgk = IFFT2c(orgk)   
    atb = c2r(atb)
    orgk = c2r(orgk)


    toc()
    print('Data prepared!')
    return csm,orgk,atb,M #,minv

    

########################################################   



def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)



def crop(img, cropy, cropx):
    nb, c, y, x = img.shape    
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty:starty + cropy, startx:startx + cropx]


#%%

def usp(x,mask,nch,nrow,ncol):
    """ This is a the A operator as defined in the paper"""
    kspace=np.reshape(x,(nch,nrow,ncol))
    res=kspace[mask!=0]
    return kspace,res

def usph(kspaceUnder,mask,nch,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nch,nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    minv=np.std(temp)    
    temp=temp/minv
    return temp,minv

def usph2(kspaceUnder,mask,nch,nrow,ncol):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((nch,nrow,ncol),dtype=np.complex64)
    temp[mask!=0]=kspaceUnder
    return temp

def generateUndersampled(org,mask):
    nSlice,nch,nrow,ncol=org.shape
    orgk=np.empty(org.shape,dtype=np.complex64)
    atb=np.empty(org.shape,dtype=np.complex64)
    minv=np.zeros((nSlice,),dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
        At = lambda z: usph(z,mask[i],nch,nrow,ncol)
        orgk[i],y=A(org[i])
        atb[i],minv[i]=At(y)
        orgk[i]=orgk[i]/minv[i]
    del org
    return orgk,atb,minv


def Ax(org,mask):
    nSlice,nch,nrow,ncol=org.shape
    atb=np.empty(org.shape,dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: usp(z,mask[i],nch,nrow,ncol)
        At = lambda z: usph2(z,mask[i],nch,nrow,ncol)
        _,y=A(org[i])
        atb[i]=At(y)
    del org
    return atb


#
#
def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    nImg,nCh,nrow,ncol=inp.shape
    out=np.zeros((nImg,nCh*2,nrow,ncol),dtype=dtype)
    out[:,0:nCh,:,:]=np.real(inp)
    out[:,nCh:nCh*2,:,:]=np.imag(inp)
    return out