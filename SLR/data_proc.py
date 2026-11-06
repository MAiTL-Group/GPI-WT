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

def getFilter1(nImg):
    print('Reading filter')
    F = h5py.File('/data/31836012/selfSLR/weight.mat')
    filt = F['weight'][:]
    filt = np.transpose(filt)
    print('filt', filt)

    filt = np.tile(filt, [nImg, 24, 1, 1])
    
    print('Data prepared!')
    return filt

def getFilter2(nImg):
    print('Reading filter')
    F = h5py.File('/data0/huayu/Aluochen/selfSLRknee/Weight1.mat')
    filt = F['weight1'][:]
    filt = np.transpose(filt)
    print('filt', filt)

    filt = np.tile(filt, [nImg, 30, 1, 1])
    
    print('Data prepared!')
    return filt

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
    ksp, atb, minv = generateUndersampled(org, mask)
    _, atb1, _ = generateUndersampled(org, mask1)

    atb1 = c2r(atb1)
    atb = c2r(atb)
    ksp = c2r(ksp)
#    org = c2r(org)
    toc()
    print('Data prepared!')

    return ksp,atb,atb1,M,M1

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
    
    atb = c2r(atb)
    #orgk = c2r(orgk)

    toc()
    print('Data prepared!')
    return csm,orgk,atb,M

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
def getData2(maskpath):
    print('Reading data')
    tic()
    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_ksp/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 28):
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

    atb1 = c2r(atb1)
    orgk = c2r(orgk)
    ksp = c2r(ksp)
    org = c2r(org)
    toc()
    print('Data prepared!')

    return ksp,orgk,atb1,M,M1



def getTestingData2(maskpath):
    tic()
    print('Reading data')
    data_dir = r"/data0/huayu/Aluochen/MyPaper1.1/data/368x368knee_test/T1_data_3/"
    
    datalist = natsorted(os.listdir(data_dir), alg=ns.PATH)
    checklist = []
    for i in range(0, 3):
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
    atb = c2r(atb)
    #orgk = c2r(orgk)

    toc()
    print('Data prepared!')
    return csm,orgk,atb,M #,minv

    

#def getTestingData2(maskpath):
#    tic()    
#    org, csm = gettestData_fastMRI()
#    nImg, *_ = org.shape
#    B = h5py.File(maskpath)    
#    mask = B['mask'][:]
#    mask = np.transpose(mask, [1, 0])
#    B.close()   
#    
#    M = np.tile(mask,[nImg,30,1,1])
#    mask = mask.astype(np.complex64)
#    mask = np.tile(mask, [nImg, 15, 1, 1])   
#    
#    toc()
#    print('Undersampling')
#    tic()
#
#    orgk, atb, minv = generateUndersampled(org, mask)    
#    atb = c2r(atb)
#    #orgk = c2r(orgk)
#
#    toc()
#    print('Data prepared!')
#    return csm,orgk,atb,M #,minv
    
##############################################
#def getData_fastMRI(nt=3):
#    print('Reading data')
#    tic()
#    data_dir = r"/data0/huayu/Aluochen/csmdataset/T1_data_33/"
#    #dir_data = data_dir + "/file1000001T1.h5"
#    A = h5py.File(r"/data0/huayu/Aluochen/csmdataset/T1_data_33/file1000001T1.h5")    
#    org = A['kspace']
#    org = org[5:, :, :, :] 
#    org = IFFT2c(org)
#    org = crop(org, cropx=368, cropy=368)
#    A.close()
#   # print('file1000001T1.h5')
#    
#    datalist = os.listdir(data_dir)
#    checklist = []
#    for i in range(1, nt - 1):
#        checklist.append(datalist[i])
#        dir_data = data_dir + datalist[i]
#
#        if dir_data == r"/data0/huayu/Aluochen/csmdataset/T1_data_33/file1000001T1.h5":
#            print('pass')
#            pass
#        else:
#            A = h5py.File(dir_data)
#            orgk = A['kspace']
#            orgk = orgk[5:, :, :, :]
#            orgk = IFFT2c(orgk)
#            orgk = crop(orgk, cropx=368, cropy=368)
#            A.close()
#            org = np.concatenate([org, orgk], 0)  
#        print(datalist[i])
#    print('Data prepared!')
#######  org是多通道的图像域 #####    
#    org = FFT2c(org)
#    toc()
#    return org
########  kspace_center ######

#def getData_fastMRI(nt=3):
#    print('Reading data')
#    tic()
#    data_dir = r"/data0/huayu/Aluochen/csmdataset/T1_data_33/"   
#    A = h5py.File(r"/data0/huayu/Aluochen/csmdataset/T1_data_33/file1000001T1.h5")    
#    org = A['kspace']
#    org = org[5:, :, :, :] 
#    org = IFFT2c(org)
#    org = crop(org, cropx=368, cropy=368)
#    A.close()
# 
#    datalist = os.listdir(data_dir)
#    checklist = []
#    for i in range(0, nt):
#        checklist.append(datalist[i])
#        dir_data = data_dir + datalist[i]
#
#        if dir_data == r"/data0/huayu/Aluochen/csmdataset/T1_data_33/file1000001T1.h5":
#            print('pass1')
#            pass
#        elif dir_data == r"/data0/huayu/Aluochen/csmdataset/T1_data_33/file1000002T1.h5":
#            print('pass2')
#            pass
#        elif dir_data == r"/data0/huayu/Aluochen/csmdataset/T1_data_33/file1000005T1.h5":
#            print('pass5')
#            pass            
#        else:
#            A = h5py.File(dir_data)
#            orgk = A['kspace']
#            orgk = orgk[5:, :, :, :]
#            orgk = IFFT2c(orgk)
#            orgk = crop(orgk, cropx=368, cropy=368)
#            A.close()
#            org = np.concatenate([org, orgk], 0)  
#        print(datalist[i])
#    print('Data prepared!') 
#    org = FFT2c(org)  #kspace_center
#    toc()
#    return org
########  kspace_center ######
#def gettestData_fastMRI():
#    print('Reading data')
#    datalist = os.listdir(r"/data2/luochen/model/testdata/T1_data_2/")
#    checklist = []
#    for i in range(0, 2):
#        checklist.append(datalist[i])
#        data_dir = r"/data2/luochen/model/testdata/T1_data_2/" + datalist[i]        
#        data_dir_csm = data_dir.replace('T1_data_2', 'csm_2')
#
#        A = h5py.File(data_dir)
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
#        B = h5py.File(data_dir_csm)
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
#
#    return org, csm   
    
#def gettestData_fastMRI():
#    print('Reading data')
#    data_dir = r"/data0/huayu/Aluochen/csmdataset/T1_data_1/file1000047T1.h5"
#    data_dir_csm = r"/data0/huayu/Aluochen/csmdataset/Output_maps_1/file1000047T1.h5"    
#    A = h5py.File(data_dir)   
#    B = h5py.File(data_dir_csm)
#    
#    org = A['kspace']
#    org = org[5:, :, :, :]
#    org = IFFT2c(org)
#    org = crop(org, cropx=368, cropy=368)
#    
#    print('Data prepared!') 
#    csm = B['s_maps']
#    csm = csm[5:, :, :, :]
#    csm = crop(csm, cropx=368, cropy=368)  
#    A.close() 
#    B.close()        
#    print('Data prepared!')       
#    org = FFT2c(org)
#    return org, csm

#def gettestData_fastMRI(data_dir):
#    print('Reading data')
#    tic()
#    dir_data = data_dir + "/file1000047T1.h5"
#    A = h5py.File(dir_data)
#    org = A['kspace']
#    org = org[5:, :, :, :]
#    org = IFFT2c(org)
#    org = crop(org, cropx=368, cropy=368)
#    A.close()
#    
##    for ii in range(org.shape[0]):
##        org[ii] = org[ii]/ np.max(np.abs(org[ii]))           
#    org = FFT2c(org)
#    return org


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



def crop(img, cropx, cropy):
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