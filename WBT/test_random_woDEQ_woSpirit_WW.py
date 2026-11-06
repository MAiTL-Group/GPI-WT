import torch
import torch.nn as nn
import h5py
import scipy.io as sio
import numpy as np
from torch.nn import init
import torch.utils.data as Data
import time
import os
from os.path import join
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.autograd import Variable
from data_proc import getTestingData1, getTestingData2, getTestingData3, get_dataset
from anderson import anderson
from unrolledK_woSpirit_WW import WBT  
from utils import r2c, c2r, SOS, ifft2c, IFFT2c, FFT2c, normalize, myPSNR, weight_mask
import torch.autograd as autograd
#from einops.layers.torch import Rearrange
from scipy.io import loadmat
parser = ArgumentParser(description='self_knee_acc4')


parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epoch1 = 95

#368_4: 120-31.65, 125-31.87
seed = 1 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
batch_size = 1
conf = {
    "epochs": 200,
    "pre_train_epochs": 0,
    "forward_eps": 1e-4,
    "max_iters": 1,
    "backward_eps": 1e-4,
    "batch_size": 4,
    "alpha": 0.5,
    "learning_rate": 0,  # 1e-4,
    "random_seed": seed,
}

dataset = '320x300mask4acs24'

if dataset =='320x300mask4acs24' :
    subDirectory = 'WW_woDEQSpirit_WBTK_Random_320x300mask4acs24'  #
    gettestdata = getTestingData3
    result_path = dataset
    maskpath = '/data0/chenluo/Mypaper4.1/random_320x300_acs24_R4.mat'



model_dir = subDirectory +'/trainn_%d_lg_%.4f'% (conf["learning_rate"], conf["epochs"])

#ny, nx, nc = 172, 232, 32 
#test_loader = get_dataset(maskpath, 'testing', ny, nx, nc)


org, atb, mask = gettestdata(maskpath) #, minv
##csm, org, atb, mask_r = gettestdata3(128, 128, maskpath)
#
org = torch.from_numpy(org)
atb = torch.from_numpy(atb)
mask = torch.from_numpy(mask)
#kernel = torch.from_numpy(kernel)
#minv = torch.from_numpy(minv)

nb, nc, nx, ny = atb.shape
print('nb, nc, ny, nx',nb, nc, ny, nx)
#_, nch, *_ = csm.shape
nch = 15
#std = 0
#noise = torch.randn_like(atb)
#noise = noise * mask
#atb = atb + std * noise

class Mydata(Data.Dataset):
    def __init__(self):
        super(Mydata,self).__init__()
        self.org = org
        self.atb = atb
        self.mask = mask
#        self.kernel = kernel
        #self.minv = minv
    def __getitem__(self,index):
        orgk = self.org[index]
        atbk = self.atb[index]
        mask = self.mask[index]
#        kernelk = self.kernel[index]
        #minv = self.minv[index]
        return orgk, atbk, mask#, kernelk#, minv
    def __len__(self):
        return len(self.org)
        
testset = Mydata()
test_loader = Data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=False)
#
##
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, m, k, w): #atb, mask, atb, kernel, W
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, m, x, k, w), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, m, x, k, w)
        return z
        
##W = weight_mask((320, 300), 3, 0.4).to(device) #  , 3e1, .4 #0.99, 1e-4
#
W = np.load('30.4W.npy')
W = torch.from_numpy(W)
W = W.to(device)
W.requires_grad_(False) 
print(W.shape, W.dtype) 

f = WBT(LayerNo=10, image_size=(ny,nx), channels = nc)

model = f.to(device)
#model = DEQFixedPoint(f, anderson, tol=1e-6, max_iter=10, beta = 1.0).to(device) # f #


model.load_state_dict(torch.load('./%s/epoch_%d.pkl' % (model_dir, num_epoch1))['net_state_dict'])

out = []#torch.complex(torch.zeros([nb, nch, nx, ny]), torch.zeros([nb, nch, nx, ny])).to(device) 
org = []#torch.complex(torch.zeros([nb, nch, nx, ny]), torch.zeros([nb, nch, nx, ny])).to(device) 
ksp = []

with torch.no_grad():
    for i, (orgk,atb,mask) in enumerate(test_loader):
        orgk,atb,mask = orgk.type(torch.FloatTensor).to(device), atb.type(torch.FloatTensor).to(device), mask.type(torch.FloatTensor).to(device)#, kernel.to(device)

#        atb = W*atb
        x = model(atb, mask, atb, W) 
#        x = x/W
        
        x = ifft2c(r2c(x))
        img = ifft2c(r2c(orgk))
        k = r2c(orgk)

        out.append(x) #out[i] = temp1 #* minv
        org.append(img) #org[i] = temp2#*W #* minv
        ksp.append(k)

out = torch.cat(out, dim=0)
org = torch.cat(org, dim=0)
ksp = torch.cat(ksp, dim=0)
print('...', org.shape, org.dtype)
out = out.cpu().data.numpy()
org = org.cpu().data.numpy()
ksp = ksp.cpu().data.numpy()

#result_path = "/data/31836012/selfSLRknee/result"
#sio.savemat(join(result_path, 'self_testDS0H.mat'), {'recon':out, 'org':org})

#out = IFFT2c(out)    
#org = IFFT2c(org) 

print('...', org.shape,org.dtype)
#orgk = IFFT2c(org)
#plt.imshow(np.abs(org[15, 6, :, :]))
#plt.show()

#out = np.sum(out * np.conj(csm), axis=1) 
#org = np.sum(org * np.conj(csm), axis=1)
out = SOS(out)
org = SOS(org)
#ksp = SOS(ksp)

#result_path = "/data0/huayu/Aluochen/selfSLRknee/result"

#sio.savemat(join(result_path+'_slrKsos.mat'), {'recon':out, 'org':org, 'orgk':ksp})
sio.savemat(join(result_path+'_WBTK_Random_woDEQSpirit_WW.mat'), {'recon':out, 'org':org})
 

from scipy.io import loadmat            
import h5py
import numpy as np
import matplotlib.pyplot as plt
 


nImg = 15

#C = h5py.File(maskpath)
C =loadmat(maskpath)
mask = C['mask'][:]
#mask = np.transpose(mask,[1,0])   

A = loadmat(result_path+'_slrKsos.mat')
org = A['org'][nImg]
out = A['recon'][nImg]

print('org', org.shape, org.dtype) 
print('out', out.shape, out.dtype) 
print('mask', mask.shape, mask.dtype) 

org = np.abs(org)
out = np.abs(out)

##sio.savemat(join("/data0/huayu/Aluochen/selfSLRknee/", 'selfSLR_kneeDS0H1.mat'), {'recon':out, 'org':org})

normorg = normalize(org)
normout = normalize(out) 
print('normorg', normorg.shape, normorg.dtype) 
print('normout', normout.shape, normout.dtype)

psnrx = myPSNR(normorg, normout)

plot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, .8))
plt.clf()
plt.subplot(131)
plot(mask) #np.fft.ifftshift(mask)
plt.axis('off')
plt.title('(a) Mask', y=-0.2)

plt.subplot(132)
plot(normorg)
plt.axis('off')
plt.title('(b) Original', y=-0.2)

plt.subplot(133)
plot(normout)
plt.title('(c) Our:'+ str(psnrx.round(2)), y=-0.2)
plt.axis('off')

plt.savefig('./testeDS0K.png')
#plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
plt.show()


