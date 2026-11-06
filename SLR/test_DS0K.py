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
from data_proc import getTestingData1, getTestingData2
from anderson import anderson
from unrolled import SLRK   
from utils import r2c, c2r, sos, ifft2c, IFFT2c, FFT2c, normalize, myPSNR
import torch.autograd as autograd
parser = ArgumentParser(description='self_knee_acc4')


parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epoch1 = 165
#165-35.12, 155-35.10, 135-35.19, 120-35.3, 110-35.24, 105-35.2
#6X:
#10X:160-32.45, 120-32.54, 125-32.57, 130-32.58
#368_6: 115-33.36
#368_4: 
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

dataset = '368x368mask4acs24'

if dataset =='368x368mask4acs24' :
    subDirectory = 'DS0K368x368mask4acs24' 
    gettestdata = getTestingData2
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_368x368_acs24_R4_split_2.mat'

 

model_dir = subDirectory +'/trainn_%d_lg_%.4f'% (conf["learning_rate"], conf["epochs"])
#model_dir = "/data0/huayu/Aluochen/mask368_4x_8acs/self_DS0H/SLRDS0H2/%s_trainn_%d_lg_%.4f" % (name, conf["learning_rate"], conf["epochs"])

csm, org, atb, mask_r = gettestdata(maskpath) #, minv

org = torch.from_numpy(org)
atb = torch.from_numpy(atb)
mask = torch.from_numpy(mask_r)
#minv = torch.from_numpy(minv)

nb, nch, nx, ny = atb.shape
_, nc, *_ = csm.shape
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
        #self.minv = minv
    def __getitem__(self,index):
        orgk = self.org[index]
        atbk = self.atb[index]
        mask = self.mask[index]
        #minv = self.minv[index]
        return orgk, atbk, mask#, minv
    def __len__(self):
        return len(self.org)
        
trainset = Mydata()
test_loader = Data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=0, pin_memory=False)

#
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, m):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, m, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, m, x)
        return z

##f = Landweber2(LayerNo=10, conv_size=64, nch=12, filter_size = 3, pad =1)
##f = Landweber(LayerNo=10, conv_size=128, nch=12, filter_size = 9, pad = 4)
f = SLRK(LayerNo=10, conv_size=64, nch=nch, filter_size=3, pad=1)
model = DEQFixedPoint(f, anderson, tol=1e-6, max_iter = 10, beta = 1.0)
#model = f
model = model.to(device)

model.load_state_dict(torch.load('./%s/epoch_%d.pkl' % (model_dir, num_epoch1))['net_state_dict'])

out = torch.complex(torch.zeros([nb, nc, nx, ny]), torch.zeros([nb, nc, nx, ny])).to(device) 
org = torch.complex(torch.zeros([nb, nc, nx, ny]), torch.zeros([nb, nc, nx, ny])).to(device) 
print('out', out.shape, out.dtype)
with torch.no_grad():
    for i, (orgk, atb, mask) in enumerate(test_loader):
        atb, mask = atb.type(torch.FloatTensor).to(device), mask.type(torch.FloatTensor).to(device)   
        #orgk, minv = orgk.to(device), minv.to(device)    
        out_temp = model(atb, mask) #, atb
        
        temp1 = r2c(out_temp)
        #temp2 = r2c(orgk)
       # print(temp1.shape, minv.shape)
        out[i] = temp1 #* minv
        org[i] = orgk #* minv

print('...', org.shape, org.dtype)
out = out.cpu().data.numpy()
org = org.cpu().data.numpy()

#result_path = "/data/31836012/selfSLRknee/result"
#sio.savemat(join(result_path, 'self_testDS0H.mat'), {'recon':out, 'org':org})

out = IFFT2c(out)    
org = IFFT2c(org) 

print('...', org.shape,org.dtype)
#orgk = IFFT2c(org)
#plt.imshow(np.abs(org[15, 6, :, :]))
#plt.show()

out = np.sum(out * np.conj(csm), axis=1) 
org = np.sum(org * np.conj(csm), axis=1)
#out = sos(out)
#org = sos(org)

#result_path = "/data0/huayu/Aluochen/selfSLRknee/result"
#sio.savemat(join(result_path, 'self_DS0H1.mat'), {'recon':out, 'org':org})
sio.savemat(join(result_path+'_DS0K.mat'), {'recon':out, 'org':org})
 

from scipy.io import loadmat            
import h5py
import numpy as np
import matplotlib.pyplot as plt
 


nImg = 44

C = h5py.File(maskpath)
#C =loadmat('/data2/luochen/model/random_368x368_acs8_R6_split2.mat')
mask = C['mask'][:]
mask = np.transpose(mask,[1,0])   

A = loadmat(result_path+'_DS0K.mat')
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

plt.savefig('./testDS0K.png')
#plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
plt.show()



