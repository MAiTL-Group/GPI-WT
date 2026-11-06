import os
import torch
import torch.nn as nn
#import h5py
import scipy.io as sio
import numpy as np
from torch.nn import init
import torch.utils.data as Data
import time
from anderson import anderson
from os.path import join
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.autograd import Variable
from data_proc import getData1, getData2, getData3, getData4, get_dataset
from unrolledK_WW import WBT
from utils import r2c, c2r, weight_mask, ifft2c, sos
import torch.autograd as autograd
from torch.cuda import empty_cache
from torch.cuda.amp import autocast, GradScaler
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


parser = ArgumentParser(description='self_knee_acc4')

os.environ["CUDA_VISIBLE_DEVICES"] = "5" #,0,1,2,4,5,6,7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



dataset = '320x300mask4acs24'


if dataset =='320x300mask4acs24' :
    subDirectory = '320x300mask4acs24' 
    getdata = getData3
    result_path = dataset
    maskpath = '/data0/tingzhao/chenluo/MyPaper4.1/random_320x300_acs24_R4.mat' 
     
elif dataset =='320x300mask6acs24' :
    subDirectory = '320x300mask6acs24' 
    getdata = getData3
    result_path = dataset
    maskpath = '/data0/chenluo/Mypaper4.1/random_320x300_acs24_R6.mat'      
   
seed = 1
lr = 1e-4
epochs = 200
batch_size = 1
gamma = 0.99
restoreWeights = False #  True #

model_dir = 'WW_woDEQ_WBTK_Random_'+subDirectory +"/trainn_%d_lg_%.4f" % (lr, epochs)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#ny, nx, nc = 320, 300, 15 

##ksp, atb, M = getdata(maskpath)
#ksp, atb, M = getdata3(128, 128, maskpath)
ksp, atb, mask, kernel = getData3(maskpath)

ksp = torch.from_numpy(ksp)
atb = torch.from_numpy(atb)
mask = torch.from_numpy(mask)
kernel = torch.from_numpy(kernel)

nb, nc, ny, nx = atb.shape
print('nb, nc, ny, nx',nb, nc, ny, nx)


class Mydata(Data.Dataset):
    def __init__(self):
        super(Mydata,self).__init__()
        self.ksp = ksp
        self.atb = atb
        self.mask = mask
        self.kernel = kernel

    def __getitem__(self,index):
        kspk = self.ksp[index]
        atbk = self.atb[index]
        mask = self.mask[index]  
        kernelk = self.kernel[index]
        #print('atb', atb.shape) #atb torch.Size([63, 30, 320, 300])
#        cailb = crop(r2c(atbk.unsqueeze(0)).permute(0, 2, 3, 1), 24, 24, nc)
#        kernel = spirit_calibrate(cailb, (7, 7), lamda=0.16, filtering=False, verbose=True)
#        kernel = r2c(c2r(kernel.permute(0, 4, 3, 1, 2)).type(torch.FloatTensor))                 
        return kspk, atbk, mask, kernelk#, kernel.squeeze(0)
        
    def __len__(self):
        return len(self.atb)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


trainset = Mydata()
#train_loader = Data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=1, pin_memory=True)
train_loader = DataLoaderX(trainset,batch_size=batch_size,shuffle=True,num_workers=0, pin_memory=True)

#train_loader = get_dataset(maskpath, 'training28P', ny, nx, nc)

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

paramsG = model.parameters()
optimG = torch.optim.Adam(paramsG, lr=lr, amsgrad=False)
##scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimG, conf["epochs"]*360, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma)
scaler = GradScaler()


if not os.path.exists(model_dir):
    os.makedirs(model_dir)
start_epoch = -1
if restoreWeights:
    restorePATH = os.path.join(model_dir, "epoch_{}.pkl".format(150))
    start_epoch = torch.load(restorePATH)['epoch']
    model.load_state_dict(torch.load(restorePATH)['net_state_dict'])
    optimG.load_state_dict(torch.load(restorePATH)['optimizer_state_dict'])
    scheduler.load_state_dict(torch.load(restorePATH)['scheduler_state_dict'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


print('begin')    
for epoch in range(start_epoch+1, epochs):

    loss_R = []
    
    for i, (ksp,atb,mask,kernel) in enumerate(train_loader):
        ksp,atb,mask,kernel = ksp.type(torch.FloatTensor).to(device), atb.type(torch.FloatTensor).to(device), mask.type(torch.FloatTensor).to(device), kernel.to(device)
        #print('ksp', ksp.shape, atb.shape)
        
        optimG.zero_grad()

        x = model(atb, mask, atb, kernel, W) #model(atb, mask, atb, kernel, W)

        out = ifft2c(r2c(x))
        img = ifft2c(r2c(ksp))
    
        out = sos(out)
        img = sos(img).to(out.device)
    
        loss = torch.mean(torch.sum(torch.pow(torch.abs(out-img), 2), 0)) #torch.mean(torch.sum(torch.pow(torch.abs(x-ksp), 2), 0))#+torch.mean(torch.sum(torch.pow(torch.abs(res), 2), 0))
        #loss = torch.mean(torch.sum(torch.abs(out-img), 0))
    
        loss.backward()
        optimG.step()
        lossr = loss #torch.mean(torch.sum(torch.pow(torch.abs(c2r(ksp)-c2r(x)), 2), 0)) #
        #loss_R += [lossr.item()]
        loss_R += [lossr.item()]

    scheduler.step()
    print('LR: ', optimG.state_dict()['param_groups'][0]['lr'])  
       
    output_data = "[%02d/%02d] cost: %.6f costr: %.6f\n" % (
            epoch, i, loss, np.mean(loss_R))
    print(output_data)

    if (epoch+1) % 1 == 0:
#        torch.save(
#        model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch+1))  # save only the parameters

        PATH=os.path.join(model_dir, "epoch_{}.pkl".format(epoch+1))
        state = {'epoch': epoch,
             'net_state_dict':  model.state_dict(),
             'optimizer_state_dict': optimG.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()
              }

        torch.save(state, PATH)
        
        
         
         

