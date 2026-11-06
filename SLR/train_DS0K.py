import os
import torch
import torch.nn as nn
import h5py
import scipy.io as sio
import numpy as np
from torch.nn import init
import torch.utils.data as Data
import time

from os.path import join
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.autograd import Variable
from data_proc import getData1, getData2
from anderson import anderson
#from DEQ import DEQ
from unrolled import SLRK
from utils import r2c, c2r
import torch.autograd as autograd
parser = ArgumentParser(description='self_knee_acc4')


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset = '368x368mask4acs24'

if dataset =='256x232mask6acs64' :
    subDirectory = '256x232mask6acs64' 
    getdata = getData1
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_256x232_acs64_R6_split_2.mat' 

elif dataset =='256x232mask10acs64' :
    subDirectory = '256x232mask10acs64' 
    getdata = getData1
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_256x232_acs64_R10_split_2.mat' 
    
elif dataset =='256x232mask6acs48' :
    subDirectory = '256x232mask6acs48' 
    getdata = getData1
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_256x232_acs48_R6_split_2.mat' 

elif dataset =='256x232mask10acs48' :
    subDirectory = '256x232mask10acs48' 
    getdata = getData1
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_256x232_acs48_R10_split_2.mat'

elif dataset =='368x368mask4acs24' :
    subDirectory = '368x368mask4acs24' 
    getdata = getData2
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_368x368_acs24_R4_split_2.mat'

elif dataset =='368x368mask6acs24' :
    subDirectory = '368x368mask6acs24' 
    getdata = getData2
    result_path = dataset
    maskpath = '/data0/huayu/Aluochen/MyPaper1.1/random_368x368_acs24_R6_split_2.mat'

elif dataset =='256x232mask6acs24' :
    subDirectory = '256x232mask6acs24' 
    gettestdata = getTestingData1
    result_path = dataset
    maskpath = '/data2/luochen/model/random_256x232_acs24_R6_split_2.mat'

   
seed = 1
lr = 5e-4
epochs = 200
batch_size = 1
gamma = 0.99
restoreWeights = False
#model_dir = "/data0/huayu/Aluochen/selfSLRknee/SLRDS0H1%s/%s_trainn_%d_lg_%.4f" % (args.model_dir, name, lr, epochs)
#model_dir = "./SLRDS0H2/%s_trainn_%d_lg_%.4f" % (name, lr, epochs)
model_dir = 'DS0K'+subDirectory +"/trainn_%d_lg_%.4f" % (lr, epochs)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

ksp, atb, atb1, M, M1 = getdata(maskpath)

ksp = torch.from_numpy(ksp)
atb = torch.from_numpy(atb)
atb1 = torch.from_numpy(atb1)
M = torch.from_numpy(M)
M1 = torch.from_numpy(M1)
nb, nc, nx, ny = atb.shape
print(atb.shape)


class Mydata(Data.Dataset):
    def __init__(self):
        super(Mydata,self).__init__()
        self.ksp = ksp
        self.atb = atb
        self.atb1 = atb1
        self.M = M
        self.M1 = M1
    def __getitem__(self,index):
        kspk = self.ksp[index]
        atbk = self.atb[index]
        atb1k = self.atb1[index]
        M = self.M[index]
        M1 = self.M1[index]
        return kspk, atbk, atb1k, M, M1
    def __len__(self):
        return len(self.atb)


trainset = Mydata()
train_loader = Data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0, pin_memory=False)

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, m, init=None):
        if init is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = init
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, m, x), init_point, **self.kwargs)
        z = self.f(z, m, x)
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, m, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

##f = Landweber(LayerNo=10, conv_size=128, nch=12, filter_size = 9, pad = 4)
##f = Landweber1(nch=24)

f = SLRK(LayerNo=10, conv_size=64, nch=nc, filter_size=3, pad=1)
#model = f
model = DEQFixedPoint(f, anderson, tol=1e-6, max_iter=10, beta = 1.0) # f #
model = model.to(device)
paramsG = model.parameters()
optimG = torch.optim.Adam(paramsG, lr=lr, amsgrad=False)
##scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimG, conf["epochs"]*360, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimG,gamma)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
start_epoch = -1
if restoreWeights:
    restorePATH = os.path.join(model_dir, "epoch_{}.pkl".format(30))
    start_epoch = torch.load(restorePATH)['epoch']
    model.load_state_dict(torch.load(restorePATH)['net_state_dict'])
    optimG.load_state_dict(torch.load(restorePATH)['optimizer_state_dict'])
    scheduler.load_state_dict(torch.load(restorePATH)['scheduler_state_dict'])
    
for epoch in range(start_epoch+1, epochs):

    loss_R = []

    for i, (ksp,atb,atb1,mask,mask1) in enumerate(train_loader):
        ksp,atb,atb1,mask,mask1 = ksp.type(torch.FloatTensor).to(device), atb.type(torch.FloatTensor).to(device),\
             atb1.type(torch.FloatTensor).to(device), mask.type(torch.FloatTensor).to(device),\
             mask1.type(torch.FloatTensor).to(device)
        
        optimG.zero_grad()
        
        x = model(atb1, mask1, atb1)
        res1 = (mask*x - atb)
        loss1 = torch.mean(torch.sum(torch.pow(torch.abs(res1), 2), 0))
        res2 = (mask-mask1)*(x - atb)
        loss2 = torch.mean(torch.sum(torch.pow(torch.abs(res2), 2), 0)) 
        loss = loss1 + 0.1*loss2
        loss.backward()
        optimG.step()
        lossr = torch.mean(torch.sum(torch.pow(torch.abs(ksp-x), 2), 0))
        loss_R += [lossr.item()]

    scheduler.step()
    print('LR: ', optimG.state_dict()['param_groups'][0]['lr'])  
       
    output_data = "[%02d/%02d] cost: %.6f costr: %.6f\n" % (
            epoch, i, loss, np.mean(loss_R))
    print(output_data)

    if (epoch+1) % 5 == 0:
#        torch.save(
#        model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch+1))  # save only the parameters

        PATH=os.path.join(model_dir, "epoch_{}.pkl".format(epoch+1))
        state = {'epoch': epoch,
             'net_state_dict':  model.state_dict(),
             'optimizer_state_dict': optimG.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()
              }

        torch.save(state, PATH)
        
        
        

