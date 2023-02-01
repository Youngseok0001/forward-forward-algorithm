import sys, os, tqdm

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from toolz.itertoolz import *

import numpy as np

import torch
from torch.autograd import Function
from torch import functional as F
from torch import nn
from torch.optim import Adam

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from torchvision import transforms as T

from sklearn.metrics import confusion_matrix
################################################################################
def MNISTDataLoader(train, batchsize = 4000):
    
    aug = T.Compose([T.ToTensor(),
                     T.Normalize(0.5, 0.5),
                     T.Lambda(lambda x: torch.flatten(x))])
                    
    return DataLoader(MNIST(root  = "./", train = train, download = True, transform=aug),
                      batch_size  = batchsize,
                      shuffle     = train,
                      num_workers = 16)
################################################################################
def overlay_y_on_x(xs, ys):
    xs_ = xs.clone()
    xs_[:, :10] *= 0.0
    xs_[range(xs.shape[0]), ys] = xs.max()
    return xs_
################################################################################
class _ZeroGradient(Function):
    @staticmethod
    def forward(_, x):
        return x
    
    @staticmethod
    def backward(_, grad_output):
        return grad_output*0.
    
class ZeroGradient(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return _ZeroGradient.apply(x)
    
class FFLayer(nn.Module):
    
    def __init__(self, cIn, cOut):
        super().__init__()
        
        self.zerog  = ZeroGradient()
        self.norm   = nn.LayerNorm(cIn)
        self.linear = nn.Linear(cIn, cOut)
        self.relu   = nn.ReLU()
        
    def forward(self, x):        
        return compose(self.relu,
                       self.linear,
                       self.norm,
                       self.zerog)(x) 
    
class FFNet(nn.Module):
    
    def __init__(self, xShape = [28, 28], COuts = [2000, 2000, 2000, 2000]):
        super().__init__()
        
        cIns  = [mul(*xShape)] + COuts[:-1]
    
        self.layers = compose(nn.ModuleList,
                              map(lambda _ : FFLayer(*_)))(zip(cIns, COuts))
        
    def forward(self, x):
        f = lambda acc, layer : acc + [layer(last(acc))]
        return reduce(f, self.layers, [x])[1:]    
#################################################################################
class Loss(nn.Module):
    
    def __init__(self, theta = 1.5):
        super().__init__()
        
        self.theta = theta
        
    def forward(self, ps, ns):
        
        ploss = torch.log(1 + torch.exp((-ps.pow(2).mean(1) + self.theta))).mean()
        nloss = torch.log(1 + torch.exp((+ns.pow(2).mean(1) - self.theta))).mean()
        
        return ploss + nloss
    
if __name__ == "__main__":
    
    lr        = 0.002
    batchsize = 2000
    epoch     = 100
    imgShape  = [28, 28]
    nHidden   = [2000, 2000, 2000, 2000]
    theta     = 1.5 

    trainLoader = MNISTDataLoader(train = True,  batchsize = batchSize)
    testLoader  = MNISTDataLoader(train = False, batchsize = 10000)
    net         = FFNet(xShape = imgShape, COuts = nHidden).cuda()
    optimizer   = Adam(net.parameters(), lr = lr)
    lossf       = Loss(theta = theta)
    #################################################################################
    net.train()
    for e in range(epoch):

        losses = []
        for xs, ys in trainLoader:

            ps = overlay_y_on_x(xs, ys)
            ns = overlay_y_on_x(xs, ys[torch.randperm(xs.size(0))]) # this has error but model trains still

            pss = net(ps.cuda())
            nss = net(ns.cuda())

            loss = sum([lossf(ps, ns) for ps, ns in zip(pss, nss)])

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            losses.append(loss.detach().cpu())

        if e%10 == 0 : print(f"epoch:{e} loss:{np.array(losses).mean()}")
    #################################################################################        
    net.eval()
    xs, ys = next(iter(testLoader))

    xss = [overlay_y_on_x(xs, (torch.ones(xs.shape[0])*i).to(torch.long) ) for i in range(10)]
    xs  = torch.cat(xss).cuda() 

    preds = (torch.stack(net(xs))
             .sum(0)
             .pow(2)
             .mean(1)
             .reshape(10, -1)
             .argmax(0))

    ys    = ys.cpu().numpy()
    preds = preds.detach().cpu().numpy()

    print(confusion_matrix(ys, preds))
    print(sum(ys == preds) / len(preds) * 100)