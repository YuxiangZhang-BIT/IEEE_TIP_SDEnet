import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *

class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)

    def forward(self, x,):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            
            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap


class SpeRandomization(nn.Module):
    def __init__(self,num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, idx_swap,y=None):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            if y!= None:
                for i in range(len(y.unique())):
                    index= y==y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
        return x


class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s): 
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
        #return (1+gamma)*(x)+beta

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Generator(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[13, 13], zdim=10, device=0):
        ''' w_ln 局部噪声权重
        '''
        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        num_morph = 4
        self.Morphology = MorphNet(imdim)
        self.adain2_morph = AdaIN2d(zdim, num_morph)

        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])
        self.conv1 = nn.Conv2d(n+n+num_morph, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, imdim, kernelsize, 1, stride)
        self.speRandom = SpeRandomization(n)
        self.spaRandom = SpaRandomization(3, device=device)

    def forward(self, x): 

        x_morph= self.Morphology(x)
        z = torch.randn(len(x), self.zdim).to(self.device)
        x_morph = self.adain2_morph(x_morph, z)

        x_spa = F.relu(self.conv_spa1(x))
        x_spe = F.relu(self.conv_spe1(x))
        x_spa, idx_swap = self.spaRandom(x_spa)
        x_spe = self.speRandom(x_spe,idx_swap)
        x_spe = self.conv_spe2(x_spe)
        x_spa = self.conv_spa2(x_spa)
        
        x = F.relu(self.conv1(torch.cat((x_spa,x_spe,x_morph),1)))
        x = torch.sigmoid(self.conv2(x))

        return x


