"""Small CNN teaching models, separate from frozen research checkpoints."""
import torch
from torch import nn
from torch.nn import functional as F


class Reconstruction(nn.Module):
    def __init__(self,bands=3,residual=False):
        super().__init__();self.residual=residual
        self.layers=nn.Sequential(nn.Conv2d(bands,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,bands,3,padding=1))
    def forward(self,x):
        y=self.layers(x)
        return x+y if self.residual else y


class UpsamplingCNN(nn.Module):
    def __init__(self,mode='shuffle',residual=True):
        super().__init__()
        if mode not in ['shuffle','transpose','interpolate']:raise ValueError('Unknown upsampling mode')
        self.mode,self.residual=mode,residual
        self.head=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU())
        if mode=='shuffle':self.tail=nn.Sequential(nn.Conv2d(8,12,3,padding=1),nn.PixelShuffle(2))
        elif mode=='transpose':self.tail=nn.ConvTranspose2d(8,3,3,stride=2,padding=1,output_padding=1)
        else:self.tail=nn.Conv2d(8,3,3,padding=1)
    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3:raise ValueError('NCHW RGB required')
        features=self.head(x)
        if self.mode=='interpolate':features=F.interpolate(features,scale_factor=2,mode='bilinear',align_corners=False)
        y=self.tail(features)
        return y+F.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False) if self.residual else y
