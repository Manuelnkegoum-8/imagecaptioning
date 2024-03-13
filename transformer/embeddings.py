import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import random,math

class shift_patch(nn.Module):
    def __init__(self,patch_size=16):
        super().__init__()        
        self.patch_size = patch_size // 2
    def forward(self, x):
        bs,_, _, chanel = x.size()
        shifts = ((-self.patch_size, self.patch_size, -self.patch_size, self.patch_size), 
                    (self.patch_size, -self.patch_size, -self.patch_size, self.patch_size),
                    (-self.patch_size, self.patch_size, self.patch_size, -self.patch_size), 
                    (self.patch_size, -self.patch_size, self.patch_size, -self.patch_size))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        return shifted_x


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class patch_embedding(nn.Module):
    def __init__(self,height=224,width=224,n_channels=3,patch_size=16,dim=512):
        super().__init__()
        
        assert height%patch_size==0 and width%patch_size==0 ,"Height and Width should be multiples of patch size wich is {0}".format(patch_size)
        #self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = posemb_sincos_2d(
            h = height // patch_size,
            w = width // patch_size,
            dim = dim,
        ) 
        self.patch_size = patch_size
        self.n_patchs = height*width//(patch_size**2)
        #self.embedding = torch.nn.Parameter(torch.randn(1,self.n_patchs+1, dim))
        self.projection = nn.Sequential(nn.LayerNorm(patch_size*patch_size*n_channels*5),
                                        nn.Linear(patch_size*patch_size*n_channels*5,dim),
                                        nn.LayerNorm(dim)
                                       )
        self.shift = shift_patch(patch_size=patch_size)
        self.patch = nn.Sequential(
            Rearrange("b c (h p1) (w p2)  -> b (h w)  (p1 p2 c)", p1 = patch_size, p2 = patch_size),
        )
    def forward(self, x):
        #x bs,h,w,c
        embedding = self.pos_embedding.to(x.device)
        left_up,right_up,left_down,right_down = self.shift(x)
        x = torch.cat([x,left_up,right_up,left_down,right_down],dim=1)
        #first we resize the inputs Bs,Num_patchs,*
        x = self.patch(x)
        #projection on the dim of model
        x = self.projection(x)
        outputs = x + embedding
        return outputs


