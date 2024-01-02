'''
Code copied from https://github.com/mahmoodlab/SurvPath/blob/main/models/model_TMIL.py and reduced to single modality
'''

from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.model_utils import *
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_vals = self.attn(self.norm(x),return_attn=return_attn)
            x = x + attn_out
            return x, attn_vals
        else:
            attn_out = self.attn(self.norm(x),return_attn=return_attn)
            x = x + attn_out
            return x


class PEG(nn.Module):
    def __init__(self, dim=256, k=7):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)


    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

'''
Code copied from https://github.com/szc19990412/HVTSurv/blob/main/models/TransMIL.py and slightly modified for this codebase
'''
# It's challenging for TransMIL to process all the high-dimensional data in the patient-level bag, so we reduce the dimension from 1024 to 128.
class TransMIL_peg(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL_peg, self).__init__()
        self.pos_layer = PEG(128)
        self._fc1 = nn.Sequential(nn.Linear(768, 128), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=128)
        self.layer2 = TransLayer(dim=128)
        self.norm = nn.LayerNorm(128)
        # self._fc2 = nn.Linear(128, self.n_classes)
        self._fc2 = nn.Sequential(*[nn.Linear(128,128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, n_classes)])

    def forward(self, x, return_attn=False):

        h = x.float().unsqueeze(0) #[B, n, 768]
        
        #---->Dimensionality reduction first
        h = self._fc1(h) #[B, n, 128]
        
        #---->padding
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 128]

        #---->Add position encoding, after a transformer
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 128]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 128]
        
        #---->Translayer x2
        if return_attn:
            h, attn_vals = self.layer2(h,return_attn) #[B, N, 128]
        else:
            h = self.layer2(h)

        h = self.norm(h)[:,0]

        #---->predict output
        logits = self._fc2(h)
        if return_attn:
            return logits, attn_vals
        else:
            return logits

