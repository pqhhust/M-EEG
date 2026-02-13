import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import concurrent.futures
from torch.jit import fork, wait  
import torch.nn.init as init
import torch.nn.functional as F
import math
from .cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location='cpu'))
        self.backbone.proj_out = nn.Identity()
        if param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 100),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(100, param.num_of_classes),
            )
        
    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        # out = feats
        return out
            