import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import concurrent.futures
from torch.jit import fork, wait  
import torch.nn.init as init
import torch.nn.functional as F
import math
from .cbramod import CBraMod

class ParamAttention(nn.Module):
    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_k: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
        """
        x_k: (b, c, d), x_q: (b, d) -> out: (b, d)
        """
        q = self.Wq(x_q)              # (b, d)
        k = self.Wk(x_k)              # (b, c, d)
        v = self.Wv(x_k)              # (b, c, d)

        scores = torch.einsum('bcd,bd->bc', k, q) / math.sqrt(k.size(-1))  # (b, c)
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum('bc,bcd->bd', attn, v)  # (b, d)
        return out

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
        if param.modality_mode == 'multi':
            self.meta_backbone = nn.Sequential(
                nn.Linear(28, 100),
                nn.ReLU(),
                nn.Dropout(param.dropout),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Dropout(param.dropout)
            )
            if param.classifier == 'all_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(19 * 30 * 200, 30 * 200),
                    nn.ReLU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(30 * 200, 100),
                    nn.ReLU(),
                    nn.Dropout(param.dropout)
                )
            elif param.classifier == 'all_patch_reps_twolayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(19 * 30 * 200, 100),
                    nn.ReLU(),
                    nn.Dropout(param.dropout),
                )
            else:
                raise NotImplementedError(f'Classifier {param.classifier} is not implemented yet.')
            self.final_classifier = nn.Sequential(
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)')
            )
            self.forward = self.multi_forward
            # self.final_classifier.apply(init_weights)
            # self.classifier.apply(init_weights)
        elif param.modality_mode == 'mono': # mono
            if param.classifier == 'avgpooling_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b d c s'),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(200, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            elif param.classifier == 'all_patch_reps_onelayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(19 * 30 * 200, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            elif param.classifier == 'all_patch_reps_twolayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(19 * 30 * 200, 200),
                    nn.ReLU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(200, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            elif param.classifier == 'all_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(19 * 30 * 200, 30 * 200),
                    nn.ReLU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(30 * 200, 200),
                    nn.ReLU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(200, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            self.forward = self.mono_forward
        elif param.modality_mode == 'multi_attend':  # multi
            self.meta_backbone = nn.Sequential(
                nn.Linear(28, 100),
                nn.ReLU(),
                nn.Dropout(param.dropout),
                nn.Linear(100, 200),
                # nn.ReLU()
            )
            self.arranger = Rearrange('b c s d -> b (c s) d') # b, 19 * 30, 200
            self.attention = ParamAttention(d=200, dropout=param.dropout)
            self.classifier = nn.Sequential(
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)')
            )
            self.forward = self.multi_attend_forward
            

    def multi_attend_forward(self, x):
        # bz, ch_num, seq_len, patch_size = x.shape
        # feats = self.eeg_backbone(x[0])
        # meta_feats = self.meta_backbone(x[1])

        f1 = fork(self.eeg_backbone_attend, x[0])
        f2 = fork(self.meta_backbone, x[1])

        feats, meta_feats = wait(f1), wait(f2)  # feats: b, c, s, d; meta_feats: b, d
        feats = self.attention(feats, meta_feats)  # b, d
        out = self.classifier(feats)  # b, 1
        return out

    def eeg_backbone_attend(self, x):
        return self.arranger(self.backbone(x))

    def eeg_backbone(self, x):
        pot = self.backbone(x)
        return self.classifier(pot)

    def multi_forward(self, x):
        # bz, ch_num, seq_len, patch_size = x.shape
        # feats = self.eeg_backbone(x[0])
        # meta_feats = self.meta_backbone(x[1])

        f1 = fork(self.eeg_backbone, x[0])
        f2 = fork(self.meta_backbone, x[1])

        feats, meta_feats = wait(f1), wait(f2)
        feats = torch.cat([feats, meta_feats], dim=1)
        # feats = feats + meta_feats
        out = self.final_classifier(feats)
        return out
            
    def mono_forward(self, x):
        # bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x[0])
        out = self.classifier(feats)
        return out