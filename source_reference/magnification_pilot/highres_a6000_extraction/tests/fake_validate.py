"""TEST FAKE of validate.py (pathology adapters) for local CPU tests: tiny deterministic encoders with the same
build(name) -> (model, preprocess, info) / forward(name, model, x) interface and the same weight-path layout."""
import os
from pathlib import Path
import torch
from torch import nn
ASSETS=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/software/uni2h/models')
BIN=Path(os.environ.get('HE_PATH_PREFIX','').rstrip('/')+'/data/cluster/bin')
DIMS={'uni':1024,'virchow2':2560}

class Tiny(nn.Module):
    def __init__(self,d,seed):
        super().__init__();g=torch.Generator().manual_seed(seed)
        self.pool=nn.AdaptiveAvgPool2d(8);self.lin=nn.Linear(192,d)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(d,192,generator=g)*.2);self.lin.bias.copy_(torch.randn(d,generator=g)*.1)
    def forward(self,x):return torch.tanh(self.lin(self.pool(x).flatten(1)))

def build(name):
    path={'uni':ASSETS/'uni'/'pytorch_model.bin','virchow2':BIN/'pytorch_model(3).bin'}[name]
    return Tiny(DIMS[name],{'uni':1,'virchow2':2}[name]),None,{'weight':str(path)}

def forward(name,m,x):return m(x)
