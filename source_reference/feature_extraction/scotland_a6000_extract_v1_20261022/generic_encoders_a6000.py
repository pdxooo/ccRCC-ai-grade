"""Frozen ImageNet-1K encoders. Strict load before removing classification heads.
lab-a6000 copy of gastritis_uni2h/outputs/generic-encoders-20261008/code/generic_encoders.py: the only change is the
HE_PATH_PREFIX mirror root in front of WEIGHTS."""
import os
from pathlib import Path
import hashlib
import torch
from torch import nn
from torchvision import models

WEIGHTS = Path(os.environ.get('HE_PATH_PREFIX', '').rstrip('/') + '/data/cluster/software/uni2h/models/imagenet')
SPECS = {
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    'vit_b_16': (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, 768),
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def build(name):
    constructor, weights, dim = SPECS[name]
    path = WEIGHTS / weights.url.rsplit('/', 1)[-1]
    digest = sha(path)
    assert digest.startswith(path.stem.rsplit('-', 1)[-1]), 'Official hash mismatch'
    model = constructor(weights=None)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    if name == 'resnet50': model.fc = nn.Identity()
    elif name == 'convnext_tiny': model.classifier[-1] = nn.Identity()
    else: model.heads = nn.Identity()
    return model.eval().requires_grad_(False), weights.transforms(), {
        'name': name, 'dimension': dim, 'weight_enum': str(weights),
        'url': weights.url, 'weight_path': str(path), 'weight_sha256': digest,
        'preprocessing': repr(weights.transforms()), 'strict_load': True,
        'pretraining': 'ImageNet-1K', 'frozen': True,
    }
