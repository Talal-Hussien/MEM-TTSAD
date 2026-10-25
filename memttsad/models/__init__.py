"""Model zoo: the two TranAD backbones and their three memory variants each.

IMPORTANT: importing this package seeds torch's RNG from TRANAD_SEED (default
1), exactly as the original src/models.py did. Model parameters are drawn from
that stream in construction order, so the seed call must precede any model
construction for runs to be reproducible (and comparable with the original
code base).
"""
import os

import torch

torch.manual_seed(int(os.environ.get('TRANAD_SEED', 1)))

from memttsad.models.backbones import TranAD_Basic, TranAD
from memttsad.models.memory_basic import (Mem_TTSAD_Basic, MemCtrl_TTSAD_Basic,
                                          MemBottleneck_TTSAD_Basic)
from memttsad.models.memory_tranad import Mem_TTSAD, MemCtrl_TTSAD, MemBottleneck_TTSAD

MODEL_REGISTRY = {cls.__name__: cls for cls in (
    TranAD_Basic, TranAD,
    Mem_TTSAD_Basic, MemCtrl_TTSAD_Basic, MemBottleneck_TTSAD_Basic,
    Mem_TTSAD, MemCtrl_TTSAD, MemBottleneck_TTSAD,
)}

BACKBONES = ('TranAD_Basic', 'TranAD')


def build_model(name, feats, lr=None, n_memory=None):
    """Construct a registered model. lr is the dataset learning rate
    (config.get_lr); n_memory=None lets memory models fall back to the
    MEMTRANAD_M environment variable (their original default)."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}')
    cls = MODEL_REGISTRY[name]
    if name in BACKBONES:
        return cls(feats, lr=lr)
    return cls(feats, n_memory=n_memory, lr=lr)
