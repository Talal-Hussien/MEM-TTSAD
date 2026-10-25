"""Mem-TTSAD: TranAD-based time-series anomaly detection with parametric memory.

Backbones:  TranAD_Basic (single-phase basic Transformer), TranAD (two-phase,
            self-conditioning + adversarial).
Memory variants per backbone (Burtsev et al. 2020 memory mechanisms):
            Direct Memory (Mem_*), MemCtrl, MemBottleneck.

Structural refactor of the original mem-tranad research code; numerical
behavior (initialization, forward pass, gradients, training) is preserved
bit-for-bit. See tests/equivalence for the verification suite.
"""
