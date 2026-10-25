"""Memory variants on the TranAD_Basic backbone (single phase, one decoder):
Direct Memory (Mem_TTSAD_Basic), MemCtrl and MemBottleneck (Burtsev et al.
2020) sharing the _MemoryTranBasic scaffolding, plus the default-off
experiment switches (decoder variants, init-norm parity, retrieval).
"""
import math
import os

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder

from memttsad.models.layers import (PositionalEncoding, TransformerEncoderLayer,
                                    TransformerDecoderLayer, MemoryLayer)
from memttsad.models.variants import (record_memory, mem_token_init_scale,
                                      decoder_variant_config, memory_retrieval_config)


class Mem_TTSAD_Basic(nn.Module):
    """TranAD_Basic + m learnable memory tokens with direct decoder access.

        Z = PE(sqrt(F) * X)          [W, B, F]
        E = Encoder([M ; Z])         [W+m, B, F]   (M: nn.Parameter [m,1,F])
        O = Sigmoid(Decoder(tgt, E))                (full KV = [Hmem; Hseq])

    Research question: does simply adding learnable parametric memory
    improve the basic single-phase Transformer baseline?"""

    def __init__(self, feats, n_memory=None, lr=None):
        super().__init__()
        if n_memory is None: n_memory = int(os.environ.get('MEMTRANAD_M', 2))
        self.name = 'Mem_TTSAD_Basic'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n_memory = n_memory
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()
        self.mem_tokens = nn.Parameter(torch.randn(n_memory, 1, feats) * 0.02)

    def forward(self, src, tgt):
        z = src * math.sqrt(self.n_feats)
        z = self.pos_encoder(z)                                # PE on real timestamps only
        m = self.mem_tokens.expand(-1, z.shape[1], -1)         # [m, B, F], no temporal PE
        memory = self.transformer_encoder(torch.cat((m, z), dim=0))   # [W+m, B, F]
        record_memory(self, memory[:self.n_memory])            # Hmem = encoder memory positions
        x = self.transformer_decoder(tgt, memory)              # full KV: [Hmem; Hseq]
        return self.fcn(x)

    def reset_memory(self):
        pass  # no runtime state


class _MemoryTranBasic(nn.Module):
    """Shared TranAD_Basic scaffolding for the two paper-faithful Basic models.
    Subclasses define only the sequence update (the single point where the
    MemCtrl and MemBottleneck equations differ)."""

    def __init__(self, feats, n_memory=None, lr=None):
        super(_MemoryTranBasic, self).__init__()
        if n_memory is None: n_memory = int(os.environ.get('MEMTRANAD_M', 2))
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n_memory = n_memory
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        _scale, self.mem_init_norm = mem_token_init_scale(feats)                           # 0.02 unless --memory_init_norm
        self.mem_tokens = nn.Parameter(torch.randn(n_memory, 1, feats) * _scale)        # Xmem (learnable slots)
        self.mem_layer = MemoryLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)  # MHmem, FFmem, LN, LN
        self.seq_layer = MemoryLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)  # MHseq, FFseq, LN, LN
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()
        # ---- default-off decoder/encoder path switches (all off => bit-identical to the original model) ----
        self.decoder_query_mode, self.decoder_keys_mode, self.history_only, self.no_memory = decoder_variant_config()
        if self.decoder_query_mode == 'learned':
            self.dec_query = nn.Parameter(torch.zeros(1, 1, feats))               # learned decoder query, zero-initialised
        if self.no_memory:                                                        # F: no memory anywhere in the graph
            assert type(self).__name__ == 'MemCtrl_TTSAD_Basic', 'no_memory is defined for MemCtrl_TTSAD_Basic only'
            del self.mem_tokens; del self.mem_layer; self.n_memory = 0
        # ---- default-off input-conditioned memory retrieval (created LAST so every pre-existing weight keeps its RNG draw) ----
        self.memory_retrieval, self.retrieval_dim, self.retrieval_context = memory_retrieval_config()
        self.retrieval_active = True                                              # runtime switch for the causal test (inference only)
        if self.memory_retrieval:
            assert not self.no_memory, 'memory retrieval needs memory'
            self.ret_q = nn.Linear(feats, self.retrieval_dim)                     # context -> retrieval query
            self.ret_k = nn.Linear(feats, self.retrieval_dim)                     # updated slot -> retrieval key
            self.ret_out = nn.Linear(feats, feats)                                # retrieved vector -> decoder residual (zero-initialised)
            nn.init.zeros_(self.ret_out.weight); nn.init.zeros_(self.ret_out.bias)
            self.ret_gate = nn.Parameter(torch.zeros(feats))                      # per-feature gate g = sigmoid(ret_gate) (0.5 at init)

    def _inputs(self, src):
        """Xseq exactly as TranAD_Basic builds its encoder input; Xmem = expanded slots.
        history_only: the window's last row (the target) is removed from the sequence input, so no encoder path sees it."""
        if self.history_only: src = src[:-1]
        xseq = self.pos_encoder(src * math.sqrt(self.n_feats))       # Xseq  [W or W-1, B, F]
        xmem = self.mem_tokens.expand(-1, xseq.shape[1], -1) if not self.no_memory else None   # Xmem  [m, B, F]
        return xmem, xseq

    def _decoder_inputs(self, hmem, hseq, tgt):
        """Query and key/value set of the decoder according to the switches (original: tgt, [Hmem; Hseq])."""
        q = self.dec_query.expand(1, tgt.shape[1], -1) if self.decoder_query_mode == 'learned' else tgt
        if self.decoder_keys_mode == 'all': kv = torch.cat((hmem, hseq), dim=0)
        elif self.decoder_keys_mode == 'memory': kv = hmem
        else: kv = hseq
        return q, kv

    def _memory_update(self, xmem, xseq):
        # Amem = LN(Xmem + MHmem(Xmem, [Xmem;Xseq], [Xmem;Xseq])); Hmem = LN(Amem + FFmem(Amem))
        return self.mem_layer(xmem, torch.cat((xmem, xseq), dim=0))

    def _sequence_update(self, xmem, xseq, hmem):
        raise NotImplementedError

    def _encode_parts(self, src):
        xmem, xseq = self._inputs(src)
        if self.no_memory:                                           # F: sequence stream attends to itself only
            return None, self.seq_layer(xseq, xseq)
        hmem = self._memory_update(xmem, xseq)                       # Hmem  [m, B, F]
        hseq = self._sequence_update(xmem, xseq, hmem) if self.decoder_keys_mode != 'memory' else None   # Hseq [W, B, F]; unused (and not computed) when the decoder reads memory only
        return hmem, hseq

    def _retrieve(self, hmem, hseq):
        """Input-conditioned retrieval over the m updated slots. hmem [m, B, F], hseq [W, B, F].
        Returns (contribution [1, B, F] already gated, alpha [B, m], logits [B, m])."""
        c = hseq.mean(0) if self.retrieval_context == 'mean' else hseq[-1]                 # context [B, F]
        H = hmem.permute(1, 0, 2)                                                           # [B, m, F]
        logits = (self.ret_q(c).unsqueeze(1) * self.ret_k(H)).sum(-1) / math.sqrt(self.retrieval_dim)   # [B, m]
        alpha = torch.softmax(logits, dim=-1)
        r = (alpha.unsqueeze(-1) * H).sum(1)                                                # retrieved slot mixture [B, F]
        contrib = (torch.sigmoid(self.ret_gate) * self.ret_out(r)).unsqueeze(0)             # gated, [1, B, F]
        return contrib, alpha, logits

    def forward(self, src, tgt):
        hmem, hseq = self._encode_parts(src)
        if hmem is not None: record_memory(self, hmem)                       # Hmem read by the decoder (diversity loss sees the real forward pass)
        q, kv = self._decoder_inputs(hmem, hseq, tgt)                        # original: q = tgt, KV = [Hmem; Hseq]
        x = self.transformer_decoder(q, kv)
        if self.memory_retrieval and self.retrieval_active:
            contrib, alpha, logits = self._retrieve(hmem, hseq)
            if getattr(self, 'collect_retrieval', False): self._retrieval = (alpha.detach(), logits.detach(), contrib.detach())
            x = x + contrib                                                  # added to the decoder output BEFORE the sigmoid
        return self.fcn(x)


class MemCtrl_TTSAD_Basic(_MemoryTranBasic):
    """MemCtrl Transformer (Burtsev et al. 2020, Sec. 2.3) on the TranAD_Basic
    backbone. Memory and sequence are updated IN PARALLEL from the layer input
    [Xmem; Xseq] by two sub-networks with separate parameters; the sequence
    attends to memory and to itself through ONE attention over the
    concatenated keys. Hseq does not depend on Hmem. No gate."""

    def __init__(self, feats, n_memory=None, lr=None):
        super(MemCtrl_TTSAD_Basic, self).__init__(feats, n_memory, lr)
        self.name = 'MemCtrl_TTSAD_Basic'

    def _sequence_update(self, xmem, xseq, hmem):
        # Aseq = LN(Xseq + MHseq(Xseq, [Xmem;Xseq], [Xmem;Xseq])); Hseq = LN(Aseq + FFseq(Aseq))
        return self.seq_layer(xseq, torch.cat((xmem, xseq), dim=0))


class MemBottleneck_TTSAD_Basic(_MemoryTranBasic):
    """MemBottleneck Transformer (Burtsev et al. 2020, Sec. 2.4) on the
    TranAD_Basic backbone. Step 1: memory attends to [Xmem; Xseq] -> Hmem.
    Step 2: the sequence attends ONLY to Hmem (no sequence-to-sequence
    attention path). No gate."""

    def __init__(self, feats, n_memory=None, lr=None):
        super(MemBottleneck_TTSAD_Basic, self).__init__(feats, n_memory, lr)
        self.name = 'MemBottleneck_TTSAD_Basic'

    def _sequence_update(self, xmem, xseq, hmem):
        # Aseq = LN(Xseq + MHseq(Xseq, Hmem, Hmem)); Hseq = LN(Aseq + FFseq(Aseq))
        return self.seq_layer(xseq, hmem)
