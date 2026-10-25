"""Memory variants on the full TranAD backbone (two phases, two decoders,
self-conditioning): Direct Memory (Mem_TTSAD), MemCtrl and MemBottleneck
(Burtsev et al. 2020) sharing the _MemoryTran scaffolding.
"""
import math
import os

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder

from memttsad.models.layers import (PositionalEncoding, TransformerEncoderLayer,
                                    TransformerDecoderLayer, MemoryLayer)
from memttsad.models.variants import record_memory


class Mem_TTSAD(nn.Module):
    """Direct Memory on the TranAD backbone: m learnable memory tokens are
    prepended to the input sequence (in F-space, before the conditioning
    concat) and flow through the unchanged encoder; the decoders read the
    FULL encoder output [Hmem; Hseq]."""

    def __init__(self, feats, n_memory=None, lr=None):
        super().__init__()
        if n_memory is None: n_memory = int(os.environ.get('MEMTRANAD_M', 4))
        self.name = 'Mem_TTSAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.n_memory = n_memory

        # Backbone: identical modules, in the identical order, to TranAD.
        # PE length is n_window + n_memory: the m memory tokens occupy
        # positions 0..m-1 and receive positional encoding (inherited
        # Mem_TranAD-family quirk).
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window + self.n_memory)
        enc_layer = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        dec1_layer = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        dec2_layer = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(enc_layer, 1)
        self.transformer_decoder1 = TransformerDecoder(dec1_layer, 1)
        self.transformer_decoder2 = TransformerDecoder(dec2_layer, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

        # m learnable memory tokens, defined LAST (init-order matters for RNG
        # parity with the memory-free backbone). Raw shape [m, 1, F]: memory
        # lives in F-space and is prepended BEFORE the conditioning concat.
        self.mem_token = nn.Parameter(torch.randn(self.n_memory, 1, feats) * 0.02)

    def encode(self, sequence_with_memory, conditioning, tgt):
        """Identical to TranAD.encode except for the sqrt(F) scaling (none here)
        and the PE covering all memory positions."""
        src2f = torch.cat((sequence_with_memory, conditioning), dim=2)  # [W+m, B, 2F]
        enc_in = self.pos_encoder(src2f)
        encoder_output = self.transformer_encoder(enc_in)               # [W+m, B, 2F]
        return tgt.repeat(1, 1, 2), encoder_output

    def _phase(self, src, tgt, decoder, c=None):
        """One TranAD phase; decoder KV is the FULL encoder output
        [Hmem; Hseq] -- all m memory positions included."""
        memory_tokens = self.mem_token.expand(-1, src.shape[1], -1)           # [m, B, F]
        sequence_with_memory = torch.cat((memory_tokens, src), dim=0)         # [W+m, B, F]
        conditioning = torch.zeros_like(sequence_with_memory) if c is None else c
        tgt_rep, encoder_output = self.encode(sequence_with_memory, conditioning, tgt)
        record_memory(self, encoder_output[:self.n_memory])                  # Hmem = encoder memory positions
        x = self.fcn(decoder(tgt_rep, encoder_output))                        # [1, B, F]
        return x, sequence_with_memory

    def forward(self, src, tgt):
        # Phase 1: zero conditioning, decoder 1; the SAME mem_token bank as phase 2.
        x1, seq1 = self._phase(src, tgt, self.transformer_decoder1, c=None)

        # Phase 2: frozen conditioning behavior, (x1 - seq1)**2 broadcast over
        # the memory-augmented sequence (all m memory positions included);
        # decoder 2.
        c2 = (x1 - seq1) ** 2
        x2, _ = self._phase(src, tgt, self.transformer_decoder2, c=c2)
        return x1, x2


class _MemoryTran(nn.Module):
    """Shared TranAD scaffolding for the two paper-faithful models. Subclasses
    only define how Hseq is computed (the single point where MemCtrl and
    MemBottleneck differ in the paper)."""

    def __init__(self, feats, n_memory=None, lr=None):
        super(_MemoryTran, self).__init__()
        if n_memory is None: n_memory = int(os.environ.get('MEMTRANAD_M', 2))
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n_memory = n_memory
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.mem_tokens = nn.Parameter(torch.randn(n_memory, 1, 2 * feats) * 0.02)       # Xmem (learnable)
        self.mem_layer = MemoryLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)  # MHmem, FFmem, LN
        self.seq_layer = MemoryLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)  # MHseq, FFseq, LN
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def _inputs(self, src, c):
        """Xseq (TranAD encoder input, unchanged) and Xmem (expanded tokens)."""
        z = torch.cat((src, c), dim=2)
        z = z * math.sqrt(self.n_feats)
        xseq = self.pos_encoder(z)                                   # Xseq  [W, B, 2F]
        xmem = self.mem_tokens.expand(-1, xseq.shape[1], -1)         # Xmem  [m, B, 2F]
        return xmem, xseq

    def _memory_update(self, xmem, xseq):
        # Amem = LN(Xmem + MHmem(Xmem, [Xmem;Xseq], [Xmem;Xseq])); Hmem = LN(Amem + FFmem(Amem))
        return self.mem_layer(xmem, torch.cat((xmem, xseq), dim=0))

    def _sequence_update(self, xmem, xseq, hmem):
        raise NotImplementedError

    def _encode_parts(self, src, c):
        xmem, xseq = self._inputs(src, c)
        hmem = self._memory_update(xmem, xseq)                       # Hmem  [m, B, 2F]
        hseq = self._sequence_update(xmem, xseq, hmem)               # Hseq  [W, B, 2F]
        return hmem, hseq

    def _phase(self, src, c, tgt, decoder):
        hmem, hseq = self._encode_parts(src, c)
        record_memory(self, hmem)                                    # Hmem read by this phase's decoder
        tgt2 = tgt.repeat(1, 1, 2)
        return self.fcn(decoder(tgt2, torch.cat((hmem, hseq), dim=0)))   # decoder KV = [Hmem; Hseq]

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self._phase(src, c, tgt, self.transformer_decoder1)

        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self._phase(src, c, tgt, self.transformer_decoder2)
        return x1, x2


class MemCtrl_TTSAD(_MemoryTran):
    """MemCtrl Transformer (Burtsev et al. 2020, Sec. 2.3) inside TranAD.
    Memory and sequence are updated IN PARALLEL from the layer input
    [Xmem; Xseq] by two separate sub-networks; the sequence attends to
    memory and to itself through ONE multi-head attention over the
    concatenated keys. No gate."""

    def __init__(self, feats, n_memory=None, lr=None):
        super(MemCtrl_TTSAD, self).__init__(feats, n_memory, lr)
        self.name = 'MemCtrl_TTSAD'

    def _sequence_update(self, xmem, xseq, hmem):
        # Aseq = LN(Xseq + MHseq(Xseq, [Xmem;Xseq], [Xmem;Xseq])); Hseq = LN(Aseq + FFseq(Aseq))
        return self.seq_layer(xseq, torch.cat((xmem, xseq), dim=0))


class MemBottleneck_TTSAD(_MemoryTran):
    """MemBottleneck Transformer (Burtsev et al. 2020, Sec. 2.4) inside TranAD.
    Step 1: memory attends to [Xmem; Xseq] and is updated to Hmem.
    Step 2: the sequence attends ONLY to Hmem (no sequence-to-sequence
    attention path). No gate."""

    def __init__(self, feats, n_memory=None, lr=None):
        super(MemBottleneck_TTSAD, self).__init__(feats, n_memory, lr)
        self.name = 'MemBottleneck_TTSAD'

    def _sequence_update(self, xmem, xseq, hmem):
        # Aseq = LN(Xseq + MHseq(Xseq, Hmem, Hmem)); Hseq = LN(Aseq + FFseq(Aseq))
        return self.seq_layer(xseq, hmem)
