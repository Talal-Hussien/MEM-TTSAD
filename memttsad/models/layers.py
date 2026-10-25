"""Building-block layers shared by all models.

PositionalEncoding / TransformerEncoderLayer / TransformerDecoderLayer are the
original TranAD implementations (post-norm-free residual layers with LeakyReLU
feed-forwards); MemoryLayer is the two-line Burtsev et al. (2020) update used
by the MemCtrl / MemBottleneck streams. All four are copied verbatim — any
change here changes the numerics of every model.
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryLayer(nn.Module):
    """One post-LN Transformer sub-network with distinct query and key/value
    inputs, implementing the two-line update used by every stream in
    Burtsev et al. (2020):

        A = LN_1(X + MH(X, KV, KV))
        H = LN_2(A + FF(A))

    MH, FF, LN_1, LN_2 are private to the instance, so a MemCtrl / MemBottleneck
    model gets separate (MHmem, FFmem) and (MHseq, FFseq) parameters by holding
    two instances. Dropout placement follows TransformerEncoderLayer above."""

    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0.1):
        super(MemoryLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)   # MH
        self.linear1 = nn.Linear(d_model, dim_feedforward)                    # FF (in)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)                    # FF (out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)
        self.norm1 = nn.LayerNorm(d_model)                                    # LN (after MH)
        self.norm2 = nn.LayerNorm(d_model)                                    # LN (after FF)

    def feed_forward(self, a):
        return self.linear2(self.dropout(self.activation(self.linear1(a))))

    def forward(self, x, kv):
        a = self.norm1(x + self.dropout1(self.attn(x, kv, kv)[0]))            # A = LN(X + MH(X, KV, KV))
        h = self.norm2(a + self.dropout2(self.feed_forward(a)))               # H = LN(A + FF(A))
        return h
