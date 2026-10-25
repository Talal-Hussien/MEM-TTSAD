"""Decoder-variant switches for MemCtrl_TTSAD_Basic: default-off equivalence, shapes, gradient flow, target leakage,
checkpoint compatibility, numerical stability. Run: python -m unittest tests.test_decoder_variants -v"""
import math
import os
import unittest

import torch

import memttsad.models as SM

F_, M_, W_, B_ = 38, 3, 10, 7
CELLS = {'A': dict(), 'B': dict(MEMTRANAD_DECODER_QUERY='learned'), 'C': dict(MEMTRANAD_DECODER_KEYS='memory'),
         'D': dict(MEMTRANAD_DECODER_QUERY='learned', MEMTRANAD_DECODER_KEYS='memory'),
         'E': dict(MEMTRANAD_DECODER_QUERY='learned', MEMTRANAD_DECODER_KEYS='memory', MEMTRANAD_HISTORY_ONLY='1'),
         'F': dict(MEMTRANAD_DECODER_QUERY='learned', MEMTRANAD_DECODER_KEYS='sequence', MEMTRANAD_HISTORY_ONLY='1', MEMTRANAD_NO_MEMORY='1')}
ENV_KEYS = ['MEMTRANAD_DECODER_QUERY', 'MEMTRANAD_DECODER_KEYS', 'MEMTRANAD_HISTORY_ONLY', 'MEMTRANAD_NO_MEMORY', 'MEMTRANAD_MEM_INIT_NORM']


def build(cell, seed=1):
    for k in ENV_KEYS: os.environ.pop(k, None)
    os.environ.update(CELLS[cell])
    torch.manual_seed(seed)
    m = SM.MemCtrl_TTSAD_Basic(F_, M_).double()
    for k in ENV_KEYS: os.environ.pop(k, None)
    return m


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    win = torch.rand(W_, B_, F_, generator=g, dtype=torch.float64); return win, win[-1:].clone()


class TestDecoderVariants(unittest.TestCase):
    def test_default_off_is_original(self):
        a = build('A'); self.assertEqual((a.decoder_query_mode, a.decoder_keys_mode, a.history_only, a.no_memory), ('target', 'all', False, False))
        self.assertFalse(hasattr(a, 'dec_query')); self.assertEqual(a.n_memory, M_)
        torch.manual_seed(1); ref = torch.randn(M_, 1, F_) * 0.02; self.assertTrue(torch.equal(a.mem_tokens.detach().float(), ref.float()))
        # forward identical to the hand-written original computation
        win, tgt = batch(); a.eval()
        with torch.no_grad():
            xseq = a.pos_encoder(win * math.sqrt(F_)); xmem = a.mem_tokens.expand(-1, B_, -1)
            hmem = a.mem_layer(xmem, torch.cat((xmem, xseq), 0)); hseq = a.seq_layer(xseq, torch.cat((xmem, xseq), 0))
            ref = a.fcn(a.transformer_decoder(tgt, torch.cat((hmem, hseq), 0)))
            self.assertTrue(torch.equal(a(win, tgt), ref))

    def test_shapes_and_parameters(self):
        win, tgt = batch()
        for cell in CELLS:
            m = build(cell); m.eval()
            with torch.no_grad(): out = m(win, tgt)
            self.assertEqual(tuple(out.shape), (1, B_, F_), cell)
            names = dict(m.named_parameters())
            self.assertEqual('dec_query' in names, CELLS[cell].get('MEMTRANAD_DECODER_QUERY') == 'learned', cell)
            self.assertEqual('mem_tokens' in names, cell != 'F', cell); self.assertEqual(any(n.startswith('mem_layer') for n in names), cell != 'F', cell)
            if cell != 'F':
                hmem, hseq = m._encode_parts(win); self.assertEqual(tuple(hmem.shape), (M_, B_, F_))
                if m.decoder_keys_mode == 'memory': self.assertIsNone(hseq)
                else: self.assertEqual(tuple(hseq.shape), (W_ - (1 if m.history_only else 0), B_, F_))
            if cell in ('B', 'D', 'E', 'F'): self.assertTrue(torch.equal(m.dec_query.detach(), torch.zeros(1, 1, F_, dtype=torch.float64)))

    def test_other_weights_identical_across_cells(self):
        ref = dict(build('A').named_parameters())
        for cell in 'BCDEF':
            for n, p in build(cell).named_parameters():
                if n in ref: self.assertTrue(torch.equal(p, ref[n]), (cell, n))

    def test_gradient_flow(self):
        win, tgt = batch()
        for cell in CELLS:
            m = build(cell); m.train(); m.zero_grad()
            loss = ((m(win, tgt) - tgt) ** 2).mean(); loss.backward()
            g = {n: (p.grad is not None and float(p.grad.norm()) > 0) for n, p in m.named_parameters()}
            if cell != 'F':
                self.assertTrue(g['mem_tokens'], cell); self.assertTrue(all(v for n, v in g.items() if n.startswith('mem_layer')), cell)
            if m.decoder_keys_mode == 'memory':
                self.assertTrue(all(not v for n, v in g.items() if n.startswith('seq_layer')), cell)    # sequence stream not in the graph
            else:
                self.assertTrue(all(v for n, v in g.items() if n.startswith('seq_layer')), cell)
            if cell in ('B', 'D', 'E', 'F'): self.assertTrue(g['dec_query'], cell)
            # with a ZERO-initialised learned query the decoder self-attention in_proj weight has an exactly zero gradient at step 0
            # (its input is zero); every other decoder parameter must receive gradient, and after one update the query is non-zero
            # and that weight receives gradient too
            skip = ('transformer_decoder.layers.0.self_attn.in_proj_weight', 'transformer_decoder.layers.0.self_attn.out_proj.weight') if m.decoder_query_mode == 'learned' else ()   # zero input and zero in_proj bias at init => V = 0
            self.assertTrue(all(v for n, v in g.items() if n.startswith('transformer_decoder') and n not in skip), cell)
            if m.decoder_query_mode == 'learned':
                with torch.no_grad(): m.dec_query -= 1e-2 * m.dec_query.grad
                m.zero_grad(); ((m(win, tgt) - tgt) ** 2).mean().backward()
                for n_ in skip: self.assertGreater(float(dict(m.named_parameters())[n_].grad.norm()), 0, (cell, n_))

    def test_target_leakage(self):
        """Perturb (1) the last row of the window, (2) the decoder tgt argument. A-D may change; E, F must not."""
        win, tgt = batch()
        for cell in CELLS:
            m = build(cell); m.eval()
            with torch.no_grad():
                base = m(win, tgt)
                w2 = win.clone(); w2[-1] += 0.3; out_w = m(w2, w2[-1:].clone())          # consistent window + target perturbation
                w3 = win.clone(); w3[-1] += 0.3; out_w_only = m(w3, tgt)                 # window row perturbed, tgt argument unchanged
                out_t = m(win, tgt + 0.3)                                                # tgt argument perturbed only
                hist = win.clone(); hist[:-1] += 0.3; out_h = m(hist, tgt)               # history perturbed: must change everywhere
            if cell in ('E', 'F'):
                self.assertTrue(torch.equal(base, out_w), cell); self.assertTrue(torch.equal(base, out_w_only), cell); self.assertTrue(torch.equal(base, out_t), cell)
            else:
                self.assertFalse(torch.equal(base, out_w), cell)
            if cell in ('B', 'D'): self.assertTrue(torch.equal(base, out_t), cell)         # learned query ignores the tgt argument
            if cell in ('A', 'C'): self.assertFalse(torch.equal(base, out_t), cell)
            self.assertFalse(torch.equal(base, out_h), cell)

    def test_checkpoint_compatibility(self):
        a = build('A'); sd = a.state_dict()
        b = build('A'); b.load_state_dict(sd, strict=True)                            # original checkpoints load unchanged
        for cell in 'BCDEF':
            m = build(cell); missing, unexpected = m.load_state_dict(sd, strict=False)
            self.assertEqual(set(unexpected), {k for k in sd if k.startswith(('mem_tokens', 'mem_layer'))} if cell == 'F' else set(), cell)
            self.assertEqual(set(missing), {'dec_query'} if cell in ('B', 'D', 'E', 'F') else set(), cell)

    def test_numerical_stability(self):
        win, tgt = batch()
        for cell in CELLS:
            m = build(cell); m.train()
            for _ in range(3):
                loss = ((m(win, tgt) - tgt) ** 2).mean(); loss.backward()
                self.assertTrue(torch.isfinite(loss), cell); self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), cell)
                with torch.no_grad():
                    for p in m.parameters():
                        if p.grad is not None: p -= 1e-3 * p.grad; p.grad = None

    def test_diversity_hook_sees_forward_hmem(self):
        win, tgt = batch()
        for cell in 'ABCDE':
            m = build(cell); m.train(); m.collect_memory_reprs = True; m._memory_reprs = []
            m(win, tgt); self.assertEqual(len(m._memory_reprs), 1, cell); self.assertEqual(tuple(m._memory_reprs[0].shape), (M_, B_, F_), cell)


if __name__ == '__main__': unittest.main()
