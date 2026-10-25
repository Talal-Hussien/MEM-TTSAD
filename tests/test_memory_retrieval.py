"""Input-conditioned memory retrieval for MemCtrl_TTSAD_Basic: default-off equivalence, shapes, retrieval variation across
windows, gradient flow (memory tokens and retrieval projections), checkpoint compatibility, parameter count, runtime disable."""
import os
import unittest

import torch

import memttsad.models as SM

F_, M_, W_, B_ = 38, 3, 10, 16
ENV = ['MEMTRANAD_MEMORY_RETRIEVAL', 'MEMTRANAD_RETRIEVAL_DIM', 'MEMTRANAD_RETRIEVAL_CONTEXT', 'MEMTRANAD_DECODER_QUERY', 'MEMTRANAD_DECODER_KEYS', 'MEMTRANAD_HISTORY_ONLY', 'MEMTRANAD_NO_MEMORY', 'MEMTRANAD_MEM_INIT_NORM']


def build(on, dim=16, ctx='mean', seed=1):
    for k in ENV: os.environ.pop(k, None)
    if on: os.environ.update(MEMTRANAD_MEMORY_RETRIEVAL='1', MEMTRANAD_RETRIEVAL_DIM=str(dim), MEMTRANAD_RETRIEVAL_CONTEXT=ctx)
    torch.manual_seed(seed)
    m = SM.MemCtrl_TTSAD_Basic(F_, M_).double()
    for k in ENV: os.environ.pop(k, None)
    return m


def batch(seed=0):
    g = torch.Generator().manual_seed(seed); win = torch.rand(W_, B_, F_, generator=g, dtype=torch.float64); return win, win[-1:].clone()


class TestMemoryRetrieval(unittest.TestCase):
    def test_default_off_and_init_equivalence(self):
        a, r = build(False), build(True); win, tgt = batch(); a.eval(); r.eval()
        self.assertFalse(a.memory_retrieval); self.assertFalse(hasattr(a, 'ret_q')); self.assertTrue(r.memory_retrieval)
        for n, p in a.named_parameters(): self.assertTrue(torch.equal(p, dict(r.named_parameters())[n]), n)   # pre-existing weights identical (retrieval params created last)
        with torch.no_grad(): self.assertTrue(torch.equal(a(win, tgt), r(win, tgt)))                        # zero-initialised output projection: same function at init

    def test_shapes_and_params(self):
        r = build(True); win, tgt = batch(); r.eval(); r.collect_retrieval = True
        with torch.no_grad(): out = r(win, tgt)
        self.assertEqual(tuple(out.shape), (1, B_, F_)); alpha, logits, contrib = r._retrieval
        self.assertEqual(tuple(alpha.shape), (B_, M_)); self.assertEqual(tuple(logits.shape), (B_, M_)); self.assertEqual(tuple(contrib.shape), (1, B_, F_))
        self.assertTrue(torch.allclose(alpha.sum(-1), torch.ones(B_, dtype=torch.float64)))
        extra = sum(p.numel() for n, p in r.named_parameters() if n.startswith('ret_')); self.assertEqual(extra, 2 * (F_ * 16 + 16) + F_ * F_ + F_ + F_)

    def test_retrieval_varies_across_windows_after_training_step(self):
        r = build(True); win, tgt = batch(); r.train()
        opt = torch.optim.AdamW(r.parameters(), lr=1e-2)
        for _ in range(5):
            opt.zero_grad(); ((r(win, tgt) - tgt) ** 2).mean().backward(); opt.step()
        r.eval(); r.collect_retrieval = True
        with torch.no_grad(): r(win, tgt)
        alpha = r._retrieval[0]; self.assertGreater(float(alpha.std(0).max()), 1e-4)                         # weights differ across the 16 windows
        self.assertGreater(float((alpha.max(1).values - 1 / M_).abs().max()), 1e-4)

    def test_gradient_flow(self):
        r = build(True); win, tgt = batch(); r.train(); r.zero_grad(); ((r(win, tgt) - tgt) ** 2).mean().backward()
        g = {n: (p.grad is not None and float(p.grad.norm()) > 0) for n, p in r.named_parameters()}
        self.assertTrue(g['mem_tokens']); self.assertTrue(g['ret_out.weight']); self.assertTrue(g['ret_out.bias'])   # step 0: output projection receives gradient
        with torch.no_grad():                                                                                     # after one step ret_out != 0 => gradient reaches gate, projections and the slots through r
            for n, p in r.named_parameters():
                if p.grad is not None: p -= 1e-2 * p.grad
        r.zero_grad(); ((r(win, tgt) - tgt) ** 2).mean().backward()
        g = {n: (p.grad is not None and float(p.grad.norm()) > 0) for n, p in r.named_parameters()}
        for n in ('ret_q.weight', 'ret_k.weight', 'ret_gate', 'mem_tokens'): self.assertTrue(g[n], n)

    def test_runtime_disable_and_checkpoints(self):
        r = build(True); win, tgt = batch(); r.eval()
        with torch.no_grad():
            r.ret_out.weight.copy_(torch.randn_like(r.ret_out.weight) * 0.05)                 # random: a constant fill cancels on LayerNorm outputs, which sum to zero
            on = r(win, tgt); r.retrieval_active = False; off = r(win, tgt); r.retrieval_active = True
            a = build(False); a.eval(); a.load_state_dict({k: v for k, v in r.state_dict().items() if not k.startswith('ret_')}, strict=True)
            self.assertTrue(torch.equal(a(win, tgt), off))                                                       # disabled retrieval == original model with the same weights
            self.assertFalse(torch.equal(on, off))
        missing, unexpected = build(True).load_state_dict(build(False).state_dict(), strict=False)             # original checkpoint loads into the retrieval model
        self.assertEqual(set(unexpected), set()); self.assertEqual(set(missing), {'ret_q.weight', 'ret_q.bias', 'ret_k.weight', 'ret_k.bias', 'ret_out.weight', 'ret_out.bias', 'ret_gate'})

    def test_last_context(self):
        r = build(True, ctx='last'); win, tgt = batch(); r.eval(); r.collect_retrieval = True
        with torch.no_grad(): r(win, tgt)
        self.assertEqual(tuple(r._retrieval[0].shape), (B_, M_))


if __name__ == '__main__': unittest.main()
