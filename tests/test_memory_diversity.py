"""Unit tests for the memory-slot diversity regulariser (run: python -m unittest tests.test_memory_diversity -v).

Covers: loss values and the m=1 edge case; gradient flow to the learnable slots (raw and updated paths) for all six
memory classes; the recorded Hmem is exactly the decoder input; the disabled path is numerically identical; no new
state_dict keys (old checkpoints stay loadable); config precedence and checkpoint suffix.
"""
import os
import unittest

import torch

import memttsad.models as M
from memttsad.regularization import cosine_sq_diversity, mean_abs_cosine, memory_parameter, DiversityConfig, diversity_terms

FEATS, MEM, W, B = 6, 3, 10, 4
FULL = ['Mem_TTSAD', 'MemCtrl_TTSAD', 'MemBottleneck_TTSAD']
BASIC = ['Mem_TTSAD_Basic', 'MemCtrl_TTSAD_Basic', 'MemBottleneck_TTSAD_Basic']


def make(name, m=MEM, seed=0):
    torch.manual_seed(seed)
    return getattr(M, name)(FEATS, n_memory=m).double()


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    window = torch.rand(W, B, FEATS, generator=g, dtype=torch.float64)
    return window, window[-1:].clone()


def brute_force(Mat):
    m = Mat.shape[0]; tot, cnt = 0.0, 0
    for i in range(m):
        for j in range(i + 1, m):
            c = torch.dot(Mat[i], Mat[j]) / (Mat[i].norm() * Mat[j].norm()); tot += float(c) ** 2; cnt += 1
    return tot / cnt


class TestLoss(unittest.TestCase):
    def test_identical_rows_give_one_and_orthogonal_give_zero(self):
        v = torch.randn(5, dtype=torch.float64)
        self.assertAlmostEqual(cosine_sq_diversity(torch.stack([v, 2 * v, -3 * v])).item(), 1.0, places=12)
        self.assertAlmostEqual(cosine_sq_diversity(torch.eye(4, dtype=torch.float64)).item(), 0.0, places=12)

    def test_matches_formula_and_batched_mean(self):
        Mat = torch.randn(4, 7, dtype=torch.float64)
        self.assertAlmostEqual(cosine_sq_diversity(Mat).item(), brute_force(Mat), places=12)
        Mb = torch.randn(4, 3, 7, dtype=torch.float64)
        per = [brute_force(Mb[:, b]) for b in range(3)]
        self.assertAlmostEqual(cosine_sq_diversity(Mb).item(), sum(per) / 3, places=12)
        self.assertTrue(0 <= mean_abs_cosine(Mb).item() <= 1)

    def test_single_slot_is_differentiable_zero(self):
        p = torch.nn.Parameter(torch.randn(1, 1, 5, dtype=torch.float64))
        l = cosine_sq_diversity(p[:, 0, :]); l.backward()
        self.assertEqual(l.item(), 0.0); self.assertTrue(torch.isfinite(p.grad).all()); self.assertEqual(p.grad.abs().sum().item(), 0.0)


class TestModels(unittest.TestCase):
    def test_memory_parameter_found(self):
        for name in FULL + BASIC:
            p = memory_parameter(make(name)); self.assertIsNotNone(p, name); self.assertEqual(tuple(p.shape[:2]), (MEM, 1), name)
        self.assertIsNone(memory_parameter(M.TranAD_Basic(FEATS).double()))

    def test_recorded_hmem_is_decoder_input_and_phase_count(self):
        window, elem = batch()
        for name in FULL + BASIC:
            model = make(name).eval(); model.collect_memory_reprs = True; model._memory_reprs = []
            with torch.no_grad():
                model(window, elem)
                reprs = model._memory_reprs
                self.assertEqual(len(reprs), 2 if name in FULL else 1, name)
                for h in reprs: self.assertEqual(tuple(h.shape[:2]), (MEM, B), name)
                # recompute the same quantity the decoder reads (phase 1 / single phase) and compare
                if name == 'Mem_TTSAD':
                    mem = model.mem_token.expand(-1, B, -1); swm = torch.cat((mem, window), 0)
                    _, enc = model.encode(swm, torch.zeros_like(swm), elem); ref = enc[:MEM]
                elif name == 'Mem_TTSAD_Basic':
                    z = model.pos_encoder(window * (FEATS ** 0.5)); ref = model.transformer_encoder(torch.cat((model.mem_tokens.expand(-1, B, -1), z), 0))[:MEM]
                elif name in FULL:
                    ref, _ = model._encode_parts(window, torch.zeros_like(window))
                else:
                    ref, _ = model._encode_parts(window)
                self.assertTrue(torch.allclose(reprs[0], ref, atol=0, rtol=0), name)
                if name in FULL:   # phase 2: recompute Hmem with the conditioning the model actually used
                    out = model(window, elem); x1 = out[0]
                    c2 = (x1 - window) ** 2
                    if name == 'Mem_TTSAD':
                        mem = model.mem_token.expand(-1, B, -1); swm = torch.cat((mem, window), 0)
                        c2m = (x1 - swm) ** 2
                        _, enc2 = model.encode(swm, c2m, elem); ref2 = enc2[:MEM]
                    else:
                        ref2, _ = model._encode_parts(window, c2)
                    self.assertTrue(torch.allclose(reprs[1], ref2, atol=0, rtol=0), name)
                    self.assertFalse(torch.equal(reprs[0], reprs[1]), name)   # phases differ

    def test_disabled_path_numerically_identical(self):
        window, elem = batch()
        for name in FULL + BASIC:
            model = make(name).train()
            torch.manual_seed(7); out_a = model(window, elem)
            model.collect_memory_reprs = True; model._memory_reprs = []
            torch.manual_seed(7); out_b = model(window, elem)
            oa = out_a if isinstance(out_a, tuple) else (out_a,); ob = out_b if isinstance(out_b, tuple) else (out_b,)
            for x, y in zip(oa, ob): self.assertTrue(torch.equal(x, y), name)
            self.assertFalse(any('collect' in k or '_memory_reprs' in k for k in model.state_dict()), name)
            fresh = make(name); fresh.load_state_dict(model.state_dict(), strict=True)   # no new keys -> old checkpoints load

    def test_gradients_reach_memory_for_every_mode(self):
        window, elem = batch()
        for name in FULL + BASIC:
            for mode in ('raw', 'updated', 'both'):
                cfg = DiversityConfig(mode, 0.1); model = make(name).train(); model.collect_memory_reprs = cfg.use_updated; model._memory_reprs = []
                z = model(window, elem); l_raw, l_upd = diversity_terms(model, cfg)
                loss = cfg.lambda_raw * l_raw + cfg.lambda_updated * l_upd      # diversity term alone
                self.assertTrue(torch.isfinite(loss)); self.assertGreater(loss.item(), 0.0, (name, mode))
                model.zero_grad(); loss.backward()
                g = memory_parameter(model).grad
                self.assertIsNotNone(g, (name, mode)); self.assertTrue(torch.isfinite(g).all()); self.assertGreater(g.abs().sum().item(), 0.0, (name, mode))
                if mode == 'updated':   # the updated loss must also train the memory-update path, not only the slots
                    enc = [p for n, p in model.named_parameters() if ('mem_layer' in n or 'transformer_encoder' in n) and p.grad is not None]
                    self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in enc), name)

    def test_m1_models_give_zero_loss_without_nan(self):
        window, elem = batch()
        for name in FULL + BASIC:
            cfg = DiversityConfig('both', 0.1); model = make(name, m=1).train(); model.collect_memory_reprs = True; model._memory_reprs = []
            model(window, elem); l_raw, l_upd = diversity_terms(model, cfg)
            self.assertEqual(l_raw.item(), 0.0); self.assertEqual(l_upd.item(), 0.0)
            (l_raw + l_upd).backward(); self.assertTrue(torch.isfinite(memory_parameter(model).grad).all())

    def test_raw_mode_records_nothing_and_no_parameters_added(self):
        window, elem = batch()
        for name in FULL + BASIC:
            model = make(name); n0 = sum(p.numel() for p in model.parameters())
            model.collect_memory_reprs = False; model._memory_reprs = []; model(window, elem)
            self.assertEqual(model._memory_reprs, [], name)
            self.assertEqual(sum(p.numel() for p in model.parameters()), n0, name)

    def test_updated_without_recording_raises(self):
        model = make('MemCtrl_TTSAD_Basic'); model._memory_reprs = []
        with self.assertRaises(RuntimeError): diversity_terms(model, DiversityConfig('updated', 0.1))
        with self.assertRaises(ValueError): diversity_terms(M.TranAD_Basic(FEATS).double(), DiversityConfig('raw', 0.1))


class TestConfig(unittest.TestCase):
    def test_defaults_and_precedence(self):
        for k in ('MEMTRANAD_DIVERSITY', 'MEMTRANAD_DIV_LAMBDA', 'MEMTRANAD_DIV_LAMBDA_RAW', 'MEMTRANAD_DIV_LAMBDA_UPD'): os.environ.pop(k, None)
        c = DiversityConfig(); self.assertFalse(c.enabled); self.assertEqual(c.ckpt_suffix(), ''); self.assertEqual((c.lambda_raw, c.lambda_updated), (0.0, 0.0))
        c = DiversityConfig('both', 0.01); self.assertEqual((c.lambda_raw, c.lambda_updated), (0.01, 0.01)); self.assertEqual(c.ckpt_suffix(), '_divboth_r0.01_u0.01')
        c = DiversityConfig('raw', 0.01); self.assertEqual((c.lambda_raw, c.lambda_updated), (0.01, 0.0))
        c = DiversityConfig('both', 0.01, lam_upd=0.1); self.assertEqual((c.lambda_raw, c.lambda_updated), (0.01, 0.1))
        os.environ['MEMTRANAD_DIVERSITY'] = 'updated'; os.environ['MEMTRANAD_DIV_LAMBDA'] = '0.5'
        c = DiversityConfig(); self.assertEqual(c.mode, 'updated'); self.assertEqual(c.lambda_updated, 0.5); self.assertEqual(c.lambda_raw, 0.0)
        c = DiversityConfig('none'); self.assertFalse(c.enabled)          # explicit CLI beats env
        for k in ('MEMTRANAD_DIVERSITY', 'MEMTRANAD_DIV_LAMBDA'): os.environ.pop(k, None)
        with self.assertRaises(ValueError): DiversityConfig('orthogonal')


if __name__ == '__main__':
    unittest.main()
