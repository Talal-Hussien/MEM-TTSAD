import math
import os
import unittest

import torch

import memttsad.models as SM


class TestMemoryInitNorm(unittest.TestCase):
    def _build(self, env=None):
        os.environ.pop('MEMTRANAD_MEM_INIT_NORM', None)
        if env is not None: os.environ['MEMTRANAD_MEM_INIT_NORM'] = str(env)
        torch.manual_seed(1)
        m = SM.MemCtrl_TTSAD_Basic(38, 3)
        os.environ.pop('MEMTRANAD_MEM_INIT_NORM', None)
        return m

    def test_disabled_is_original(self):
        m = self._build(None)
        torch.manual_seed(1); ref = torch.randn(3, 1, 38) * 0.02
        self.assertTrue(torch.equal(m.mem_tokens.detach(), ref)); self.assertIsNone(m.mem_init_norm)

    def test_enabled_same_direction_scaled(self):
        a = self._build(None); b = self._build(12.5)
        ra = a.mem_tokens.detach() / 0.02; rb = b.mem_tokens.detach() / (12.5 / math.sqrt(38))
        self.assertTrue(torch.allclose(ra, rb)); self.assertEqual(b.mem_init_norm, 12.5)
        self.assertAlmostEqual(float(b.mem_tokens.detach()[:, 0, :].norm(dim=-1).mean()), 12.5, delta=12.5 * 0.4)
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):   # every other parameter identical (same RNG stream)
            if na != 'mem_tokens': self.assertTrue(torch.equal(pa, pb), na)

    def test_direct_model_unaffected(self):
        os.environ['MEMTRANAD_MEM_INIT_NORM'] = '12.5'
        torch.manual_seed(1); m = SM.Mem_TTSAD_Basic(38, 3)
        self.assertFalse(hasattr(m, 'mem_init_norm')); self.assertLess(float(m.mem_tokens.detach().norm(dim=-1).mean()), 1.0)
        os.environ.pop('MEMTRANAD_MEM_INIT_NORM', None)


if __name__ == '__main__': unittest.main()
