import sys
import types
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.multimodal_gen import minimax_h3 as h3

TARGET = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils"

class VaeRopeTests(unittest.TestCase):
    def call(self, q, k, native, hcu=True):
        module = types.ModuleType(TARGET)
        module.native_rope_cache = lambda *args: native
        seen = []
        def apply(original, a, b, cache, positions):
            seen.append((a.shape, b.shape, cache, positions))
            a.add_(2)
            b.add_(3)
        with patch.dict(sys.modules, {TARGET: module}), patch.object(h3, "is_hcu", return_value=hcu), patch.object(h3, "around_apply_rope_qk", side_effect=apply):
            result = h3.around_vae_apply_rope_qk(lambda *args: "fallback", q, k, ())
        return result, seen

    def test_batch_major_native_cache_and_noncontiguous_inputs(self):
        q = torch.zeros(2, 5, 3, 128).transpose(0, 1)
        k = torch.ones_like(q)
        cache, positions = torch.ones(2, 96), torch.arange(10)
        (a, b), seen = self.call(q, k, (cache, positions))
        self.assertEqual(a.shape, q.shape)
        self.assertTrue(a.is_contiguous())
        self.assertTrue(torch.equal(a, torch.full_like(a, 2)))
        self.assertTrue(torch.equal(b, torch.full_like(b, 4)))
        self.assertTrue(torch.equal(q, torch.zeros_like(q)))
        self.assertEqual(seen[0][:2], (torch.Size((10, 3, 128)), torch.Size((10, 3, 128))))
        self.assertIs(seen[0][2], cache)
        self.assertIs(seen[0][3], positions)

    def test_video_decoder_head64_rotary48(self):
        q = torch.zeros(1, 7, 32, 64)
        k = torch.ones_like(q)
        (a, b), seen = self.call(q, k, (torch.ones(7, 48), torch.arange(7)))
        self.assertEqual(seen[0][:2], (torch.Size((7, 32, 64)), torch.Size((7, 32, 64))))
        self.assertTrue(torch.equal(a, torch.full_like(a, 2)))
        self.assertTrue(torch.equal(b, torch.full_like(b, 4)))

    def test_official_ineligible_cache_retains_original(self):
        result, seen = self.call(torch.zeros(1, 2, 3, 128), torch.zeros(1, 2, 3, 128), None)
        self.assertEqual(result, "fallback")
        self.assertFalse(seen)

    def test_other_geometry_retains_original(self):
        for head, rotary in ((64, 32), (128, 64)):
            q = torch.zeros(1, 2, 3, head)
            result, seen = self.call(q, q.clone(), (torch.ones(2, rotary), torch.arange(2)))
            self.assertEqual(result, "fallback")
            self.assertFalse(seen)

    def test_non_hcu_retains_original(self):
        result, seen = self.call(None, None, None, hcu=False)
        self.assertEqual(result, "fallback")
        self.assertFalse(seen)
