from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import hcu_sglang
from sglang.srt.plugins.hook_registry import HookRegistry


class IndexTopKShareTests(unittest.TestCase):
    def test_public_class_and_seed_helper_apply_source_conversion(self):
        hcu_sglang.register()
        from hcu_sglang.layers.attention import index_topk_share as owned

        HookRegistry.apply_hooks()
        from sglang.srt.layers.attention import index_topk_share as public
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mha,
        )

        self.assertIs(public.IndexTopKShareState, owned.IndexTopKShareState)
        self.assertIs(public.write_mtp_seed, owned.write_mtp_seed)
        seed = torch.zeros(2, 3, dtype=torch.int64)
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mode = SimpleNamespace(is_extend=lambda **kwargs: True)
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                dsa_seed_topk_select=None,
                dsa_seed_topk_capture=seed,
                dsa_topk_indices=None,
            ),
            forward_mode=mode,
            reuse_dsa_topk_indices=False,
        )
        backend = SimpleNamespace(
            prepare_mtp_seed_indices=lambda value, mode: value.flip(0)
        )
        with patch.object(forward_mha, "resolve_attn_backend", return_value=backend):
            public.write_mtp_seed(batch, seed, topk)
            self.assertTrue(torch.equal(seed, topk.flip(0)))
            seed.zero_()
            state = public.IndexTopKShareState(batch, topk)
            state.publish()
            self.assertTrue(torch.equal(seed, topk.flip(0)))


if __name__ == "__main__":
    unittest.main()
