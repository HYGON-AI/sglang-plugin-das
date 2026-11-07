# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Adapter (adapter-das gdn_aiter / upstream fla triton) owns these faces now:
# chunk fwd_h + packed decode route through HCU_LINEAR_ATTN_GDN, chunk_o stays
# upstream, gated layernorm stays upstream. target_verify was part of the same
# REPLACE family and was disabled with it in the stacked combo validation.
REMOVED_TARGETS = (
    "sglang.kernels.ops.attention.fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h",
    "sglang.kernels.ops.attention.fla.chunk_o.chunk_fwd_o",
    "sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode",
    "sglang.kernels.ops.attention.fla.layernorm_gated._layer_norm_fwd",
    "sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify",
)


class FlaGdnHooksRemovedTests(unittest.TestCase):
    def test_plugin_entry_does_not_register_the_flad_gdn_hooks(self):
        text = (ROOT / "hcu_sglang/__init__.py").read_text()
        self.assertNotIn("linear_attention", text)
        self.assertNotIn("layernorm_gated", text)

    def test_declared_target_contract_no_longer_lists_them(self):
        declared = json.loads(
            (ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text()
        )["expected_hook_targets"]
        for target in REMOVED_TARGETS:
            self.assertNotIn(target, declared)

    def test_l2norm_hooks_remain_declared(self):
        declared = json.loads(
            (ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text()
        )["expected_hook_targets"]
        for kept in (
            "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
            "sglang.kernels.ops.attention.fla.chunk.l2norm_fwd",
            "sglang.kernels.ops.attention.fla.kda.l2norm_fwd",
        ):
            self.assertIn(kept, declared)


if __name__ == "__main__":
    unittest.main()
