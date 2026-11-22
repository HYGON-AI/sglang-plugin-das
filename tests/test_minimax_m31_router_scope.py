# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hcu_sglang.models.minimax_m31_router import _forward_moe, _target_verify_router


class TestMiniMaxRouterScope(unittest.TestCase):
    def test_extends_only_dp16_target_verify_and_restores_scope(self):
        for enabled, verify, ep, expected in (
            ("0", True, 16, False),
            ("1", False, 16, False),
            ("1", True, 8, False),
            ("1", True, 16, True),
        ):
            with self.subTest(enabled=enabled, verify=verify, ep=ep):
                runtime = SimpleNamespace(get_parallel=lambda: SimpleNamespace(attn_dp_size=16, moe_ep_size=ep))
                batch = SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: verify))
                with (
                    patch.dict("os.environ", {"SGLANG_MINIMAX_EXTEND_ROUTER_GEMV": enabled}),
                    patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}),
                    patch.object(torch.version, "hip", "test"),
                ):
                    self.assertEqual(_forward_moe(lambda *_: _target_verify_router.get(), None, None, batch), expected)
                    self.assertFalse(_target_verify_router.get())
                    def abort(*_):
                        raise RuntimeError("step failed")

                    with self.assertRaisesRegex(RuntimeError, "step failed"):
                        _forward_moe(abort, None, None, batch)
                    self.assertFalse(_target_verify_router.get())
