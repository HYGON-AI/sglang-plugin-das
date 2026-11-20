# SPDX-License-Identifier: Apache-2.0
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.moe import minimax_m31_shared_overlap as overlap
from hcu_sglang.layers.moe.minimax_m31_shared_overlap import _shared_in_receive_windows


class TestMiniMaxSharedOverlap(unittest.TestCase):
    def test_shared_gemms_run_in_receive_windows_and_restore_dispatcher(self):
        low = SimpleNamespace(return_recv_hook=False)
        hooks = {}
        calls = []

        def register(name, hook):
            self.assertTrue(low.return_recv_hook)
            hooks[name] = hook
            return SimpleNamespace(remove=lambda: hooks.pop(name))

        dispatcher = SimpleNamespace(
            _low_latency_dispatcher=low,
            register_deepep_dispatch_hook=lambda hook: register("dispatch", hook),
            register_deepep_combine_hook=lambda hook: register("combine", hook),
        )

        class Experts:
            def __call__(self, x, topk):
                calls.append("dispatch_send")
                hooks["dispatch"](dispatcher)
                calls.extend(("dispatch_recv", "combine_send"))
                hooks["combine"](dispatcher)
                calls.append("combine_recv")
                return x + topk

        def project(name, x):
            calls.append(name)
            return x + 1, None

        experts = Experts()
        experts.dispatcher = dispatcher
        moe = SimpleNamespace(
            experts=experts,
            topk=lambda x, routes: routes,
            _compute_router_logits=lambda x: x * 2,
            shared_experts=SimpleNamespace(
                gate_up_proj=lambda x: project("shared_gate_up", x),
                _swiglu_oai_mxfp8=lambda _: None,
                act_fn=lambda x: x * 3,
                down_proj=lambda x: project("shared_down", x),
            ),
        )
        torch.testing.assert_close(_shared_in_receive_windows(moe, torch.ones(1), dispatcher), torch.tensor([10.0]))
        self.assertEqual(calls, [
            "dispatch_send", "shared_gate_up", "dispatch_recv", "combine_send",
            "shared_down", "combine_recv",
        ])
        self.assertFalse(hooks)
        self.assertFalse(low.return_recv_hook)

        def fail(_):
            raise RuntimeError("shared projection failed")

        moe.shared_experts.down_proj = fail
        with self.assertRaisesRegex(RuntimeError, "shared projection failed"):
            _shared_in_receive_windows(moe, torch.ones(1), dispatcher)
        self.assertFalse(hooks)
        self.assertFalse(low.return_recv_hook)


    def test_auto_dispatch_mode_uses_only_its_low_latency_branch(self):
        class Dispatcher:
            deepep_mode = SimpleNamespace(enable_low_latency=lambda: True)

            def __init__(self):
                self._low_latency_dispatcher = object()
                self._normal_dispatcher = object()
                self.active = self._low_latency_dispatcher

            def _get_impl(self):
                return self.active

        class Wrapper:
            def __init__(self, inners):
                self._inners = inners

        dispatcher = Dispatcher()
        moe = SimpleNamespace(
            shared_experts=object(), num_fused_shared_experts=0,
            experts=SimpleNamespace(dispatcher=Wrapper([dispatcher]), _minimax_w4a8_hipc=True),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True)
        )
        modules = {
            "sglang.srt.batch_overlap.two_batch_overlap": SimpleNamespace(
                MaybeTboDeepEPDispatcher=Wrapper,
            ),
            "sglang.srt.layers.moe.utils": SimpleNamespace(
                is_sbo_enabled=lambda: False, is_tbo_enabled=lambda: False,
            ),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_disagg=lambda: SimpleNamespace(disaggregation_mode="null"),
                get_memory=lambda: SimpleNamespace(enable_hisparse=False),
                get_parallel=lambda: SimpleNamespace(attn_dp_size=8, moe_ep_size=8),
            ),
            "hcu_sglang.layers.moe.deepep_current": SimpleNamespace(
                DeepEPDispatcher=Dispatcher
            ),
        }
        original = Mock()
        states = torch.ones((1, 1))
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"SGLANG_MINIMAX_DECODE_SHARED_OVERLAP": "1"}),
            patch.object(torch.version, "hip", "test"),
            patch.object(overlap, "_shared_in_receive_windows") as projected,
        ):
            self.assertIs(overlap._forward_deepep(original, moe, states, forward_batch),
                          projected.return_value)
            dispatcher.active = dispatcher._normal_dispatcher
            self.assertIs(overlap._forward_deepep(original, moe, states, forward_batch),
                          original.return_value)
            dispatcher.deepep_mode = SimpleNamespace(enable_low_latency=lambda: False)
            del dispatcher._low_latency_dispatcher
            self.assertIs(overlap._forward_deepep(original, moe, states, forward_batch),
                          original.return_value)
            moe.experts.dispatcher = Wrapper([dispatcher, dispatcher])
            self.assertIs(overlap._forward_deepep(original, moe, states, forward_batch),
                          original.return_value)
        projected.assert_called_once_with(moe, states, dispatcher)
        self.assertEqual(original.call_count, 3)
