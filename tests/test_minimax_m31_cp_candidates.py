# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hcu_sglang.config import MINIMAX_CP_PACK_PREFILL_SEGMENTS_ENV

ROOT = Path(__file__).resolve().parent.parent


def load_module(name, modules):
    path = ROOT.joinpath(*name.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestMiniMaxCPCandidates(unittest.TestCase):
    def test_packed_shape_order_and_fallback(self):
        namespace = "hcu_sglang.layers.attention"
        cp = load_module(f"{namespace}.minimax_m31_cp", {
            "torch": SimpleNamespace(),
            f"{namespace}.minimax_m31_cp_kv": SimpleNamespace(try_store_cp_kv=None),
            f"{namespace}.minimax_m31_flashmla": SimpleNamespace(flash_mla_gfx938_available=None),
            f"{namespace}.minimax_m31_sparse": SimpleNamespace(cp_prefill_metadata=None),
        })
        metadata = SimpleNamespace(
            cu_seqlens_q_combined_tensor=object(),
            kv_len_combined_tensor=object(),
            actual_seq_q_combined_tensor=object(),
        )
        first = (0, 5, None, None, None, [2, 3], [128, 256])
        second = (5, 9, None, None, None, [1, 3], [384, 512])
        self.assertEqual(cp._pack_prefill_pieces(metadata, (first, second)), ((
            0, 9, metadata.cu_seqlens_q_combined_tensor,
            metadata.kv_len_combined_tensor, metadata.actual_seq_q_combined_tensor,
            [2, 3, 1, 3], [128, 256, 384, 512],
        ),))
        self.assertIsNone(cp._pack_prefill_pieces(metadata, (
            first, (5, 8, None, None, None, [0, 3], [384, 512]),
        )))
        self.assertIsNone(cp._pack_prefill_pieces(metadata, (
            first, (5, 262144, None, None, None, [1, 262138], [512, 262144]),
        )))
        registry = Mock()
        stubs = {
            "sglang.srt.plugins.hook_registry": SimpleNamespace(
                HookRegistry=registry, HookType=SimpleNamespace(AROUND="around"),
            ),
        }
        with patch.dict(sys.modules, stubs):
            for flag, expected in (("0", False), ("1", True)):
                with patch.dict("os.environ", {MINIMAX_CP_PACK_PREFILL_SEGMENTS_ENV: flag}):
                    cp.register_hooks()
                target, hook, kind = registry.register.call_args.args
                self.assertTrue(target.endswith("MiniMaxSparseAttnBackend.forward_extend"))
                self.assertIs(hook.func, cp._prefill_cp)
                self.assertIs(hook.keywords["pack_prefill_segments"], expected)
                self.assertEqual(kind, "around")

    def test_early_projection_order_and_symmetric_memory_fallback(self):
        namespace = "hcu_sglang.layers.moe"
        events = []

        class SharedOutput:
            def record_stream(self, stream):
                events.append("record")

            def __radd__(self, other):
                return other + 1

        main = SimpleNamespace(wait_stream=lambda _: events.append("main_wait"))
        side = SimpleNamespace(wait_stream=lambda _: events.append("side_wait"))
        torch = SimpleNamespace(
            version=SimpleNamespace(hip="test"),
            cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: False,
                current_stream=lambda _: main,
                stream=lambda _: contextlib.nullcontext(),
            ),
        )
        hooks = []
        dispatcher = SimpleNamespace(
            deepep_mode=SimpleNamespace(is_normal=lambda: True),
            _normal_dispatcher=SimpleNamespace(async_finish=True),
        )

        def register(hook):
            hooks.append(hook)
            return SimpleNamespace(remove=lambda: hooks.remove(hook))

        def experts(*args):
            events.append("routed")
            for hook in hooks:
                hook(dispatcher)
            return 2

        def gate(hidden):
            events.append("gate")
            return hidden, None

        def down(*args):
            events.append("down")
            return SharedOutput()

        experts.dispatcher = dispatcher
        experts._minimax_w4a8_hipc = True
        dispatcher.register_deepep_normal_combine_hook = register
        moe = SimpleNamespace(
            experts=experts, shared_experts=SimpleNamespace(gate_up_proj=gate),
            num_fused_shared_experts=0, _compute_router_logits=lambda x: x,
            topk=lambda x, y: y,
        )
        overlap = load_module(f"{namespace}.minimax_m31_prefill_overlap", {
            "torch": torch,
            f"{namespace}.minimax_m31_shared_overlap": SimpleNamespace(
                resolve_single_dispatcher=lambda value: value,
            ),
        })
        overlap._shared_int8_kernel = Mock(return_value=object())
        overlap._shared_down = down
        symmetric = Mock()
        stubs = {
            "sglang.srt.layers.moe.utils": SimpleNamespace(
                is_sbo_enabled=lambda: False, is_tbo_enabled=lambda: False,
            ),
            "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
                ForwardMode=SimpleNamespace(EXTEND=1, SPLIT_PREFILL=2),
            ),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_forward=lambda: SimpleNamespace(sp_active=False),
                get_parallel=lambda: SimpleNamespace(attn_cp_size=8),
                get_stream=lambda _: side,
            ),
            "sglang.srt.distributed.device_communicators.pynccl_allocator":
                SimpleNamespace(is_symmetric_memory_enabled=symmetric),
        }
        early = ["side_wait", "gate", "down", "routed", "main_wait", "record"]
        split = ["side_wait", "gate", "routed", "side_wait", "down", "main_wait"]
        with patch.dict(sys.modules, stubs):
            for method, use_symmetric, expected in (
                ("early", False, early), ("split", False, split), ("early", True, split),
            ):
                events.clear()
                symmetric.return_value = use_symmetric
                original = Mock()
                actual = overlap._forward_deepep(
                    original, moe, SimpleNamespace(shape=(1,), device="test"),
                    SimpleNamespace(forward_mode=1), method=method, use_lightop=True,
                )
                self.assertEqual(actual, 3)
                self.assertEqual(events, expected)
                self.assertFalse(hooks)
                original.assert_not_called()
