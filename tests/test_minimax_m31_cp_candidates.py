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

    def test_prefill_cp_selects_packed_segment_and_reuses_slots(self):
        namespace = "hcu_sglang.layers.attention"

        class Query:
            def __init__(self, rows, heads, dim=128):
                self.rows = rows
                self.shape = (len(rows), heads, dim)

            def __getitem__(self, key):
                return Query(self.rows[key], *self.shape[1:])

        @contextlib.contextmanager
        def metadata_context(meta, segment, lengths, query_lengths):
            yield segment

        native = Mock(side_effect=lambda **kwargs: (
            None, SimpleNamespace(flatten=lambda _: (kwargs["q"].rows, kwargs["prefix_lens"])),
        ))
        stream = SimpleNamespace(cuda_stream=11)
        cuda = SimpleNamespace(is_current_stream_capturing=Mock(), current_stream=lambda _: stream)
        torch = SimpleNamespace(cuda=cuda, cat=lambda parts, dim: tuple(parts))
        available = Mock()
        store = Mock()
        cp = load_module(f"{namespace}.minimax_m31_cp", {
            "torch": torch,
            f"{namespace}.minimax_m31_cp_kv": SimpleNamespace(try_store_cp_kv=store),
            f"{namespace}.minimax_m31_flashmla": SimpleNamespace(flash_mla_gfx938_available=available),
            f"{namespace}.minimax_m31_sparse": SimpleNamespace(cp_prefill_metadata=metadata_context),
        })
        strategy = SimpleNamespace(name="zigzag")
        modules = {
            "sglang.srt.layers.cp.utils": SimpleNamespace(
                get_cp_strategy=lambda: strategy, is_cp_active=lambda _: True,
            ),
            "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse": SimpleNamespace(
                minimax_sparse_prefill=native,
            ),
        }
        kv = SimpleNamespace(shape=(2, 4, 128))
        original = Mock()
        kv_pool = SimpleNamespace(
            get_kv_buffer=lambda _: (kv, kv), get_index_k_buffer=lambda _: kv,
            set_fused_kv_index_buffer=Mock(),
        )
        layer = SimpleNamespace(layer_id=3, **{
            f"{name}_scale_float": 1.0 for name in ("q", "k", "v", "idx_q", "idx_k", "idx_v")
        })
        cases = (
            ("packed", {}, (2,)),
            ("opt-in disabled", {"enabled": False}, (0, 1)),
            ("no native KV", {"native_kv": False}, (0, 1)),
            ("score type", {"backend": {"score_type": "sum"}}, (0, 1)),
            ("page size", {"backend": {"page_size": 64}}, (0, 1)),
            ("block size", {"backend": {"block_size_k": 64}}, (0, 1)),
            ("topk", {"backend": {"topk_blocks": 8}}, (0, 1)),
            ("Q shape", {"q_heads": 32}, (0, 1)),
            ("index shape", {"idx_heads": 2}, (0, 1)),
            ("capture", {"capture": True}, (0, 1)),
            ("no gfx938 kernel", {"gfx938": False}, (0, 1)),
            ("empty second segment", {"next_rows": 0}, (0,)),
            ("scratch limit", {"next_rows": 262142, "next_kv_len": 262144}, (0, 1)),
        )
        cp._gather_kv = Mock(return_value=(kv, kv, kv, None))
        with patch.dict(sys.modules, modules):
            for name, overrides, expected_segments in cases:
                with self.subTest(case=name):
                    native.reset_mock()
                    kv_pool.set_fused_kv_index_buffer.reset_mock()
                    store.return_value = (kv, kv, kv) if overrides.get("native_kv", True) else None
                    available.return_value = overrides.get("gfx938", True)
                    cuda.is_current_stream_capturing.return_value = overrides.get("capture", False)
                    q_next = overrides.get("next_rows", 3)
                    total = 2 + q_next
                    meta = SimpleNamespace(total_q_prev_tokens=2, total_q_next_tokens=q_next)
                    for segment, q_rows, k_rows in (
                        ("prev", 2, 128), ("next", q_next, overrides.get("next_kv_len", 256)),
                    ):
                        setattr(meta, f"actual_seq_q_{segment}_list", [q_rows])
                        setattr(meta, f"kv_len_{segment}_list", [k_rows])
                    for segment in ("prev", "next", "combined"):
                        for tensor in ("cu_seqlens_q", "kv_len", "actual_seq_q"):
                            setattr(meta, f"{tensor}_{segment}_tensor", object())
                    repeated_slots = (7, 9, 7, 9)
                    slot_ids = SimpleNamespace(
                        is_cuda=True, device="cuda:0", data_ptr=lambda: 64,
                        repeat=Mock(side_effect=lambda factor: (7, 9) * factor),
                    )
                    batch = SimpleNamespace(
                        attn_cp_metadata=meta, req_pool_indices=slot_ids,
                        out_cache_loc=SimpleNamespace(numel=lambda: 2),
                    )
                    config = dict(
                        score_type="max", page_size=128, block_size_k=128, topk_blocks=16,
                    )
                    config.update(overrides.get("backend", {}))
                    backend = SimpleNamespace(
                        **config, is_npu=False, _loc_mapping=None, disable_value_layer_ids={3},
                        block_size_q=1, init_blocks=1, local_blocks=1, kv_pool=kv_pool,
                        req_to_token=object(),
                    )
                    q = Query(range(total), overrides.get("q_heads", 64))
                    idx = Query(range(total), overrides.get("idx_heads", 4))

                    args = (original, backend, q, kv, kv, layer, batch)
                    kwargs = dict(idx_q=idx, idx_k=kv, idx_v=None,
                                  pack_prefill_segments=overrides.get("enabled", True))
                    actual = cp._prefill_cp(*args, **kwargs)
                    calls = [call.kwargs for call in native.call_args_list]
                    self.assertEqual(tuple(call["prefix_lens"] for call in calls), expected_segments)
                    self.assertIsNone(actual[0])
                    if expected_segments == (2,):
                        slot_ids.repeat.assert_called_once_with(2)
                        self.assertEqual(calls[0]["slot_ids"], repeated_slots)
                        self.assertIs(calls[0]["cu_seqlens"], meta.cu_seqlens_q_combined_tensor)
                        self.assertIs(calls[0]["seq_lens"], meta.kv_len_combined_tensor)
                        self.assertEqual(calls[0]["seqlens_cpu"], [2, q_next])
                        self.assertEqual(actual[1], (range(total), 2))
                        self.assertEqual(cp._prefill_cp(*args, **kwargs), actual)
                        slot_ids.repeat.assert_called_once_with(2)
                        self.assertIs(native.call_args.kwargs["slot_ids"], calls[0]["slot_ids"])
                    else:
                        slot_ids.repeat.assert_not_called()
                        expected_rows = (range(2), range(2, total))
                        for call, segment in zip(calls, expected_segments):
                            self.assertIs(call["slot_ids"], slot_ids)
                            self.assertEqual(call["q"].rows, expected_rows[segment])
                        self.assertEqual(actual[1], tuple(
                            (expected_rows[segment], segment) for segment in expected_segments
                        ))
        original.assert_not_called()

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
