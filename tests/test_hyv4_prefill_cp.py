# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1] / "hcu_sglang" / "models"


def _load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(name, **values):
    module = ModuleType(name)
    module.__dict__.update(values)
    return module


def upstream_family(args, config):
    cfg = args
    if config.architectures[0] in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
        if cfg.enable_prefill_cp:
            raise ValueError("--enable-prefill-cp is not supported for HYV4")
        if cfg.dcp_size > 1:
            raise ValueError("HYV4 does not support DCP")
    return {"attn_cp_size": cfg.attn_cp_size, "page_size": 64}


class HYV4PrefillCPTests(unittest.TestCase):
    def setUp(self):
        detection = "hcu_sglang.utils.detection"
        native = "hcu_sglang.layers.attention.current_dsa.flashmla_backend"
        published = "hcu_sglang.models.hyv4_prefill_cp"
        registry = "sglang.srt.plugins.hook_registry"
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {
            detection: _module(detection, is_hcu=lambda: True),
            native: _module(native, get_flashmla_op=lambda *a, **k: self.flashmla),
            registry: _module(registry, HookRegistry=object(), HookType=object()),
            "torch": _module("torch"),
        }).start()
        self.policy = _load(published, _ROOT / "hyv4_prefill_cp.py")
        patch.dict(sys.modules, {published: self.policy}).start()
        self.model = _load("hyv4_model_hooks_test", _ROOT / "hunyuan_v4.py")
        self.cfg = SimpleNamespace(
            enable_prefill_cp=True, cp_strategy="interleave",
            tp_size=8, attn_cp_size=8, ep_size=8, moe_dense_tp_size=1,
            moe_a2a_backend="deepep", attention_backend="dsa",
            dsa_prefill_backend="flashmla_sparse", dsa_decode_backend="flashmla_kv",
            attn_dp_size=1, dp_size=1, pp_size=1, dcp_size=1,
            disaggregation_mode="null", speculative_algorithm=None,
            enable_mixed_chunk=False, enable_dsa_cache_layer_split=False,
            enable_hisparse=False, enable_hierarchical_cache=False,
            enable_two_batch_overlap=False,
        )
        self.hf = SimpleNamespace(
            architectures=["HYV4ForCausalLM"], learnable_sink=True,
        )

    @staticmethod
    def flashmla(q, *, attn_sink=None):
        return q

    def test_only_validated_topology_can_use_native_sink(self):
        self.policy._validate_profile(self.cfg, self.hf)
        for k, invalid in (
            ("dcp_size", 2), ("ep_size", 4), ("moe_dense_tp_size", 8),
            ("moe_a2a_backend", "none"), ("cp_strategy", "zigzag"),
        ):
            with self.subTest(k=k):
                invalid_profile = vars(self.cfg) | {k: invalid}
                with self.assertRaisesRegex(ValueError, "requires interleave CP8"):
                    self.policy._validate_profile(SimpleNamespace(**invalid_profile), self.hf)
        self.flashmla = lambda q: q
        with self.assertRaisesRegex(RuntimeError, "native attn_sink"):
            self.policy._validate_profile(self.cfg, self.hf)

    def test_model_provider_keeps_other_guard_and_does_not_mutate_upstream(self):
        new_provider = self.policy._cp_family_provider(upstream_family)
        self.assertEqual(new_provider(self.cfg, self.hf), {"attn_cp_size": 8, "page_size": 64})
        with self.assertRaisesRegex(ValueError, "not supported for HYV4"):
            upstream_family(self.cfg, self.hf)
        self.cfg.dcp_size = 2
        with self.assertRaisesRegex(ValueError, "does not support DCP"):
            new_provider(self.cfg, self.hf)

    def test_only_validated_hyv4_prefill_enables_strategy_cp(self):
        base = "sglang.srt.arg_groups.model_override_base"
        environ = "sglang.srt.environ"
        activations = []
        field = SimpleNamespace(set=activations.append)
        modules = {
            base: _module(
                base,
                resolving_view=lambda args: args,
                model_config_of=lambda args: SimpleNamespace(hf_config=self.hf),
            ),
            environ: _module(environ, envs=SimpleNamespace(SGLANG_ENABLE_CP_V2=field)),
        }
        original, adapted = object(), object()
        with (
            patch.dict(sys.modules, modules),
            patch.object(self.policy, "adapt_model_resolution", return_value=adapted),
        ):
            self.assertIs(
                self.policy.adapt_if_requested(
                    original, self.cfg, model_arch="DeepseekV4ForCausalLM"
                ),
                original,
            )
            self.cfg.enable_prefill_cp = False
            self.assertIs(
                self.policy.adapt_if_requested(
                    original, self.cfg, model_arch="HYV4ForCausalLM"
                ),
                original,
            )
            self.cfg.enable_prefill_cp = True
            self.cfg.dcp_size = 2
            with self.assertRaises(ValueError):
                self.policy.adapt_if_requested(
                    original, self.cfg, model_arch="HYV4ForCausalLM"
                )
            self.assertEqual(activations, [])

            self.cfg.dcp_size = 1
            self.assertIs(
                self.policy.adapt_if_requested(
                    original, self.cfg, model_arch="HYV4ForCausalLM"
                ),
                adapted,
            )
            self.assertEqual(activations, [True])

    def test_cp_runner_gets_full_embeddings_and_positions_for_sharding(self):
        model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: list(ids)))
        batch = SimpleNamespace(input_ids=(10, 20, 30), positions=(0, 1, 2))
        self.assertEqual(
            self.policy.prepare_cp_inputs(model, batch, marker=42),
            ([10, 20, 30], (0, 1, 2), {"marker": 42}),
        )
        supplied = object()
        self.assertIs(
            self.policy.prepare_cp_inputs(model, batch, input_embeds=supplied)[0], supplied
        )

    def test_only_cp_dense_constructor_uses_replicated_weights(self):
        def construct_mlp(self, **kwargs):
            return kwargs.get("parallel_group", "tp")

        def construct_layer(self, config, layer_id):
            return self_hooks.around_mlp_init(construct_mlp, self)

        self_hooks = self.model
        cfg = SimpleNamespace(mlp_layer_types=["dense", "sparse"])
        with patch.object(self_hooks, "_prefill_cp_enabled", lambda: True):
            self.assertEqual(self_hooks.around_decoder_init(construct_layer, object(), cfg, 0), "replicated")
            self.assertEqual(self_hooks.around_decoder_init(construct_layer, object(), cfg, 1), "tp")
        with patch.object(self_hooks, "_prefill_cp_enabled", lambda: False):
            self.assertEqual(self_hooks.around_decoder_init(construct_layer, object(), cfg, 0), "tp")
        self.assertEqual(self_hooks.around_mlp_init(construct_mlp, object()), "tp")
