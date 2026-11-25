# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from hcu_sglang import multimodal_gen
from hcu_sglang.multimodal_gen import minimax_h3, qknorm_rope


class DiffusionPluginRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.srt_calls = []
        self.diffusion_calls = []
        srt_calls, diffusion_calls = self.srt_calls, self.diffusion_calls

        class SrtRegistry:
            _hooks = defaultdict(list)

            @staticmethod
            def register(*args):
                srt_calls.append(args)
                SrtRegistry._hooks[args[0]].append((args[2], args[1], None))

        class DiffusionRegistry:
            _hooks = defaultdict(list)

            @staticmethod
            def register(*args):
                diffusion_calls.append(args)
                DiffusionRegistry._hooks[args[0]].append((args[2], args[1], None))

        self.srt = ModuleType('sglang.srt.plugins.hook_registry')
        self.srt.HookRegistry = SrtRegistry
        self.srt.HookType = type('HookType', (), {'AROUND': object(), 'AFTER': object(), 'REPLACE': object()})
        self.diffusion = ModuleType('sglang.multimodal_gen.runtime.platforms.plugins')
        self.diffusion.HookRegistry = DiffusionRegistry
        self.platforms = ModuleType('sglang.multimodal_gen.runtime.platforms')
        self.platforms.current_platform = SimpleNamespace(is_rocm=lambda: True)
        self.modules = patch.dict('sys.modules', {
            self.srt.__name__: self.srt,
            self.diffusion.__name__: self.diffusion,
            self.platforms.__name__: self.platforms,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_diffusion_callback_uses_only_its_owned_registry(self):
        multimodal_gen.register()
        self.assertEqual(self.srt_calls, [])
        self.assertEqual(len(self.diffusion_calls), 43)
        targets = {call[0] for call in self.diffusion_calls}
        self.assertEqual(targets, {
            'sglang.kernels.ops.diffusion.rope.qknorm_rope_jit.can_use_fused_inplace_qknorm_rope',
            'sglang.multimodal_gen.configs.pipeline_configs.ltx_2.LTX2PipelineConfig.supports_dynamic_batching',
            'sglang.multimodal_gen.configs.pipeline_configs.wan.Wan2_2_T2V_A14B_Config.__post_init__',
            'sglang.multimodal_gen.configs.pipeline_configs.wan.Wan2_2_TI2V_5B_Config.__post_init__',
            'sglang.multimodal_gen.configs.pipeline_configs.wan.WanI2V480PConfig.__post_init__',
            'sglang.multimodal_gen.configs.sample.sampling_params.SamplingParams.add_cli_args',
            'sglang.multimodal_gen.configs.sample.sampling_params.SamplingParams.from_pretrained',
            'sglang.multimodal_gen.configs.sample.sampling_params.SamplingParams.from_user_sampling_params_args',
            'sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerImpl.forward_varlen',
            'sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn.SparseLinearAttentionImpl.forward',
            'sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn._initialize_linear_attention',
            'sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.SubBlockSparseAttentionImpl._build_dense_impl',
            'sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn._get_subblock_sparse_attention_runner',
            'sglang.multimodal_gen.runtime.layers.attention.layer.USPAttention._forward_with_replicated_kv_prefix_split',
            'sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.__init__',
            'sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.forward',
            'sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.TransformerLoader.load_state_dict_model',
            'sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.inspect_minimax_h3_safetensors',
            'sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader.resolve_transformer_quant_load_spec',
            'sglang.multimodal_gen.runtime.loader.fsdp_load.load_model_from_full_model_state_dict',
            'sglang.multimodal_gen.runtime.models.dits.ltx_2.LTX2Attention.forward',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3Attention.__init__',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTBlock.__init__',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTBlock.forward',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.post_load_weights',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3FinalLayer.__init__',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3FinalLayer.forward',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._modulate_gate',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._modulate_rmsnorm_scale_shift',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._modulate_scale_shift',
            'sglang.multimodal_gen.runtime.models.dits.qwen_image21.QwenImage21Attention.__init__',
            'sglang.multimodal_gen.runtime.models.dits.wanvideo.WanTransformer3DModel.__init__',
            'sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae.AutoencoderKL.setup_forward',
            'sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor.VAEProcessor.convert_numpy_to_tensor',
            'sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils.apply_rotary_pos_emb_qk',
            'sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor.PipelineExecutor.profile_execution',
            'sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising.MiniMaxH3DenoisingStage._run_full_loop',
            'sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatformBase.get_attn_backend_cls_str',
            'sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform.get_attn_backend_cls_str',
            'sglang.multimodal_gen.runtime.platforms.rocm_platform_plugin',
            'sglang.multimodal_gen.runtime.server_args.server_args.ServerArgs._adjust_attention_backend',
        })
        self.assertTrue(all(callable(call[1]) for call in self.diffusion_calls))

    def test_minimax_h3_attention_hook_preserves_requested_backends(self):
        minimax_h3.register_hooks(self.diffusion.HookRegistry)
        target = 'sglang.multimodal_gen.runtime.server_args.server_args.ServerArgs._adjust_attention_backend'
        [(hook_type, hook, _)] = self.diffusion.HookRegistry._hooks[target]
        self.assertIs(hook_type, self.srt.HookType.AFTER)

        server_args_module = ModuleType('sglang.multimodal_gen.runtime.server_args.server_args')
        server_args_module.Backend = SimpleNamespace(DIFFUSERS=object())
        pipeline = type('MiniMaxH3PipelineConfig', (), {})()
        cases = (
            ('fa', {}, {'text_encoder': 'torch_sdpa', 'transformer': 'fa'}),
            ('fa', {'transformer': 'fa', 'text_encoder': 'fa'},
                   {'transformer': 'fa', 'text_encoder': 'fa'}),
            ('fa', {'transformer': 'fa2'}, {'transformer': 'fa2', 'text_encoder': 'torch_sdpa'}),
            ('fa2', {}, {'text_encoder': 'torch_sdpa', 'transformer': 'fa2'}),
        )
        with patch.dict('sys.modules', {server_args_module.__name__: server_args_module}):
            with patch.object(minimax_h3, 'is_hcu', return_value=True):
                for backend, components, expected in cases:
                    with self.subTest(backend=backend, components=components):
                        args = SimpleNamespace(
                            attention_backend=backend,
                            backend=object(),
                            pipeline_config=pipeline,
                            component_attention_backends=dict(components),
                            component_attention_backend_roles={
                                'transformer': {'self': 'fa', 'cross': 'torch_sdpa'}
                            },
                        )
                        result = object()
                        self.assertIs(hook(result, args), result)
                        self.assertEqual(args.attention_backend, backend)
                        self.assertEqual(args.component_attention_backends, expected)
                        self.assertEqual(
                            args.component_attention_backend_roles,
                            {'transformer': {'self': 'fa', 'cross': 'torch_sdpa'}},
                        )

    def test_h3_qknorm_probe_is_scoped_to_gfx936_constructor(self):
        qknorm_rope.register_hooks(self.diffusion.HookRegistry)
        probe, attention_init = (
            hook for _, hook, _ in self.diffusion_calls
        )

        def construct(attention):
            attention._use_fused_qknorm_rope = probe(lambda: True)

        with (
            patch.object(qknorm_rope, "is_hcu", return_value=True),
            patch.object(qknorm_rope.torch.cuda, "current_device", return_value=2),
            patch.object(qknorm_rope, "get_gcn_arch") as arch,
        ):
            for name, expected in (("gfx936:sramecc+", True), ("gfx938", False)):
                with self.subTest(arch=name):
                    arch.return_value = name
                    attention = SimpleNamespace()
                    attention_init(construct, attention)
                    self.assertIs(attention._use_fused_qknorm_rope, expected)
                    arch.assert_called_with(2)
                    self.assertFalse(probe(lambda: True))

            arch.return_value = "gfx936"
            attention = SimpleNamespace()
            attention_init(
                lambda obj: setattr(obj, "_use_fused_qknorm_rope", probe(lambda: False)),
                attention,
            )
            self.assertFalse(attention._use_fused_qknorm_rope)

            def failed_constructor(attention):
                construct(attention)
                raise ValueError("constructor failed")

            with self.assertRaisesRegex(ValueError, "constructor failed"):
                attention_init(failed_constructor, SimpleNamespace())
            self.assertFalse(probe(lambda: True))

        with patch.object(qknorm_rope, "is_hcu", return_value=False):
            attention = SimpleNamespace()
            attention_init(construct, attention)
            self.assertTrue(attention._use_fused_qknorm_rope)

    def test_existing_srt_registration_stays_in_the_srt_registry(self):
        minimax_h3.register_hooks()
        qknorm_rope.register_hooks()
        self.assertEqual(len(self.srt_calls), 19)
        self.assertLessEqual(
            {
                'sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae.AutoencoderKL.setup_forward',
                'sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor.VAEProcessor.convert_numpy_to_tensor',
                'sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising.MiniMaxH3DenoisingStage._run_full_loop',
            },
            {call[0] for call in self.srt_calls},
        )
        self.assertEqual(self.diffusion_calls, [])

    def test_registration_failure_is_not_hidden(self):
        def reject(*args):
            raise RuntimeError('diffusion registry rejected target')

        with patch.object(self.diffusion.HookRegistry, 'register', side_effect=reject):
            with self.assertRaisesRegex(RuntimeError, 'diffusion registry rejected target'):
                multimodal_gen.register()
        self.assertEqual(self.srt_calls, [])

    def test_two_entry_point_aliases_do_not_wrap_targets_twice(self):
        multimodal_gen.register()
        first = list(self.diffusion_calls)
        multimodal_gen.register()
        self.assertEqual(self.diffusion_calls, first)

    def test_oot_platform_does_not_register_rocm_only_backends(self):
        self.platforms.current_platform.is_rocm = lambda: False
        multimodal_gen.register()
        targets = {call[0] for call in self.diffusion_calls}
        self.assertNotIn(
            'sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform.get_attn_backend_cls_str',
            targets,
        )
        self.assertNotIn(
            'sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerImpl.forward_varlen',
            targets,
        )
        self.assertIn(
            'sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.forward',
            targets,
        )
        self.assertEqual(len(targets), 41)
        self.assertEqual(self.srt_calls, [])

    def test_registry_reset_allows_registration_again(self):
        multimodal_gen.register()
        self.diffusion.HookRegistry._hooks.clear()
        self.diffusion_calls.clear()
        multimodal_gen.register()
        self.assertEqual(len(self.diffusion_calls), 43)

    def test_foreign_hooks_on_the_same_target_are_preserved(self):
        target = 'sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk'
        def foreign(original, *args):
            return original(*args)

        self.diffusion.HookRegistry.register(target, foreign, self.srt.HookType.AROUND)
        multimodal_gen.register()
        first = list(self.diffusion_calls)
        multimodal_gen.register()
        self.assertEqual(self.diffusion_calls, first)
        self.assertIs(self.diffusion.HookRegistry._hooks[target][0][1], foreign)
        self.assertEqual(len(self.diffusion.HookRegistry._hooks[target]), 2)


if __name__ == '__main__':
    unittest.main()
