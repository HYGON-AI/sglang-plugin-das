# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import ModuleType
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
            @staticmethod
            def register(*args):
                srt_calls.append(args)

        class DiffusionRegistry:
            @staticmethod
            def register(*args):
                diffusion_calls.append(args)

        self.srt = ModuleType('sglang.srt.plugins.hook_registry')
        self.srt.HookRegistry = SrtRegistry
        self.srt.HookType = type('HookType', (), {'AROUND': object(), 'AFTER': object()})
        self.diffusion = ModuleType('sglang.multimodal_gen.runtime.platforms.plugins')
        self.diffusion.HookRegistry = DiffusionRegistry
        self.modules = patch.dict('sys.modules', {
            self.srt.__name__: self.srt,
            self.diffusion.__name__: self.diffusion,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_diffusion_callback_uses_only_its_owned_registry(self):
        multimodal_gen.register()
        self.assertEqual(self.srt_calls, [])
        self.assertEqual(len(self.diffusion_calls), 5)
        targets = {call[0] for call in self.diffusion_calls}
        self.assertEqual(targets, {
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm',
            'sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.post_load_weights',
            'sglang.multimodal_gen.runtime.server_args.server_args.ServerArgs._adjust_attention_backend',
            'sglang.kernels.ops.diffusion.rope.qknorm_rope_jit.can_use_fused_inplace_qknorm_rope',
        })
        self.assertTrue(all(callable(call[1]) for call in self.diffusion_calls))

    def test_existing_srt_registration_stays_in_the_srt_registry(self):
        minimax_h3.register_hooks()
        qknorm_rope.register_hooks()
        self.assertEqual(len(self.srt_calls), 5)
        self.assertEqual(self.diffusion_calls, [])

    def test_registration_failure_is_not_hidden(self):
        def reject(*args):
            raise RuntimeError('diffusion registry rejected target')

        with patch.object(self.diffusion.HookRegistry, 'register', side_effect=reject):
            with self.assertRaisesRegex(RuntimeError, 'diffusion registry rejected target'):
                multimodal_gen.register()
        self.assertEqual(self.srt_calls, [])


if __name__ == '__main__':
    unittest.main()
