from __future__ import annotations

import unittest

import hcu_sglang
from sglang.srt.plugins.hook_registry import HookRegistry


class MuseMediaConfigTests(unittest.TestCase):
    def test_public_config_preserves_native_fields_and_muse_extensions(self):
        hcu_sglang.register()
        from hcu_sglang.multimodal.encoder_preprocessing import (
            EncoderMediaProcessorConfig,
        )

        HookRegistry.apply_hooks()
        from sglang.srt.multimodal import encoder_preprocessing

        self.assertIs(
            encoder_preprocessing.EncoderMediaProcessorConfig,
            EncoderMediaProcessorConfig,
        )
        value = encoder_preprocessing.EncoderMediaProcessorConfig(
            preprocess_off_event_loop=True, audio_sample_rate=24_000
        )
        self.assertTrue(value.preprocess_off_event_loop)
        self.assertEqual(value.audio_sample_rate, 24_000)
        self.assertFalse(value.image_decode_mode)
        self.assertFalse(value.preserve_media_metadata)
        default = encoder_preprocessing.EncoderMediaProcessorConfig()
        self.assertFalse(default.preprocess_off_event_loop)
        self.assertIsNone(default.audio_sample_rate)


if __name__ == "__main__":
    unittest.main()
