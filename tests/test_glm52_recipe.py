# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/glm52-int4/launch.sh'

class Glm52RecipeTests(unittest.TestCase):
    def run_script(self, **values):
        env = os.environ.copy()
        for key in ['PYTHON_BIN', 'LOG_DIR', 'DRY_RUN', 'HOST', 'PORT', 'MODEL_PATH', 'NATIVE_ENV_FILE', 'AITER_SITE', 'FLASH_MLA_SITE', 'SGLANG_PYTHON_ROOT', 'ADAPTER_ROOT', 'PLUGIN_ROOT']:
            env.pop(key, None)
        env.update(values)
        return subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True)

    def test_missing_checkpoint_fails_before_launch(self):
        result = self.run_script(DRY_RUN='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Set MODEL_PATH', result.stderr)

    def test_dry_run_keeps_checkpoint_argument_and_validated_recipe(self):
        result = self.run_script(MODEL_PATH='/models/GLM 5.2 INT4', DRY_RUN='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertEqual(args[args.index('--model-path') + 1], '/models/GLM 5.2 INT4')
        for key, value in [('--tp-size', '8'), ('--ep-size', '8'), ('--attn-cp-size', '8'), ('--moe-dense-tp-size', '1'), ('--quantization', 'slimquant_w4a8_marlin'), ('--speculative-num-steps', '3'), ('--speculative-eagle-topk', '1'), ('--speculative-num-draft-tokens', '4')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertNotIn('--disable-cuda-graph', args)
        self.assertNotIn('--disaggregation-mode', args)
        self.assertNotIn('--enable-dcp', args)

    def test_native_environment_file_must_exist(self):
        result = self.run_script(MODEL_PATH='/models/glm52', DRY_RUN='1', NATIVE_ENV_FILE='/missing/glm52-env')
        self.assertNotEqual(result.returncode, 0)

    def test_source_paths_env_flags_and_pipeline_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            fake = temp / 'fake-python'
            arguments, flags = temp / 'args', temp / 'flags'
            fake.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$ARGS_FILE"\nprintf "%s\\n" "$HCU_SLIMQUANT" "$SGLANG_HCU_GLM_CT_BF16_ROUTER" "$SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS" "$PYTHONPATH" > "$FLAGS_FILE"\nexit 23\n')
            fake.chmod(0o755)
            native = temp / 'native.env'
            native.write_text('export AITER_SITE="/native/aiter site"\nexport FLASH_MLA_SITE="/native/flash site"\n')
            result = self.run_script(MODEL_PATH='/models/GLM 5.2', PYTHON_BIN=str(fake), ARGS_FILE=str(arguments), FLAGS_FILE=str(flags), LOG_DIR=str(temp / 'log dir'), NATIVE_ENV_FILE=str(native), PLUGIN_ROOT='/sources/plugin', ADAPTER_ROOT='/sources/adapter', SGLANG_PYTHON_ROOT='/sources/sglang', PYTHONPATH='/prior/path', DRY_RUN='0')
            self.assertEqual(result.returncode, 23, result.stderr)
            args = arguments.read_bytes().decode().split('\0')[:-1]
            self.assertEqual(args[args.index('--model-path') + 1], '/models/GLM 5.2')
            env = flags.read_text().splitlines()
            self.assertEqual(env[:3], ['0', '1', '1'])
            self.assertEqual(env[3], '/native/aiter site:/native/flash site:/sources/sglang:/sources/adapter:/sources/plugin:/prior/path')
            self.assertEqual(len(list((temp / 'log dir').glob('*.log'))), 1)

if __name__ == '__main__':
    unittest.main()
