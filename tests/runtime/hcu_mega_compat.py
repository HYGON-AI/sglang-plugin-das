from types import SimpleNamespace as S
import os
import unittest
from unittest.mock import Mock, patch

from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import server_args_hooks as H
from sglang.srt.arg_groups import mega_moe_hook as N
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


class MegaMoECompatibilityTests(unittest.TestCase):
    def view(self, **changes):
        return S(**({"moe_a2a_backend":"megamoe", "enable_two_batch_overlap":False, "enable_single_batch_overlap":False,"disaggregation_mode":"prefill"}|changes))

    def test_public_gate_accepts_provider_without_changing_platform(self):
        view=self.view()
        hardware=S(is_cuda=True,is_sm90=False,is_sm100=False,is_hip=False)
        with patch.object(N,"resolving_view",side_effect=lambda x:x), patch.object(N,"get_platform",return_value=hardware):
            with self.assertRaisesRegex(ValueError,"SM90"):
                N.check_mega_moe_compat(view)
            HookRegistry.reset()
            HookRegistry.register("sglang.srt.arg_groups.mega_moe_hook.check_mega_moe_compat",H.around_check_mega_moe_compat,HookType.AROUND)
            HookRegistry.apply_hooks()
            with patch.object(H,"is_hcu",return_value=True), patch.object(H,"_resolving_view",side_effect=lambda x:x), patch.dict(os.environ,{"SGLANG_HCU_MEGA_MOE_RUNTIME":"megamoe"}), patch("importlib.util.find_spec",return_value=object()):
                N.check_mega_moe_compat(view)
            self.assertFalse(hardware.is_sm90)
            self.assertFalse(hardware.is_hip)

    def test_non_hcu_and_other_backend_preserve_original(self):
        sentinel=object();original=Mock(return_value=sentinel)
        with patch.object(H,"is_hcu",return_value=False):
            self.assertIs(H.around_check_mega_moe_compat(original,self.view()),sentinel)
        with patch.object(H,"is_hcu",return_value=True), patch.object(H,"_resolving_view",side_effect=lambda x:x):
            self.assertIs(H.around_check_mega_moe_compat(original,self.view(moe_a2a_backend="deepep")),sentinel)
        self.assertEqual(original.call_count,2)

    def test_other_runtime_keeps_original_exception(self):
        error=ValueError("native runtime refusal");original=Mock(side_effect=error)
        with patch.object(H,"is_hcu",return_value=True), patch.object(H,"_resolving_view",side_effect=lambda x:x), patch.dict(os.environ,{"SGLANG_HCU_MEGA_MOE_RUNTIME":"deep_gemm"}):
            with self.assertRaises(ValueError) as result:
                H.around_check_mega_moe_compat(original,self.view())
        self.assertIs(result.exception,error)

    def test_overlap_decode_and_missing_provider_are_rejected(self):
        with patch.object(H,"is_hcu",return_value=True), patch.object(H,"_resolving_view",side_effect=lambda x:x), patch.dict(os.environ,{"SGLANG_HCU_MEGA_MOE_RUNTIME":"megamoe"}):
            for changes in [{"enable_single_batch_overlap":True},{"enable_two_batch_overlap":True},{"disaggregation_mode":"decode"}]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    H.around_check_mega_moe_compat(Mock(),self.view(**changes))
            with patch("importlib.util.find_spec",return_value=None), self.assertRaisesRegex(RuntimeError,"provider"):
                H.around_check_mega_moe_compat(Mock(),self.view())


if __name__=="__main__":unittest.main()
