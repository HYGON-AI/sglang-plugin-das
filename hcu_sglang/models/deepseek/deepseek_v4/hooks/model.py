from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.models.deepseek.hook_utils import register_method_replaces_from_path


def register_hooks():
    if not is_hcu():
        return 0
    from sglang.srt.models.deepseek_v4 import DeepseekV4Model
    from hcu_sglang.models.deepseek.current_v4.execution import core
    if not hasattr(DeepseekV4Model, "_forward_impl"):
        DeepseekV4Model._forward_impl = core.DeepseekV4Model._forward_impl
    return register_method_replaces_from_path(
        "sglang.srt.models.deepseek_v4", "DeepseekV4Model", ('__init__', '_check_late_layer_tail_readers', '_forward_layers_hc_pre_from_prev', '_forward_layers_tbo', 'forward', '_forward_impl'),
        "hcu_sglang.models.deepseek.deepseek_v4.methods.model",
    )
