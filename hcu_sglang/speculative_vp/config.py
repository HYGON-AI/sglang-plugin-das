import os

VP_SIZE_ENV = "HCU_PLUGIN_DRAFT_LM_HEAD_VP_SIZE"


def vp_size():
    value = int(os.environ.get(VP_SIZE_ENV, "1"))
    if value not in (1, 4, 8, 16):
        raise ValueError("Draft LM-head VP size must be one of 1, 4, 8, 16")
    return value


def after_add_cli_args(result, parser):
    if "--speculative-draft-lm-head-vp-size" not in parser._option_string_actions:
        parser.add_argument(
            "--speculative-draft-lm-head-vp-size",
            type=int, choices=(1, 4, 8, 16), default=None,
            help="Node-local vocabulary parallelism for EAGLE draft LM-head top-1.",
        )
    return result


def before_from_cli_args(cls, args):
    value = getattr(args, "speculative_draft_lm_head_vp_size", None)
    if value is not None:
        if value not in (1, 4, 8, 16):
            raise ValueError("Draft LM-head VP size must be one of 1, 4, 8, 16")
        os.environ[VP_SIZE_ENV] = str(value)


def validate(view):
    size = vp_size()
    if size == 1:
        return {}
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu() or not view.device.startswith("cuda"):
        raise ValueError("Draft LM-head VP requires the HCU plugin on a HIP device")
    if view.speculative_algorithm != "EAGLE" or view.speculative_eagle_topk != 1:
        raise ValueError("Draft LM-head VP requires EAGLE with topk=1")
    if view.attn_dp_size <= 1 or not view.enable_dp_lm_head:
        raise ValueError("Draft LM-head VP requires attention DP and DP LM-head")
    if view.tp_size != view.attn_dp_size or view.attn_cp_size != 1 or view.pp_size != 1:
        raise ValueError("Draft LM-head VP requires TP=attention DP, CP=1 and PP=1")
    if view.tp_size % (view.nnodes * size):
        raise ValueError("Draft LM-head VP groups must fit within each node")
    if view.speculative_token_map is not None:
        raise ValueError("Draft LM-head VP does not support token maps")
    if view.enable_fp32_lm_head or view.enable_lora:
        raise ValueError("Draft LM-head VP does not support FP32 LM-head or LoRA")
    if view.speculative_use_rejection_sampling:
        raise ValueError("Draft LM-head VP returns top-1, not a rejection-sampling distribution")
    if view.speculative_adaptive or view.enable_multi_layer_eagle:
        raise ValueError("Draft LM-head VP requires fixed-step, single-layer EAGLE")
    return {}
