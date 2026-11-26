_DEFAULTS = {
    "SGLANG_MINIMAX_H3_QUANT_OPTIMIZE": "true",
    "SGLANG_MINIMAX_H3_INT8_ENABLE": "true",
    "SGLANG_MINIMAX_H3_INT8_FUSION": "true",
    "SGLANG_MINIMAX_H3_FP8_OPTIMIZE": "true",
    "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_LIGHTOP_GEMM": "false",
}


def __getattr__(name):
    if name not in _DEFAULTS:
        raise AttributeError(name)
    from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
    return get_bool_env_var(name, _DEFAULTS[name])
