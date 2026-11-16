"""HCU adaptations for SGLang multimodal generation."""


def register() -> None:
    from hcu_sglang.compat.aiter import bootstrap_aiter_compat
    bootstrap_aiter_compat()
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
    from hcu_sglang.multimodal_gen import attention, minimax_h3
    attention.register_hooks(HookRegistry)
    minimax_h3.register_hooks(HookRegistry)
