"""HCU adaptations for SGLang multimodal generation."""


# anchor-baseline(②): main@5792cf6e loads diffusion hooks from a separate process-local registry.
def register():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

    from hcu_sglang.multimodal_gen import minimax_h3, qknorm_rope

    minimax_h3.register_hooks(registry=HookRegistry)
    qknorm_rope.register_hooks(registry=HookRegistry)
