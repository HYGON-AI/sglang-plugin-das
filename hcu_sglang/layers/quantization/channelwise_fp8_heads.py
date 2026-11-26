import torch


def compute_lm_head(original, self, hidden_states, lm_head, embedding_bias=None):
    from .channelwise_fp8 import ChannelwiseFp8Method

    method = getattr(lm_head, "quant_method", None)
    if self.use_fp32_lm_head and isinstance(method, ChannelwiseFp8Method):
        return method.apply_with_output_dtype(
            lm_head, hidden_states, embedding_bias, output_dtype=torch.float32
        )
    return original(self, hidden_states, lm_head, embedding_bias)


def init_lora_head(original, self, base_layer, lora_backend):
    from .channelwise_fp8 import ChannelwiseFp8Method

    if isinstance(getattr(base_layer, "quant_method", None), ChannelwiseFp8Method):
        raise ValueError("Channelwise FP8 LM head does not support lm_head LoRA")
    return original(self, base_layer, lora_backend)


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if is_hcu():
        HookRegistry.register(
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
            compute_lm_head,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.lora.layers.ParallelLMHeadWithLoRA.__init__",
            init_lora_head,
            HookType.AROUND,
        )
