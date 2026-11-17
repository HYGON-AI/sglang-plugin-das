import torch
from torch import nn
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase



def enabled(layer, quant_config, prefix):
    if quant_config is None or quant_config.get_name() not in (
        "compressed_tensors",
        "compressed-tensors",
    ):
        return False
    scheme = quant_config.get_scheme_dict(layer, layer_name=prefix)
    if scheme is None:
        return False
    w, a = scheme.get("weights"), scheme.get("input_activations")
    if w is None or w.type != "int":
        return False
    if not (
        w.num_bits == 8
        and w.strategy == "channel"
        and w.symmetric
        and not w.dynamic
        and a is not None
        and a.num_bits == 8
        and a.type == "int"
        and a.strategy == "token"
        and a.symmetric
        and a.dynamic
        and (scheme.get("format") or quant_config.quant_format) == "int-quantized"
    ):
        raise ValueError(
            f"Unsupported INT8 projection scheme for {prefix}; need channel W8/token A8"
        )
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        raise ValueError("DeepSeek V4.1 INT8 projection runtime currently requires HCU")
    return True

def _grouped(x, w, s, groups, rank, *, output_dtype=None):
    output_dtype = output_dtype or x.dtype
    if (
        x.ndim != 3 or x.shape[1] != groups or groups <= 0 or rank < 0
        or w.dtype != torch.int8 or w.shape != (groups * rank, x.shape[-1])
        or s.dtype != torch.float32 or s.shape != (groups * rank, 1)
        or not w.is_contiguous() or not s.is_contiguous()
        or x.device != w.device or x.device != s.device
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or output_dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError("Invalid channelwise INT8 projection tensors")
    m = x.shape[0]
    if not m or not rank:
        return torch.empty((m, groups, rank), dtype=output_dtype, device=x.device)
    if x.is_cuda and (groups, rank, x.shape[-1]) == (8, 1024, 4096):
        from hcu_sglang.layers.quantization.dsv41_int8_native import project, supported

        if supported(x, groups, rank, output_dtype):
            return project(x, w, s, groups, rank)
    xf = x.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scales = absmax / 127
    z = xf * (127 / absmax)
    q = (z.sign() * (z.abs() + 0.5).floor()).clamp(-127, 127).to(torch.int8)
    padded_m = max(32, (m + 7) // 8 * 8)
    outputs = []
    for group in range(groups):
        a = torch.nn.functional.pad(q[:, group, :].contiguous(), (0, 0, 0, padded_m - m))
        start, end = group * rank, (group + 1) * rank
        acc = torch._int_mm(a, w[start:end].t())[:m]
        y = (acc.float() * scales[:, group, :]) * s[start:end].t()
        outputs.append(y.to(output_dtype))
    return torch.stack(outputs, dim=1)


def grouped(x, layer, groups, rank):
    return _grouped(x, layer.weight, layer.weight_scale, groups, rank)

def linear(x, weight, scale, *, output_dtype=None):
    """Run a row-major W8A8 projection without dequantizing the weight."""
    if x.ndim < 2 or weight.ndim != 2:
        raise ValueError("INT8 projection expects matrix weights and batched inputs")
    shape = x.shape
    y = _grouped(
        x.reshape(-1, 1, shape[-1]),
        weight,
        scale,
        1,
        weight.shape[0],
        output_dtype=output_dtype,
    )
    return y.view(*shape[:-1], weight.shape[0])

class ChannelwiseInt8Method(QuantizeMethodBase):
    supports_lm_head_lora = False

    def process_weights_after_loading(self, layer):
        for name in ("weight", "weight_scale"):
            if not getattr(getattr(layer, name), "_int8_projection_loaded", False):
                raise ValueError(f"Missing INT8 projection {name}")
        scale = layer.weight_scale
        valid_rows = getattr(layer, "_int8_projection_valid_rows", scale.shape[0])
        if not torch.all(torch.isfinite(scale)) or not torch.all(
            scale[:valid_rows] > 0
        ):
            raise ValueError("INT8 projection requires finite positive channel scales")



    def apply(self, layer, x, bias=None):
        return self.apply_with_output_dtype(layer, x, bias, output_dtype=x.dtype)

    def apply_with_output_dtype(self, layer, x, bias=None, *, output_dtype):
        y = linear(x, layer.weight, layer.weight_scale, output_dtype=output_dtype)
        return y if bias is None else y + bias

def install(layer):
    """Keep the original TP/padded-vocabulary loader for weights and row scales."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    old = layer.weight
    loader = getattr(old, "weight_loader", default_weight_loader)
    output_dim = getattr(old, "output_dim", None)
    shard = getattr(layer, "shard_indices", None)
    layer._int8_projection_valid_rows = (
        shard.org_vocab_end_index - shard.org_vocab_start_index
        if shard is not None
        else old.shape[0]
    )
    for name, shape, dtype in (
        ("weight", old.shape, torch.int8),
        ("weight_scale", (old.shape[0], 1), torch.float32),
    ):
        param = nn.Parameter(
            torch.empty(shape, dtype=dtype, device=old.device), requires_grad=False
        )
        if output_dim is not None:
            param.output_dim = output_dim
        if name == "weight":
            param.input_dim = 1

        def load(param, value, *, _loader=loader, _name=name):
            if _name == "weight" and value.dtype != torch.int8:
                raise ValueError("INT8 projection requires INT8 checkpoint weights")
            if _name == "weight_scale" and (
                value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 1
            ):
                raise ValueError("INT8 projection requires FP32 [N, 1] scales")
            _loader(param, value)
            param._int8_projection_loaded = True

        param.weight_loader = load
        setattr(layer, name, param)
    layer.quant_method = ChannelwiseInt8Method()

def around_project_bias(original, head, latent_states, *, weight=None):
    method = getattr(head.markov_w2, "quant_method", None)
    if not isinstance(method, ChannelwiseInt8Method):
        return original(head, latent_states, weight=weight)
    if weight is not None:
        raise ValueError("Quantized Markov slices require matching scale offsets")
    dtype = torch.bfloat16 if head._opt_markov_w2_bf16 else torch.float32
    return method.apply(head.markov_w2, latent_states.to(dtype)).float()


def around_step_bias(original, head, prev_tokens, *, hidden_states, state):
    method = getattr(head.markov_w2, "quant_method", None)
    if not isinstance(method, ChannelwiseInt8Method):
        return original(head, prev_tokens, hidden_states=hidden_states, state=state)
    latent = head.get_prev_embeddings(prev_tokens)
    shard = head._tp_shard
    if shard is None:
        return head.project_bias(latent), state
    start, end = shard.org_vocab_start, shard.org_vocab_end
    dtype = torch.bfloat16 if head._opt_markov_w2_bf16 else torch.float32
    return linear(latent, head.markov_w2.weight[start:end],
                  head.markov_w2.weight_scale[start:end], output_dtype=dtype), state


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.models.deepseek_v4_dspark.DSparkV4MarkovHead.project_bias",
        around_project_bias, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.deepseek_v4_dspark.DSparkV4MarkovHead.step_bias",
        around_step_bias, HookType.AROUND,
    )

