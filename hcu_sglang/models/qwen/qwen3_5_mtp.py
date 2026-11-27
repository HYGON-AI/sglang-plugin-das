from __future__ import annotations

import torch
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer,
)
from sglang.srt.utils import add_prefix

from hcu_sglang.utils.detection import is_hcu


class MTPReplicatedLinear(ReplicatedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], self.output_size))
        output, _ = super().forward(x)
        return output


def around_init(original, self, config, quant_config=None, prefix=""):
    original(self, config, quant_config=quant_config, prefix=prefix)
    quant_config = self.quant_config
    if (
        not is_hcu()
        or type(self.fc) is not torch.nn.Linear
        or not isinstance(quant_config, CompressedTensorsConfig)
    ):
        return

    fc_prefix = add_prefix("mtp.fc", prefix)
    if should_ignore_layer(fc_prefix, ignore=quant_config.ignore):
        return
    try:
        target = find_matched_target(fc_prefix, self.fc, quant_config.target_scheme_map)
    except ValueError:
        return
    scheme = quant_config.target_scheme_map[target]
    weights = scheme.get("weights")
    activations = scheme.get("input_activations")
    if (
        weights is None
        or activations is None
        or (weights.type, weights.num_bits) != ("int", 8)
        or (activations.type, activations.num_bits) != ("int", 8)
    ):
        return
    self.fc = MTPReplicatedLinear(
        self.fc.in_features,
        self.fc.out_features,
        bias=False,
        params_dtype=self.fc.weight.dtype,
        quant_config=quant_config,
        prefix=fc_prefix,
    )
