# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from typing import Any, Dict

from sglang.srt.weight_cache import protocol as upstream
from sglang.srt.weight_cache.protocol import _get_quant_field


class CacheConfig(upstream.CacheConfig):
    weight_layout_stamp: str = ""


def _hcu_channelwise_fp8_round_trips_via_ipc(quant_config: Any) -> bool:
    """HCU per-channel W8A8 FP8 served as compressed-tensors (DeepSeek V4.1
    Channel-FP8 checkpoints).

    The serialized FP8 weights are loaded as-is and the per-channel scales are
    ordinary parameters, so the state_dict carries the whole numeric state.
    What the load leaves outside it (fix-up flags, derived tensors) is carried
    by weight_cache.extra_state. Other compressed-tensors schemes repack
    weights (Marlin, W4A8) and stay unsupported.
    """
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return False
    if _get_quant_field(quant_config, "format") != "float-quantized":
        return False
    groups = _get_quant_field(quant_config, "config_groups")
    if not isinstance(groups, dict) or not groups:
        return False
    for group in groups.values():
        weights = group.get("weights") if isinstance(group, dict) else None
        if not isinstance(weights, dict) or not (
            weights.get("num_bits") == 8
            and weights.get("type") == "float"
            and weights.get("strategy") == "channel"
            and not weights.get("dynamic")
        ):
            return False
    return True



def check_ipc_quant_support(quant_method, quant_config, *, where):
    if quant_method == "compressed-tensors" and _hcu_channelwise_fp8_round_trips_via_ipc(quant_config):
        return
    upstream.check_ipc_quant_support(quant_method, quant_config, where=where)


def compute_weight_layout_stamp() -> str:
    """Fingerprint of the static expert placement.

    ``--init-expert-location`` / ``--ep-num-redundant-experts`` decide which
    expert weights each EP rank materializes. A client started with another
    placement would map tensors of the right shape holding the wrong experts,
    so the placement has to be part of the cache identity. The placement file
    is hashed by content: it is routinely regenerated under the same name.
    """
    from sglang.srt.runtime_context import get_exec

    moe = get_exec().moe
    location = moe.init_expert_location
    digest = hashlib.sha256()
    if getattr(moe, "moe_a2a_backend", None) == "megamoe":
        from hcu_sglang.layers.moe.mega_moe import get_hcu_mega_moe_runtime

        digest.update(f"channel-fp8-megamoe-v1:{get_hcu_mega_moe_runtime()};".encode())
    digest.update(f"redundant={moe.ep_num_redundant_experts};".encode())
    if isinstance(location, str) and os.path.isfile(location):
        with open(location, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    else:
        digest.update(f"location={location}".encode())
    return digest.hexdigest()[:16]


# Environment prefixes whose values routinely select a weight layout or a
# kernel path at load time. They are not part of the cache identity (most are
# runtime-only), but a daemon/engine difference is worth a loud warning.
LAYOUT_ENV_PREFIXES = ("SGLANG_", "DSPARK", "MOE_", "K3_", "ROCM_USE_", "MEGAMOE_")
# Per-process knobs that legitimately differ between a daemon and an engine.
LAYOUT_ENV_IGNORED = frozenset(
    {
        "SGLANG_TORCH_PROFILER_DIR",
        "SGLANG_BLOCK_NONZERO_RANK_CHILDREN",
        "SGLANG_RUN_ID",
        "SGLANG_WEIGHT_CACHE_DEBUG_DIR",
    }
)


def collect_layout_env() -> Dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(LAYOUT_ENV_PREFIXES) and k not in LAYOUT_ENV_IGNORED
    }


