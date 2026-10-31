#!/usr/bin/env python3
"""Extract HCU fork methods from hcu-sglang-now deepseek_common / deepseek_nextn."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOW = ROOT.parent / "hcu-sglang-now/hcu-sglang/python/sglang/srt/models"

COMMON_SOURCE = NOW / "deepseek_common/deepseek_weight_loader.py"
NEXTN_SOURCE = NOW / "deepseek_nextn.py"
UTILS_SOURCE = NOW / "utils.py"

COMMON_OUT = ROOT / "sglang_hcu_plugin/models/deepseek/deepseek_common/methods"
NEXTN_OUT = ROOT / "sglang_hcu_plugin/models/deepseek/deepseek_nextn/methods"
UTILS_OUT = ROOT / "sglang_hcu_plugin/models/model_utils/methods"

COMMON_CONST = ["_is_hcu"]
NEXTN_CONST: list[str] = []
UTILS_CONST = ["_is_hcu"]

COMMON_HEADER = '''\
"""HCU deepseek_common fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_common import constants as C
from sglang_hcu_plugin.models.deepseek.deepseek_common.methods._runtime import *  # noqa: F403

'''

NEXTN_HEADER = '''\
"""HCU deepseek_nextn fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_nextn.methods._runtime import *  # noqa: F403

'''

UTILS_HEADER = '''\
"""HCU sglang.srt.models.utils fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.model_utils import constants as C
from sglang_hcu_plugin.models.model_utils.methods._runtime import *  # noqa: F403

'''


def transform_source(src: str, const_names: list[str]) -> str:
    for name in const_names:
        src = re.sub(rf"\b{name}\b", f"C.{name}", src)
    return src


def extract_method(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    seg = ast.get_source_segment(source, item)
                    if not seg:
                        raise RuntimeError(f"no source for {class_name}.{method_name}")
                    return seg
    raise RuntimeError(f"missing {class_name}.{method_name}")


def extract_function(source: str, func_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            seg = ast.get_source_segment(source, node)
            if not seg:
                raise RuntimeError(f"no source for {func_name}")
            return seg
    raise RuntimeError(f"missing {func_name}")


def write_methods(
    out_dir: Path,
    header: str,
    source_path: Path,
    specs: list[tuple[str, str, str | None]],
    const_names: list[str],
    filename: str,
) -> None:
    text = source_path.read_text(encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [header]
    for class_name, method_name, export in specs:
        body = extract_method(text, class_name, method_name)
        body = transform_source(body, const_names)
        export_name = export or method_name
        if export_name != method_name:
            body = body.replace(f"def {method_name}(", f"def {export_name}(", 1)
        parts.append(body)
        parts.append("\n\n")
    (out_dir / filename).write_text("".join(parts), encoding="utf-8")
    print(f"wrote {filename} ({len(specs)} methods)")


def main() -> None:
    write_methods(
        COMMON_OUT,
        COMMON_HEADER,
        COMMON_SOURCE,
        [
            ("DeepseekV2WeightLoaderMixin", "do_load_weights", None),
            ("DeepseekV2WeightLoaderMixin", "post_load_weights", None),
        ],
        COMMON_CONST,
        "weight_loader.py",
    )

    write_methods(
        NEXTN_OUT,
        NEXTN_HEADER,
        NEXTN_SOURCE,
        [
            ("DeepseekModelNextN", "__init__", None),
            ("DeepseekModelNextN", "forward", None),
        ],
        NEXTN_CONST,
        "model.py",
    )

    utils_text = UTILS_SOURCE.read_text(encoding="utf-8")
    UTILS_OUT.mkdir(parents=True, exist_ok=True)
    body = extract_function(utils_text, "enable_fused_set_kv_buffer")
    body = transform_source(body, UTILS_CONST)
    (UTILS_OUT / "utils.py").write_text(UTILS_HEADER + body + "\n", encoding="utf-8")
    print("wrote utils.py (1 function)")

    print("done")


if __name__ == "__main__":
    main()
