import ast
import functools
import hashlib
import inspect
import logging
import os
import textwrap
import types


logger = logging.getLogger(__name__)
_SOURCE_HASHES = {
    "aiter_moe": "b4491162eb3828455c743387147e2c2daca9f2015d32281f017d039de0142e92",
    "fused_experts_impl_marlin": "fe37bc89348d300e1146d560c0ea4923b2bea97cc900ea4cc7d10f8ec605cbe8",
    "inplace_fused_experts": "714f065fb9ace416f6fc0a763f3da09ba0831bef4fea5eeb8f62b5955efb95c7",
    "outplace_fused_experts": "901a6be194666bec1bbe596a3772c67172ffac795a9000f9586e89d0e2bcb984",
    "moe_c_fused_experts": "3971a124291b6ffa6373896362707f2b8cae429706f0873408f15e7e8f523484",
}


def _syntax(original):
    if not isinstance(original, types.FunctionType) or original.__closure__:
        raise RuntimeError("AITER activation fusion requires verified Python functions")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)).strip())
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    if digest != _SOURCE_HASHES[original.__name__]:
        raise RuntimeError("AITER activation fusion source changed; review compatibility or disable fusion")
    return tree


def _clone(original, namespace):
    _syntax(original)
    result = types.FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def _private_entry(original, namespace):
    if not original.__closure__:
        if original.__name__ != "moe_c_fused_experts":
            raise RuntimeError("AITER activation fusion entry leaf changed")
        return _clone(original, namespace)
    if original.__module__ != "aiter.jit.utils.torch_guard":
        raise RuntimeError("AITER activation fusion entry wrapper changed")
    children = [i for i, cell in enumerate(original.__closure__)
                if isinstance(cell.cell_contents, types.FunctionType)]
    if len(children) != 1:
        raise RuntimeError("AITER activation fusion entry wrapper contract changed")
    return _private_entry(original.__closure__[children[0]].cell_contents, namespace)


class _FuseActivation(ast.NodeTransformer):
    def __init__(self):
        self.activations = 0
        self.consumers = 0

    def visit_If(self, node):
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
                and isinstance(node.body[0].value.func, ast.Name)
                and node.body[0].value.func.id == "moe_c_silu_and_mul"):
            self.activations += 1
            replacement = ast.parse(
                "if (use_int8_w4a8 and block_shape is None and curr_a2_scale is None "
                "and tokens_in_chunk >= 128 and intermediate_cache1.dtype == torch.bfloat16 "
                "and N == 1280 "
                "and _hcu_use_triton_quant(intermediate_cache2)):\n"
                "    _hcu_q, _hcu_scale = _hcu_fuse(intermediate_cache1.view(-1, N), dtype=torch.int8)\n"
                "else:\n"
                "    pass\n"
            ).body[0]
            replacement.orelse = node.body
            node.body = [replacement]
            return [*ast.parse("_hcu_q = None\n_hcu_scale = None").body, node]
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name)
                and node.func.id == "invoke_fused_moe_kernel_marlin"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "intermediate_cache2"):
            self.consumers += 1
            node.args[0] = ast.parse("_hcu_q if _hcu_q is not None else intermediate_cache2", mode="eval").body
            node.args[3] = ast.parse("_hcu_scale if _hcu_q is not None else curr_a2_scale", mode="eval").body
        return self.generic_visit(node)


class _PrivateImport(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_ImportFrom(self, node):
        if (node.module == "fused_moe_c" and node.level == 1
                and len(node.names) == 1 and node.names[0].name == "moe_c_fused_experts"):
            self.count += 1
            return ast.parse("moe_c_fused_experts = _hcu_moe_c_fused_experts").body[0]
        return node


def _compile(original, tree, namespace):
    ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getfile(original), "exec"), namespace)
    return namespace[original.__name__]


def _use_triton_quant(activation):
    backend = os.environ.get("AITER_PTQ_I8_BACKEND", "auto")
    return backend in ("legacy", "triton") or (
        backend == "auto" and activation.numel() >=
        int(os.environ.get("AITER_PTQ_I8_HIP_MAX_ELEMS", "25000000")))


def lazy_aiter_act_quant(original):
    @functools.wraps(original)
    def result(hidden_states, *args, **kwargs):
        if hidden_states.shape[0] < 128:
            return original(hidden_states, *args, **kwargs)
        return build_aiter_act_quant(original)(hidden_states, *args, **kwargs)

    return result


@functools.lru_cache(maxsize=None)
def build_aiter_act_quant(original):
    from aiter import fused_moe_c
    from hcu_sglang.layers.quantization.aiter_act_quant_kernel import silu_mul_int8

    namespace = dict(vars(fused_moe_c))
    namespace["_hcu_fuse"] = silu_mul_int8
    namespace["_hcu_use_triton_quant"] = _use_triton_quant
    core = fused_moe_c.fused_experts_impl_marlin
    tree = _syntax(core)
    transform = _FuseActivation()
    tree = transform.visit(tree)
    if (transform.activations, transform.consumers) != (1, 1):
        raise RuntimeError("AITER activation fusion producer/consumer contract changed")
    _compile(core, tree, namespace)
    for name in ("inplace_fused_experts", "outplace_fused_experts"):
        namespace[name] = _clone(getattr(fused_moe_c, name), namespace)
    namespace["moe_c_fused_experts"] = _private_entry(fused_moe_c.moe_c_fused_experts, namespace)
    entry_namespace = dict(original.__globals__)
    entry_namespace["_hcu_moe_c_fused_experts"] = namespace["moe_c_fused_experts"]
    tree = _syntax(original)
    transform = _PrivateImport()
    tree = transform.visit(tree)
    if transform.count != 1:
        raise RuntimeError("AITER activation fusion entry contract changed")
    private = _compile(original, tree, entry_namespace)

    @functools.wraps(original)
    def result(hidden_states, *args, **kwargs):
        import torch

        if hidden_states.shape[0] < 128 or torch.compiler.is_compiling():
            return original(hidden_states, *args, **kwargs)
        return private(hidden_states, *args, **kwargs)

    logger.info("W4A8 AITER/moe_c activation INT8 fusion enabled for BF16 width 640 with Triton quantization")
    return result
