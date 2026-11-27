import __future__
import ast
import inspect
from functools import lru_cache, update_wrapper
from textwrap import dedent, indent
from types import CellType, CodeType, FunctionType


def _select_verify_intermediate_state_indices(
    default_indices, req_pool_indices, valid, pool_size
):
    from sglang.srt.layers.attention.linear.utils import (
        pp_spec_stable_rows_enabled,
        select_verify_intermediate_state_indices,
    )

    if callable(valid):
        if not pp_spec_stable_rows_enabled():
            return default_indices
        valid = valid()
    return select_verify_intermediate_state_indices(
        default_indices, req_pool_indices, valid, pool_size
    )


@lru_cache(maxsize=2)
def defer_verify_valid_mask(original):
    if not inspect.isfunction(original) or original.__closure__ is not None:
        return original
    try:
        lines, first_line = inspect.getsourcelines(original)
    except OSError:
        return original

    selector_name = "select_verify_intermediate_state_indices"
    source = f"def bind({selector_name}):\n" + indent(dedent("".join(lines)), "    ")
    tree = ast.parse(source)
    eager_mask = ast.dump(
        ast.parse("cache_indices[: query_start_loc.shape[0] - 1] >= 0", mode="eval").body
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == selector_name
            and len(node.args) == 4
            and ast.dump(node.args[2]) == eager_mask
        ):
            deferred = ast.parse("lambda: None", mode="eval").body
            deferred.body = node.args[2]
            node.args[2] = ast.copy_location(deferred, node.args[2])
            break
    else:
        return original

    ast.increment_lineno(tree, first_line - 2)
    ast.fix_missing_locations(tree)
    flags = original.__code__.co_flags & __future__.annotations.compiler_flag
    module_code = compile(
        tree, original.__code__.co_filename, "exec", flags, dont_inherit=True
    )
    binder_code = next(
        code for code in module_code.co_consts
        if isinstance(code, CodeType) and code.co_name == "bind"
    )
    method_code = next(
        code for code in binder_code.co_consts
        if isinstance(code, CodeType) and code.co_name == original.__name__
    )
    replacement = FunctionType(
        method_code,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        (CellType(_select_verify_intermediate_state_indices),),
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(replacement, original)
