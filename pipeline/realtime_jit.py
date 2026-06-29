"""Realtime JAX-in-the-tab driver (the L7 goal).

    jax.jit(f).lower(*args).compiler_ir("stablehlo")   # pure python, works in Pyodide
      -> reparse into the eudsl mlir Context (stablehlo dialect + passes registered)
      -> run pipeline.PIPELINE  (stablehlo -> ... -> llvm dialect, index-bitwidth=32)
      -> WasmExecutionEngine(module)  (emit wasm object, wasm-ld, dlopen, in tab)
      -> invoke with 32-bit memref descriptors over numpy arrays

This runs inside Pyodide using the eudsl `mlir` wheel built+patched by this kit.
It is the wiring that turns "type a function, shift-enter, get a result" into one
call. The lowering (L2/L3) is proven; the numeric-call ABI (L4/L5) is the part the
build is meant to discover, so the sharp edges there are flagged, not hidden.
"""
from __future__ import annotations
import ctypes
import numpy as np

from .pipeline import PIPELINE, assert_lowered

# Naming the entry `@main` is fine: eudsl's engine only forbids *calling* the raw
# `main` symbol; the `_mlir_ciface_main` wrapper (emitted via llvm-request-c-wrappers
# in PIPELINE) is callable. Verified against eudsl test_wasm_execution_engine.py.
ENTRY = "main"


def stablehlo_from_jax(f, *args):
    """Pure-Python in Pyodide: get StableHLO text for f at the given args."""
    import jax
    return str(jax.jit(f).lower(*args).compiler_ir("stablehlo"))


def _prepare_module(stablehlo_text):
    """Reparse StableHLO into the eudsl context and lower to LLVM dialect."""
    from mlir.ir import Context, Module, InsertionPoint  # eudsl mlir wheel
    from mlir.passmanager import PassManager

    ctx = Context()
    # The patched eudsl build registers StableHLO passes globally
    # (mlirRegisterAllStablehloPasses). The stablehlo + chlo DIALECTS must also be
    # loadable in this Context to parse the text; the eudsl wheel must expose
    # their dialect handles (see patches/apply_eudsl_stablehlo.sh note). With
    # upstream RegisterEverything only, allow_unregistered lets parsing proceed.
    ctx.allow_unregistered_dialects = True

    module = Module.parse(stablehlo_text, ctx)
    PassManager.parse(PIPELINE, context=ctx).run(module.operation)
    assert_lowered(str(module))   # L3 gate
    return module


def compile_callable(stablehlo_text):
    """Lower + compile to wasm in the tab; return an invoker(*np_arrays)."""
    from mlir.wasm_execution_engine import (
        WasmExecutionEngine, get_ranked_memref_descriptor,
    )
    module = _prepare_module(stablehlo_text)
    engine = WasmExecutionEngine(module)   # emit wasm + wasm-ld + dlopen, in tab

    def dptr(a):
        # eudsl convention (test_wasm_execution_engine.py): each memref arg is
        # passed as pointer-to-pointer-to-descriptor.
        return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))

    def invoke(*np_args, out_shape, out_dtype=np.float32, abi="sret"):
        # ABI grounded in eudsl test_wasm_execution_engine.py + observed lowering.
        # Two cases depending on whether buffer-results-to-out-params fires:
        #   abi="out_param": @main takes the output as a trailing memref arg the
        #       caller allocates -> (inputs..., out).
        #   abi="sret" (observed default): @main returns a memref, so
        #       emit_c_interface puts a result descriptor pointer FIRST; the
        #       function fills it with an internally-allocated buffer that we copy
        #       back. -> (out, inputs...).
        out = np.zeros(out_shape, out_dtype)
        if abi == "out_param":
            engine.invoke(f"_mlir_ciface_{ENTRY}",
                          *[dptr(a) for a in np_args], dptr(out))
            return out
        # sret: pass a result descriptor first; the function fills it. We read the
        # data back through the descriptor's `aligned` pointer + shape.
        res_desc = get_ranked_memref_descriptor(out)
        engine.invoke(f"_mlir_ciface_{ENTRY}",
                      ctypes.pointer(ctypes.pointer(res_desc)),
                      *[dptr(a) for a in np_args])
        n = int(np.prod(out_shape))
        elem_ptr = ctypes.cast(res_desc.aligned,
                               ctypes.POINTER(ctypes.c_float * n))
        result = np.ctypeslib.as_array(elem_ptr.contents).reshape(out_shape).copy()
        # NOTE: the 32b descriptor struct (offsets of allocated/aligned/offset/
        # shape/strides) comes from wasm_execution_engine.make_nd_memref_descriptor;
        # confirm field layout + buffer ownership/free against the real wasm build
        # at L5 before trusting numerics.
        return result

    return invoke


def realtime(f, *args, out_shape, **kw):
    """One-call convenience: JAX function + concrete args -> wasm result."""
    text = stablehlo_from_jax(f, *args)
    return compile_callable(text)(*[np.asarray(a) for a in args], out_shape=out_shape, **kw)
