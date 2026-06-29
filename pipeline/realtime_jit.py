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

ENTRY = "entry"  # wasm cannot export a function named "main" (see eudsl engine)


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

    # wasm cannot export `main`; rename the JAX entry before compiling.
    stablehlo_text = stablehlo_text.replace("@main", f"@{ENTRY}")

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

    def invoke(*np_args, out_shape, out_dtype=np.float32):
        # memref ABI: the lowered function (one-shot-bufferize + func-to-llvm)
        # takes the destination + sources as memref descriptors. For a clean
        # pointer-to-descriptor ABI you typically lower with llvm.emit_c_interface
        # and call `_mlir_ciface_<entry>`. This is the L4/L5 detail the build
        # discovers; wire the exact descriptor packing to match the chosen ABI.
        out = np.zeros(out_shape, out_dtype)
        descs = [ctypes.pointer(get_ranked_memref_descriptor(out))]
        descs += [ctypes.pointer(get_ranked_memref_descriptor(a)) for a in np_args]
        engine.invoke(f"_mlir_ciface_{ENTRY}", *descs)
        return out

    return invoke


def realtime(f, *args, out_shape, **kw):
    """One-call convenience: JAX function + concrete args -> wasm result."""
    text = stablehlo_from_jax(f, *args)
    return compile_callable(text)(*[np.asarray(a) for a in args], out_shape=out_shape, **kw)
