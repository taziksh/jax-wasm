"""The verified StableHLO -> wasm-ready-LLVM lowering pipeline.

Every pass here was confirmed present (and the middle of the chain validated on
real GPT-2 linalg via iree-opt; see tests/native_lowering_check.sh). Index width
is pinned to 32 because the target is wasm32: MLIR's LLVMTypeConverter funnels
every `index` to IntegerType(indexBitwidth) via convertIndexType, so
index-bitwidth=32 makes every descriptor/loop-index an i32, matching the wasm32
data layout.
"""

# Anchored, comma-separated MLIR textual pass pipeline.
#
# The last block sets up the call ABI, grounded in eudsl's working
# test_wasm_execution_engine.py:
#   * buffer-results-to-out-params turns a function that RETURNS a memref (what
#     bufferizing JAX's `@main` returning a tensor produces) into one that takes
#     the output as a trailing memref OUT-PARAM, so the caller pre-allocates it.
#   * llvm-request-c-wrappers stamps `llvm.emit_c_interface` on every func, so
#     convert-func-to-llvm emits the `_mlir_ciface_<name>` wrapper that the
#     WasmExecutionEngine calls with pointer-to-pointer-to-descriptor args.
# `inline` is first: the StableHLO from JAX keeps small helpers as separate
# func.func (e.g. the GPT-2 block's `tril`/`_where`), and stablehlo-legalize-to-
# linalg does NOT inline them. Flattening to a single region first (verified:
# 1 func, 0 calls on the GPT-2 fixture) avoids cross-function bufferization.
PIPELINE = (
    "builtin.module("
    "inline,"
    "func.func(stablehlo-legalize-to-linalg),"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "buffer-results-to-out-params,"
    "func.func(convert-linalg-to-loops),"
    "convert-scf-to-cf,"
    "func.func(llvm-request-c-wrappers),"
    "finalize-memref-to-llvm{index-bitwidth=32},"
    "convert-func-to-llvm{index-bitwidth=32},"
    "convert-arith-to-llvm{index-bitwidth=32},"
    "convert-math-to-llvm,"
    "convert-cf-to-llvm{index-bitwidth=32},"
    "reconcile-unrealized-casts)"
)
# Call ABI note (L4/L5): `buffer-results-to-out-params` is intended to turn the
# returned tensor into a trailing caller-allocated out-param. If it does NOT fire
# (observed: `@main` still `-> memref<8x64xf32>`), emit_c_interface uses the
# leading-sret convention instead: _mlir_ciface_main(result_desc*, arg0*, ...),
# where the function fills `result_desc` with an internally-allocated buffer the
# caller reads back (and frees). realtime_jit.py documents both; confirm which
# fires against the real wasm build before wiring numerics.


def lower(module, context=None):
    """Run PIPELINE on an mlir.ir.Module in place; return it.

    `module` is an mlir.ir Module (e.g. reparsed from
    jax.jit(f).lower(*args).compiler_ir("stablehlo")). Requires the StableHLO
    passes to be registered in the process (the eudsl build patched by
    patches/apply_eudsl_stablehlo.sh calls mlirRegisterAllStablehloPasses()).
    """
    from mlir.passmanager import PassManager  # provided by the eudsl mlir wheel
    pm = PassManager.parse(PIPELINE, context=context or module.context)
    pm.run(module.operation)
    return module


def assert_lowered(text):
    """Sanity gate (L3): no stablehlo/linalg/unrealized casts must remain."""
    for bad in ("stablehlo.", "linalg.", "unrealized_conversion_cast"):
        n = text.count(bad)
        assert n == 0, f"L3 FAIL: {n} leftover {bad!r}"
    return True
