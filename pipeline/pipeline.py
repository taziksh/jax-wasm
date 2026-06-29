"""The verified StableHLO -> wasm-ready-LLVM lowering pipeline.

Every pass here was confirmed present (and the middle of the chain validated on
real GPT-2 linalg via iree-opt; see tests/native_lowering_check.sh). Index width
is pinned to 32 because the target is wasm32: MLIR's LLVMTypeConverter funnels
every `index` to IntegerType(indexBitwidth) via convertIndexType, so
index-bitwidth=32 makes every descriptor/loop-index an i32, matching the wasm32
data layout.
"""

# Anchored, comma-separated MLIR textual pass pipeline.
PIPELINE = (
    "builtin.module("
    "func.func(stablehlo-legalize-to-linalg),"
    "one-shot-bufferize{bufferize-function-boundaries},"
    "func.func(convert-linalg-to-loops),"
    "convert-scf-to-cf,"
    "finalize-memref-to-llvm{index-bitwidth=32},"
    "convert-func-to-llvm{index-bitwidth=32},"
    "convert-arith-to-llvm{index-bitwidth=32},"
    "convert-math-to-llvm,"
    "convert-cf-to-llvm{index-bitwidth=32},"
    "reconcile-unrealized-casts)"
)


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
