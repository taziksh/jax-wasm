# Op coverage: `stablehlo-legalize-to-linalg` vs. the GPT-2 fixture

Source-level audit of the **upstream** pass (openxla/stablehlo
`stablehlo/conversions/linalg/transforms/`) against every op in
`fixtures/gpt2_block.stablehlo.mlir`. Pass mode: `applyPartialConversion`, default
options (`enablePrimitiveOps=false`). This is what resolves the crux without
needing the wasm build to discover it.

| stablehlo op | pattern | → target | wasm32 / bitwidth concern |
|---|---|---|---|
| constant | `ConstConverterTensor` | `arith.constant` | none |
| reduce (add) | `ReduceOpToGenericConverter` | `linalg.generic` reduction | none |
| reduce (maximum) | same; body → `arith.maximumf` | `linalg.generic` | none (float, no index) |
| reduce terminator (`stablehlo.return`) | `ReduceRegionReturnOpConversion` | `linalg.yield` | none |
| broadcast_in_dim | `HloBroadcastInDimConverter` | `linalg.generic` | none |
| divide/subtract/multiply/add | `PointwiseToLinalgConverter` | `linalg.generic` + `arith.*` | none (f32 & i32) |
| sqrt/exponential/tanh | `PointwiseToLinalgConverter` | `linalg.generic` + `math.*` | none (→ libm; see BUILD.md) |
| **dot_general** (2D, [1]x[0]) | `DotGeneralOpConversion`, `isSimpleDot()` | **`linalg.matmul`** | none — no batch, no preprocessing |
| slice (static) | `SliceConverter` | `tensor.extract_slice` | none (offsets are static i64 *attrs*, compile-time) |
| transpose | `TransposeConverter` | `linalg.generic` | none |
| convert (f32→f32) | `mapConvertOpToStdScalarOp` | identity (no-op) | none |
| compare GT FLOAT → i1 | `PointwiseToLinalgConverter` | `arith.cmpf OGT` | none |
| compare GE SIGNED → i1 | same | `arith.cmpi sge` (on i32) | none (i32, not index/i64) |
| select | `PointwiseToLinalgConverter` | `arith.select` | none |
| **iota** (i32) | `IotaConverter` | `linalg.generic`+`linalg.index`+`index_cast`→**i32** | none — i64 variant (`IotaToMapConverter`) is gated behind `enablePrimitiveOps`, off by default |
| func.func / call / return (`tril`,`_where`) | left legal by partial conversion | unchanged | none — but NOT inlined (see below) |

## Verdict

**The upstream pass fully lowers this fixture with default options and no
preprocessing.** All `dot_general`s are simple 2-D contractions → `linalg.matmul`
directly (no `dot-general-to-dot`, no batch path, no CHLO). No op takes an
i64/index path that would break on wasm32: the only index cast (iota) targets
i32, and slice indices are static attributes folded into the result type.

## One operational consequence (already applied)

The pass does **not** inline the `tril`/`_where` helpers — the `func.call`s
survive. So `pipeline/pipeline.py` runs `inline` first (verified to flatten the
fixture to a single func, 0 calls) before `stablehlo-legalize-to-linalg`, which
avoids cross-function bufferization downstream.

Registration path: `mlirRegisterAllStablehloPasses()` →
`registerStablehloLinalgTransformsPasses()`
(`stablehlo/integrations/c/StablehloPasses.cpp`).
