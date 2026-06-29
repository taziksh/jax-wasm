# Build architecture (grounded in eudsl + stablehlo source)

The wasm toolchain is built in stages because you cannot run tablegen inside a
wasm cross-build, and StableHLO must be compiled against the *same* LLVM as the
wasm MLIR. All of this mirrors `llvm/eudsl` (`scripts/llvm_wasm`,
`scripts/build_mlir_native_tools.sh`, `projects/mlir-python-bindings-wasm`) and
`openxla/stablehlo` (`build_tools/llvm_version.txt`,
`stablehlo/conversions/linalg/transforms`, `stablehlo/integrations/c`).

## Version anchor

StableHLO pins an exact LLVM commit (`build_tools/llvm_version.txt`); eudsl tracks
llvm `main` (shallow). For stage 3 to link, **stage 2 and StableHLO use the same
LLVM commit** = StableHLO's pin. `scripts/env.sh` sets `LLVM_SHA` to it and
`01_fetch_sources.sh` asserts the match.

## Stages

1. **Native host tools** (`02`) — configure native LLVM/MLIR, build only
   `llvm-tblgen` / `mlir-tblgen` / `mlir-linalg-ods-yaml-gen` / `mlir-pdll`
   (`LLVM_OPTIMIZED_TABLEGEN=ON`). eudsl normally `pip download`s these as
   `mlir_native_tools`; that host (`llvm.github.io`) isn't always reachable, so
   we build the identical binaries from source.
2. **Wasm LLVM/MLIR/lld** (`03`) — `emcmake cmake` with `LLVM_ENABLE_PROJECTS=mlir;lld`,
   `LLVM_TARGETS_TO_BUILD=WebAssembly`, `wasm32-unknown-emscripten`, static,
   threads off, host tablegen wired via `LLVM_TABLEGEN`/`MLIR_TABLEGEN`. We install
   only library/header/cmake-export components (not `ninja install`) — building
   LLVM executables to wasm is pointless and can fail. Linker flags
   (`-sALLOW_TABLE_GROWTH -sWASM_BIGINT …`) are verbatim from eudsl.
3. **StableHLO wasm** (`04`) — `emcmake cmake` against the stage-2
   `lib/cmake/mlir`+`llvm`; build `StablehloOps ChloOps StablehloLinalgTransforms
   StablehloCAPI`. `StablehloCAPI` carries `mlirRegisterAllStablehloPasses()`,
   which (verified in source) calls `registerPasses()` +
   `registerStablehloLinalgTransformsPasses()` — i.e. registers
   `stablehlo-legalize-to-linalg`.
4. **Harness / bindings** —
   - `05` builds `tests/pipeline_runner.c` to wasm and runs it in node (L2/L3),
     no pyodide. This is the direct proof StableHLO linked and lowers in wasm.
   - `patches/apply_eudsl_stablehlo.sh` does the L7 path: adds the StableHLO libs
     to eudsl's `WasmCompilerLinkerLoaderCAPI` `LINK_LIBS` and calls
     `mlirRegisterAllStablehloPasses()` at nanobind module init, so the in-tab
     PassManager resolves the pipeline. Then `pyodide build` produces the
     loadable wheel.

## Index width

`index-bitwidth=32` on the `*-to-llvm` passes: MLIR's `LLVMTypeConverter` funnels
every `index` through `convertIndexType → IntegerType::get(ctx, indexBitwidth)`,
so 32 makes every loop index / memref descriptor field an `i32`, matching the
wasm32 data layout. The 32-bit memref descriptor on the Python side is already in
eudsl's `wasm_execution_engine.py` (`get_ranked_memref_descriptor`).

## libm

softmax/gelu emit `fmaxf`/`expf`/`tanhf`. On wasm these are undefined symbols;
eudsl's wasm-ld passes `--allow-undefined` so host/emscripten libm resolves them
at instantiate time (or add a libm side-module / a small bitcode shim).
