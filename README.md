# jax-wasm — realtime JAX-in-the-browser via an in-tab MLIR→wasm JIT

Type a JAX function in a browser tab, hit shift-enter, and it compiles **in the
tab** to fresh WebAssembly and runs — no server, no AOT step. The MLIR compiler
itself is cross-compiled to wasm and runs in Pyodide, emitting new wasm per cell.

```
JAX (Python, in Pyodide)
  │  jax.jit(f).lower(*args).compiler_ir("stablehlo")     ← pure Python, in-tab
  ▼
StableHLO ──► [MLIR pass pipeline, COMPILED TO WASM, running in the tab]
  │  stablehlo-legalize-to-linalg → one-shot-bufferize → linalg-to-loops
  │  → …-to-llvm{index-bitwidth=32} → translate-to-LLVM-IR
  ▼
LLVM IR ──► (LLVM wasm backend, in tab) ──► wasm object
  │  wasm-ld (as a library, in tab) → dlopen fresh wasm → call
  ▼
result (numpy ⟷ 32-bit memref descriptors)
```

The realtime property comes from the bracketed line: this is the eudsl
`WasmExecutionEngine` mechanism (already live for arith/memref/func at
<https://llvm.github.io/eudsl/console>). **What this kit adds is StableHLO + its
linalg-conversion passes into that wasm build**, so the input can be real JAX
output instead of hand-written low-level MLIR.

## What's here

```
scripts/      env.sh + 01..05 — the staged build, from sources to in-wasm test
  01_fetch_sources.sh         llvm-project (pinned) + stablehlo + eudsl + emsdk (tarballs)
  02_build_native_tools.sh    host tablegen from source  (eudsl can't pip-download it here)
  03_build_wasm_llvm.sh        LLVM+MLIR+lld → wasm32     (the multi-hour stage)
  04_build_stablehlo_wasm.sh   StableHLO → wasm32         (resolves the "does it link" unknown)
  05_build_pipeline_runner.sh  build + run the in-wasm lowering harness (L2/L3)
patches/      apply_eudsl_stablehlo.sh — wire StableHLO into eudsl's pyodide bindings (L7)
pipeline/     PIPELINE string + the in-tab realtime driver (L7)
tests/        pipeline_runner.c (in-wasm harness) + native_lowering_check.sh (x86 de-risk)
fixtures/     gpt2_block.stablehlo.mlir (real JAX), smoke, generator + numeric oracles (.npz)
docs/         BUILD.md, STATUS.md, VERIFICATION_LADDER.md
```

## Run it

```bash
scripts/01_fetch_sources.sh      # sources + emsdk
scripts/02_build_native_tools.sh # host tablegen   (~tens of min)
scripts/03_build_wasm_llvm.sh    # wasm LLVM/MLIR/lld  (HOURS; the gate is L0: wasm not ELF)
scripts/04_build_stablehlo_wasm.sh
scripts/05_build_pipeline_runner.sh   # → L2/L3: lowers smoke + gpt2 IN WASM
```

See `docs/VERIFICATION_LADDER.md` for the L0–L7 pass/fail ladder and
`docs/STATUS.md` for an honest proven-vs-unknown breakdown.

## Honest status

The lowering is proven on x86 (real GPT-2 StableHLO → wasm32 LLVM IR, and the
finicky bufferize step validated on real GPT-2 linalg — see
`tests/native_lowering_check.sh`). The single remaining build-gated unknown is
whether StableHLO cross-compiles to wasm32 and links into the engine — which
`scripts/03`+`04`+`05` resolve directly. This repo's scripts were authored and
the early stages executed on a constrained box (4 cores, no preinstalled
emscripten); `docs/STATUS.md` records exactly how far execution got.