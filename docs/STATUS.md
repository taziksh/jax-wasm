# Status — proven vs. discovered (honest)

## Proven by running code

- **JAX → StableHLO in Pyodide** — pure Python; works in-tab.
- **In-tab MLIR→wasm JIT (arith/memref/func)** — deployed, live at the eudsl console.
- **Real GPT-2 StableHLO lowers fully to wasm32 LLVM IR** — via IREE's compiler
  (1985 LLVM ops, 0 leftover linalg, wasm32 triple).
- **Crux corroborated** — IREE's stablehlo→linalg of the real GPT-2 block
  (`fixtures/gpt2_block.stablehlo.mlir`) emits **pure standard upstream linalg**
  (matmul/generic/fill/index/yield + tensor/arith/math), **0 IREE-private ops** —
  so it converges with the upstream `stablehlo-legalize-to-linalg` this kit links.
  Reproduce: `tests/native_lowering_check.sh`.
- **Bufferization de-risked** — the step the handoff flagged as most likely to
  break. On the real GPT-2 linalg, `one-shot-bufferize{bufferize-function-boundaries}`
  → exit 0, 0 leftover tensor, 0 stray casts; then linalg→loops, scf→cf,
  reconcile → 0 linalg / 0 scf / **0 unrealized casts**, residual exactly
  `cf+arith+memref+math+func` (the →llvm input domain).
- **GPT-2 + smoke fixtures are bitwidth-clean** — 0 i64, 0 index, 0 dynamic dims.
- **Numeric oracles exist AND are independently validated** — `fixtures/smoke_io.npz`,
  `fixtures/gpt2_block_io.npz` (desktop-JAX reference). Cross-checked by compiling
  the *same* fixtures with IREE's native llvm-cpu backend and running them:
  IREE-CPU vs JAX = **3.6e-7** (smoke) and **1.2e-6** (full GPT-2 block). This is
  an independent native validation of the lowering's numerics (L5/L6 for the math),
  separate from the wasm mechanics still to come.
  - Two findings from this: (1) the committed fixture and the regenerated one are
    bit-identical computations (IREE diff 0.0) — the op-histogram difference is
    only helper-function formulation (`tril`/`_where` inlined vs not). (2) test
    inputs must use `1/sqrt(fan_in)` weight scaling; unnormalized inputs drive
    outputs to O(1e4), making a correct 1e-6 *relative* error look like 0.02
    absolute — so L5/L6 must compare with relative tolerance.

## Executed while authoring this kit (constrained box: 4 cores, 15 GB RAM, ~29 GB disk, no preinstalled emscripten)

- ✅ **Stage 1** native host tablegen — built from source (llvm/mlir host tools).
- ✅ **L0 toolchain** — emscripten 3.1.46 hand-assembled around a proxy that
  truncated emsdk's Python downloads (node + the wasm-binaries bundle fetched via
  `curl` from googleapis); `emcc` compiles C→wasm and runs in node.
- 🔄 **Stage 2** wasm LLVM/MLIR/lld — configured clean (emcmake) and building
  (5212 targets). This is the multi-hour stage; see the run log for where it got.

## Will be discovered (needs the build to finish)

- Does StableHLO cross-compile to wasm32 against this LLVM and **link** into the
  harness/CAPI (stage 3 + L2). No fundamental blocker seen — StableHLO's deps are
  all upstream dialects already in stage 2.
- **L3 in wasm** — does the full pipeline string lower the fixtures with 0
  leftovers when the passes run *inside* wasm (`scripts/05`).
- **L4/L5 numerics** — the memref-descriptor call ABI (likely needs
  `llvm.emit_c_interface`); compare to the `.npz` oracles.
- **Performance** — first version is correct-but-slow (no SIMD). `-msimd128` and
  kernel work come later. Keep shapes static (dynamic shapes reintroduce the i64
  hazards audited out here).

## Environment notes (why the scripts look the way they do)

- This network allows `codeload.github.com` tarballs, `raw.githubusercontent.com`,
  `storage.googleapis.com`, apt, and PyPI — but **blocks** `git clone` over
  github's smart-HTTP and `llvm.github.io` (GitHub Pages, where eudsl's prebuilt
  wheels live). Hence: fetch sources as tarballs; build host tablegen from source
  instead of `pip download mlir_native_tools`.
- Stage 2 is built with `MLIR_ENABLE_BINDINGS_PYTHON=OFF` and
  `MLIR_ENABLE_EXECUTION_ENGINE=OFF` deliberately — the eudsl CAPI links neither,
  and dropping them isolates the real unknown (cross-compile + link) from the
  pyodide/CPython packaging, which is a separately-solved L7 concern.
