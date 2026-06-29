# Verification ladder (L0–L7)

Run bottom-up. Each stage has a trusted oracle one level down, so a failure is
localized rather than ambiguous.

| L | Check | How | Oracle below |
|---|-------|-----|--------------|
| L0 | toolchain emits **wasm not ELF** | `file` on emcc output and on every stage-2/3 `.a`/`.so`; the gate that silently fooled the jaxlib effort for years | — |
| L1 | engine smoke | arith.addf module → eudsl `WasmExecutionEngine` → correct result (already live in public eudsl console) | L0 |
| L2 | StableHLO passes **registered** in wasm | `pipeline_runner.wasm` parses `stablehlo-legalize-to-linalg` (proves `mlirRegisterAllStablehloPasses()` linked) | L1 |
| L3 | pipeline **lowers** | run `PIPELINE` on smoke + gpt2; success = 0 leftover `stablehlo.`/`linalg.`/`unrealized_conversion_cast` | L2 |
| L4 | codegen | translate → wasm object → wasm-ld → dlopen; module instantiates, symbol resolves (handle libm `fmaxf`/`expf`/`tanhf`) | L3 |
| L5 | **numerics** | call with numpy inputs; compare to desktop JAX (`fixtures/*_io.npz`). matmul atol 1e-4, softmax 1e-5, elementwise 1e-6 | L4 |
| L6 | gpt2 block | repeat L3–L5 on `fixtures/gpt2_block.stablehlo.mlir` | L5 |
| L7 | realtime | wire `jax.jit → lower → PIPELINE → WasmExecutionEngine` behind one call in JupyterLite | L6 |

## What this kit executes directly

- **L0** — `scripts/03` asserts no ELF in the installed wasm libs; emcc L0 already
  verified (compiles C→wasm, runs in node).
- **L2 + L3** — `scripts/05_build_pipeline_runner.sh` builds `tests/pipeline_runner.c`
  to wasm and runs it in node on the fixtures. This is the tightest proof of the
  crux **in wasm**, with no pyodide in the loop.
- The **native** middle-of-pipeline de-risk (incl. the finicky bufferize) is
  reproducible on any x86 box with `tests/native_lowering_check.sh`.

## If a stage fails

- **L0 emits ELF** → emscripten toolchain wasn't used; check `emcmake`.
- **L2 unregistered pass** → a StableHLO lib didn't link into the harness/CAPI
  (stage 3 / patch); the error names the missing pass.
- **L3 leftover op** → a dialect/pass missing from the wasm build; the error names
  the unlegalized op — add its dialect.
- **L4 missing symbol (`fmaxf`/`expf`/`tanhf`)** → libm; eudsl's wasm-ld already
  passes `--allow-undefined` so host libm resolves at instantiate, or add a shim.
- **crux (upstream pass) fails where IREE's variant succeeded** → port IREE's
  preprocessing (`dot-general-to-dot`, `unfuse-batch-norm`, …) before
  legalize-to-linalg. These passes are visible in `iree-opt --help`.
