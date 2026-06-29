# jax-wasm — realtime JAX-in-the-browser via an in-tab MLIR→wasm JIT

Type a JAX function in a browser tab, shift-enter, and it compiles **in the tab**
to fresh WebAssembly and runs. The MLIR compiler is cross-compiled to wasm and
runs in Pyodide, emitting new wasm per cell. See `README.md` for the architecture.

## STATUS — L0–L6 PROVEN in wasm (this is real, not planned)

The whole pipeline was built to wasm and run end-to-end. Numbers match desktop JAX.

| Gate | State | Evidence |
|---|---|---|
| L0 toolchain emits wasm not ELF | ✅ | objects inside every `.a` are "WebAssembly binary module" |
| L2 StableHLO passes register in wasm | ✅ | `pipeline_runner.wasm` runs `stablehlo-legalize-to-linalg` |
| L3 lowering in wasm (smoke + GPT-2) | ✅ | 0 leftover stablehlo/linalg/unrealized (784 / 3892 LLVM ops) |
| L4 wasm-emitted kernel object | ✅ | `numeric_runner.wasm` emits `kernel.o` via in-wasm LLVM WebAssembly backend |
| L5 numerics (smoke softmax) vs JAX | ✅ | max\|wasm−jax\| = **3.6e-7** |
| L6 numerics (full GPT-2 block) vs JAX | ✅ | max\|wasm−jax\| = **1.2e-6** |
| **L7 live in-browser demo** | ⬜ **TODO** | needs the pyodide wheel + a browser (your Mac) |

Full details + exact numbers: `build_artifacts/RESULTS.md`. Honest state:
`docs/STATUS.md`. Ladder: `docs/VERIFICATION_LADDER.md`.

## IMPORTANT: the binaries don't transfer; the kit reproduces them

L0–L6 were proven on a remote Linux box; its wasm artifacts (the 690 MB MLIR
install, `pipeline_runner.wasm`, `numeric_runner.wasm`, the StableHLO libs) lived
in that box's ephemeral `/tmp` and are GONE. This repo has the **scripts +
patches** that rebuild them. On a fresh machine (e.g. this Mac):

```bash
scripts/01_fetch_sources.sh        # llvm-project@pin + stablehlo + eudsl + emsdk
scripts/02_build_native_tools.sh   # host tablegen (~tens of min)
scripts/03_build_wasm_llvm.sh      # LLVM+MLIR+lld -> wasm32 (HOURS; the big one)
scripts/04_build_stablehlo_wasm.sh # StableHLO -> wasm32 (applies the patches/ shims)
scripts/05_build_pipeline_runner.sh# re-confirm L2/L3 in wasm
```
Mac notes: `brew install emscripten` (or let `01` install emsdk); Apple-Silicon
native host tools build fine (`LLVM_TARGETS_TO_BUILD=host`); more cores = faster
than the 4-core box this was proven on. No proxy here, so downloads "just work"
(the original box needed curl workarounds; not your problem).

Pins (in `scripts/env.sh`): LLVM `b713aaedb6eedd7d20f666038d945e50eafd4c27`
(StableHLO's `build_tools/llvm_version.txt`), emscripten `3.1.46`.

## The verified pipeline (`pipeline/pipeline.py`)

`inline → stablehlo-legalize-to-linalg → one-shot-bufferize → buffer-results-to-
out-params → convert-linalg-to-loops → convert-scf-to-cf → llvm-request-c-wrappers
→ expand-strided-metadata → finalize-memref-to-llvm{index-bitwidth=32} →
convert-{func,arith,math,cf}-to-llvm{index-bitwidth=32} → reconcile-unrealized-
casts`. Every clause earned its place by running the build (see below).

## Integration issues already found + fixed (don't re-discover these)

All are committed as documented patch scripts wired into the build:
1. libraries-only install exports ~90 unbuilt tool targets → `patches/fixup_wasm_mlir_install.sh` (placeholder + native tablegen).
2. StableHLO lit test subdirs need FileCheck → dummy targets (`patches/patch_stablehlo_source.sh`).
3. StableHLO's own tablegen built to wasm can't read host files via node → disable `integrations/cpp/builder` (same patch).
4. emscripten `find_library` ignores abs paths (root-path mode) → link archives by direct path (`tests/CMakeLists.txt`).
5. `mlirRegisterAllStablehloPasses` needs the full pass/Vhlo/`Version` lib closure → link all + build `Version`.
6. `wait4` host libc symbol from LLVM support → `-sERROR_ON_UNDEFINED_SYMBOLS=0`.
7. GPT-2 QKV `memref.subview` left 4 unrealized casts → add `expand-strided-metadata`.
8. JAX entry lowers to symbol `main`, collides with C `main` → `numeric_runner` renames LLVM `main`→`jitfn`.

## NEXT: L7 (the only thing left) — best done here on the Mac

Goal: load the StableHLO-augmented MLIR in Pyodide and run the realtime loop in a
browser. Steps:
1. Build eudsl's `projects/mlir-python-bindings-wasm` **with** `MLIR_ENABLE_BINDINGS_PYTHON=ON`
   (this is the pyodide path — needs `pyodide build` + the xbuildenv; differs from
   the bindings-OFF build used for L0–L6).
2. Apply `patches/apply_eudsl_stablehlo.sh <eudsl>` (adds StableHLO libs to the
   CAPI LINK_LIBS + calls `mlirRegisterAllStablehloPasses()` at module init).
3. `pyodide build` → a `.whl`; `await micropip.install(...)` it in JupyterLite.
4. Wire `pipeline/realtime_jit.py` (jax.jit → lower via `pipeline.PIPELINE` →
   `WasmExecutionEngine` → call). The call ABI is verified:
   `_mlir_ciface_<fn>(result_sret*, arg0*, ...)` with 28-byte 32-bit memref
   descriptors; the function `malloc`s the sret result. See `tests/driver*.c`.

Open risk for L7: the pyodide build rebuilds MLIR with python bindings to wasm
(CPython cross-compile) — more involved than the bindings-OFF build proven here,
but it's packaging, not a conceptual unknown. The hard part (does StableHLO lower
real JAX to a correct wasm kernel) is DONE.

## Branch / git

Work branch: `claude/jax-browser-wasm-jit-as9tg5`. Commit + push as you go.
