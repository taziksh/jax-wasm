# L0–L3 results — produced by the actual wasm build (this session)

Toolchain: emscripten 3.1.46; LLVM @ b713aaed (StableHLO's pin); 4-core x86 host.
`pipeline_runner.wasm` (94 MB) = MLIR + StableHLO + the C-API harness, linked to
wasm32, run under node.

| Gate | Result |
|------|--------|
| L0 toolchain emits wasm not ELF | PASS — objects inside every stage-2/3 `.a` are "WebAssembly (wasm) binary module" |
| L2 StableHLO passes register in wasm | PASS — pipeline (incl. `stablehlo-legalize-to-linalg`) parses & runs in `pipeline_runner.wasm` |
| L3 smoke (matmul+softmax) lowers in wasm | PASS — 0 stablehlo, 0 linalg, 0 unrealized; 784 llvm ops |
| L3 GPT-2 block lowers in wasm | PASS — 0 stablehlo, 0 linalg, 0 memref, 0 unrealized; 3892 llvm ops |

The lowered LLVM-dialect IR (gzipped next to this file) shows index-bitwidth=32
(i32 args/descriptors for wasm32), `llvm.emit_c_interface` wrappers, and the
leading-sret result descriptor — i.e. the IR is ready for codegen + the
WasmExecutionEngine call ABI.

Integration issues discovered + fixed to get here (none were in the handoff):
1. libraries-only install exports unbuilt tool targets -> placeholder + native tablegen shim
2. StableHLO unconditional lit test subdirs need FileCheck -> dummy targets
3. StableHLO's own tablegen built to wasm can't read host files via node -> disable cpp/builder
4. emscripten find_library ignores abs paths (root-path mode) -> link by direct path
5. mlirRegisterAllStablehloPasses needs the full pass/Vhlo/Version lib closure -> link all + Version
6. wait4 host libc symbol from LLVM support -> -sERROR_ON_UNDEFINED_SYMBOLS=0
7. GPT-2 QKV memref.subview left 4 unrealized casts -> add expand-strided-metadata

## L4/L5 — wasm-emitted kernel computes correct numbers (added)

| Gate | Result |
|------|--------|
| L4 codegen lowered MLIR -> wasm object IN wasm | PASS — `numeric_runner.wasm` (in-wasm LLVM WebAssembly backend) emits `kernel.o` ("WebAssembly binary module") from the lowered smoke IR |
| L5 numerics of the wasm-emitted kernel vs JAX | PASS — softmax(x@w): **max\|wasm-jax\| = 3.58e-07**, row sums = 1.0 |

End-to-end proven: JAX -> StableHLO -> (lowered to LLVM in wasm) -> (wasm object
emitted by the wasm-compiled LLVM backend) -> executed in node -> result equals
desktop JAX to fp32 precision. The wasm-emitting JIT produces a correct kernel.

Gotcha fixed: a JAX entry lowers to a symbol named `main`, which collides with the
C runtime main when linked into a driver (node silently ran the kernel as main).
numeric_runner renames the LLVM `main` -> `jitfn`; the _mlir_ciface_ wrapper is
unaffected. (This is exactly why eudsl forbids calling a `main` symbol.)

## L6 — full GPT-2 block as a wasm-emitted kernel (added)

| Gate | Result |
|------|--------|
| L6 full GPT-2 block, wasm-emitted kernel vs JAX | PASS — **max\|wasm-jax\| = 1.19e-06** (LayerNorm+QKV+causal-attn+softmax+GELU, 5 inputs, out [8x64]) |

Complete: the realtime path produces a correct wasm kernel for a real transformer
block, not just a toy. Only L7 (in-browser pyodide packaging) remains.
