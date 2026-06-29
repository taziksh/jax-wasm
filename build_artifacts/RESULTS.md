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
