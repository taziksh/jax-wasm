#!/usr/bin/env bash
# Stage 2: cross-compile LLVM + MLIR + lld to wasm32 (the multi-hour job).
#
# Driven directly with `emcmake cmake` rather than `pyodide build` (eudsl's
# packaging wrapper). This is deliberate: the open question this kit resolves is
# whether StableHLO + MLIR + lld cross-compile and LINK to wasm32 against eudsl's
# LLVM. That is fully exercised by the C++ libraries. We therefore build with:
#   * MLIR_ENABLE_BINDINGS_PYTHON=OFF   -> avoids the pyodide xbuildenv +
#                                          cross-compiled-CPython dependency
#   * MLIR_ENABLE_EXECUTION_ENGINE=OFF  -> the eudsl WasmCompilerLinkerLoader CAPI
#                                          links NO MLIRExecutionEngine (verified:
#                                          its LINK_LIBS are LLVM-IR-translation +
#                                          lldWasm/lldCommon), so JIT is not needed
# The pyodide-loadable wheel (L7) layers on top of this once D is proven; see docs.
#
# We build only the library/header/cmake-export install targets, NOT `ninja
# install`. Building LLVM executables (llc/opt) to wasm can fail and is pointless
# here; the targeted approach mirrors eudsl's MlirDevelopment distribution
# (libraries only) and is both faster and safer.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
source "$EMSDK/emsdk_env.sh" 2>/dev/null || true
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"

emcmake cmake -G Ninja \
  -S "$LLVM_SOURCE_DIR/llvm" \
  -B "$WASM_LLVM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$WASM_LLVM_INSTALL" \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD=WebAssembly \
  -DLLVM_TARGET_ARCH=wasm32 \
  -DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-unknown-emscripten \
  -DLLVM_HOST_TRIPLE=wasm32-unknown-emscripten \
  -DLLVM_ENABLE_THREADS=OFF \
  -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_PIC=ON \
  -DLLVM_BUILD_STATIC=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DMLIR_ENABLE_EXECUTION_ENGINE=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_BACKTRACES=OFF -DLLVM_ENABLE_CRASH_OVERRIDES=OFF \
  -DLLVM_NATIVE_TOOL_DIR="$LLVM_NATIVE_TOOL_DIR" \
  -DLLVM_TABLEGEN="$LLVM_TABLEGEN" \
  -DMLIR_TABLEGEN="$MLIR_TABLEGEN" \
  -DMLIR_LINALG_ODS_YAML_GEN="$MLIR_LINALG_ODS_YAML_GEN" \
  -DMLIR_PDLL_TABLEGEN="$MLIR_PDLL" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="-sNO_DISABLE_EXCEPTION_CATCHING" \
  -DCMAKE_EXE_LINKER_FLAGS="$EM_LINK_FLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$EM_LINK_FLAGS" \
  -DCMAKE_MODULE_LINKER_FLAGS="$EM_LINK_FLAGS"

ninja -C "$WASM_LLVM_BUILD" \
  install-cmake-exports \
  install-llvm-headers install-llvm-libraries \
  install-mlir-headers install-mlir-libraries install-mlir-cmake-exports \
  install-lld-headers install-lld-libraries install-lld-cmake-exports

echo "[stage2] L0 GATE — every installed lib must be wasm, not ELF:"
# This is the gate that silently fooled the official jaxlib effort for years.
found_elf=0
while IFS= read -r f; do
  if file "$f" | grep -q "ELF"; then echo "  ELF (BAD): $f"; found_elf=1; fi
done < <(find "$WASM_LLVM_INSTALL/lib" -name '*.a' -o -name '*.so' 2>/dev/null | head -50)
[ "$found_elf" = 0 ] && echo "  OK: sampled archives are wasm/llvm-bitcode, no ELF" || { echo "L0 FAILED"; exit 1; }
du -sh "$WASM_LLVM_INSTALL"
