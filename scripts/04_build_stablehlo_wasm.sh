#!/usr/bin/env bash
# Stage 3: cross-compile StableHLO to wasm32 against the stage-2 MLIR install.
#
# Grounded in openxla/stablehlo: the legalize-to-linalg passes live in
# stablehlo/conversions/linalg/transforms (target StablehloLinalgTransforms);
# mlirRegisterAllStablehloPasses() (which registers them) is carried by
# StablehloCAPI via stablehlo/integrations/c/StablehloPasses.h. All of
# StableHLO's deps are upstream dialects already built in stage 2.
#
# Build targets are the libraries the eudsl CAPI will link (STEP 1b of the kit):
#   StablehloOps  ChloOps  StablehloLinalgTransforms  StablehloCAPI
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
source "$EMSDK/emsdk_env.sh" 2>/dev/null || true
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"

emcmake cmake -G Ninja \
  -S "$STABLEHLO_SOURCE_DIR" \
  -B "$STABLEHLO_WASM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$STABLEHLO_WASM_INSTALL" \
  -DMLIR_DIR="$WASM_LLVM_INSTALL/lib/cmake/mlir" \
  -DLLVM_DIR="$WASM_LLVM_INSTALL/lib/cmake/llvm" \
  -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
  -DSTABLEHLO_BUILD_EMBEDDED=OFF \
  `# EMBEDDED=OFF + stablehlo as top-level source => STABLEHLO_STANDALONE_BUILD,` \
  `# which calls find_package(MLIR REQUIRED CONFIG) using MLIR_DIR above.` \
  -DLLVM_ENABLE_THREADS=OFF \
  -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_PIC=ON \
  -DLLVM_TARGETS_TO_BUILD=WebAssembly \
  -DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-unknown-emscripten \
  -DLLVM_HOST_TRIPLE=wasm32-unknown-emscripten \
  -DLLVM_TABLEGEN="$LLVM_TABLEGEN" \
  -DMLIR_TABLEGEN="$MLIR_TABLEGEN" \
  -DMLIR_LINALG_ODS_YAML_GEN="$MLIR_LINALG_ODS_YAML_GEN" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_FLAGS="-sNO_DISABLE_EXCEPTION_CATCHING" \
  -DCMAKE_EXE_LINKER_FLAGS="$EM_LINK_FLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$EM_LINK_FLAGS" \
  -DCMAKE_MODULE_LINKER_FLAGS="$EM_LINK_FLAGS"

ninja -C "$STABLEHLO_WASM_BUILD" \
  StablehloOps ChloOps StablehloLinalgTransforms StablehloCAPI ChloCAPI

echo "[stage3] StableHLO wasm static libs:"
find "$STABLEHLO_WASM_BUILD" -name 'libStablehlo*.a' -o -name 'libChlo*.a' 2>/dev/null | head
echo "[stage3] L0 gate on a StableHLO lib:"
lib="$(find "$STABLEHLO_WASM_BUILD" -name 'libStablehloLinalgTransforms.a' | head -1)"
[ -n "$lib" ] && file "$lib"
