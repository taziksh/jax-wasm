#!/usr/bin/env bash
# Stage 1: build host tablegen tools from source.
#
# Faithful to llvm/eudsl scripts/build_mlir_native_tools.sh. eudsl normally
# `pip download mlir_native_tools -f https://llvm.github.io/eudsl`, but that host
# is not always reachable; building from source gives the identical binaries.
# You cannot run tablegen inside a wasm cross-build, so these host tools are
# required by stages 2 and 3 (LLVM_TABLEGEN / MLIR_TABLEGEN / ...).
#
# Cost observed (4 cores, ccache cold): configure ~30s, build ~ tens of minutes,
# build tree ~ a few GB. Only the tablegen targets + their deps are built.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

cmake -G Ninja \
  -S "$LLVM_SOURCE_DIR/llvm" \
  -B "$NATIVE_TOOLS_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$NATIVE_TOOLS_INSTALL" \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build "$NATIVE_TOOLS_BUILD" --target \
  install-llvm-tblgen \
  install-llvm-config \
  install-mlir-tblgen \
  install-mlir-linalg-ods-yaml-gen \
  install-mlir-pdll

echo "[stage1] native tools:"
ls -la "$NATIVE_TOOLS_INSTALL/bin"
"$MLIR_TABLEGEN" --version | head -1

# Optional: reclaim disk. The install/ tree is all stages 2-3 need.
# rm -rf "$NATIVE_TOOLS_BUILD"
