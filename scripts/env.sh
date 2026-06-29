# shellcheck shell=bash
# Shared configuration for the jax-wasm build kit.
# Source this from every stage script:  source "$(dirname "$0")/env.sh"
#
# All values below are grounded in source actually read from llvm/eudsl
# (projects/mlir-python-bindings-wasm, scripts/llvm_wasm, scripts/build_mlir_native_tools.sh)
# and openxla/stablehlo (build_tools/llvm_version.txt) during investigation.

set -euo pipefail

# --- The single version anchor -------------------------------------------------
# StableHLO pins an exact LLVM commit; eudsl tracks llvm main (shallow). For
# stage 3 (StableHLO) to compile against the stage-2 wasm MLIR, BOTH must be the
# SAME llvm commit. So StableHLO's pin is the anchor for everything.
#   openxla/stablehlo @ main : build_tools/llvm_version.txt
export LLVM_SHA="${LLVM_SHA:-b713aaedb6eedd7d20f666038d945e50eafd4c27}"
export STABLEHLO_REF="${STABLEHLO_REF:-main}"
export EUDSL_REF="${EUDSL_REF:-main}"

# emscripten version that pyodide-build 0.25.1 pairs with (Makefile.envs:
# PYODIDE_EMSCRIPTEN_VERSION ?= 3.1.46). Any consistent emscripten validates the
# cross-compile/link (L0-L6); matching a *deployed* Pyodide matters only at L7.
export EMSCRIPTEN_VERSION="${EMSCRIPTEN_VERSION:-3.1.46}"

# --- Layout --------------------------------------------------------------------
export WORK="${WORK:-/tmp/jaxwasm}"
export LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-$WORK/llvm-project}"
export STABLEHLO_SOURCE_DIR="${STABLEHLO_SOURCE_DIR:-$WORK/stablehlo}"
export EUDSL_SOURCE_DIR="${EUDSL_SOURCE_DIR:-$WORK/eudsl}"
export EMSDK="${EMSDK:-$WORK/emsdk}"

export NATIVE_TOOLS_BUILD="${NATIVE_TOOLS_BUILD:-$WORK/native-tools-build}"
export NATIVE_TOOLS_INSTALL="${NATIVE_TOOLS_INSTALL:-$WORK/native-tools-install}"

export WASM_LLVM_BUILD="${WASM_LLVM_BUILD:-$WORK/wasm-llvm-build}"
export WASM_LLVM_INSTALL="${WASM_LLVM_INSTALL:-$WORK/wasm-llvm-install}"

export STABLEHLO_WASM_BUILD="${STABLEHLO_WASM_BUILD:-$WORK/stablehlo-wasm-build}"
export STABLEHLO_WASM_INSTALL="${STABLEHLO_WASM_INSTALL:-$WORK/stablehlo-wasm-install}"

export BINDINGS_BUILD="${BINDINGS_BUILD:-$WORK/bindings-build}"

# Host compiler (eudsl CI uses clang-18; clang 18.1.3 verified working here).
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export CCACHE_DIR="${CCACHE_DIR:-$WORK/ccache}"

# Native host tablegen produced by stage 1 (eudsl normally pip-downloads these
# as mlir_native_tools from llvm.github.io; that host is not always reachable, so
# stage 1 builds them from source instead).
export LLVM_NATIVE_TOOL_DIR="${LLVM_NATIVE_TOOL_DIR:-$NATIVE_TOOLS_INSTALL/bin}"
export LLVM_TABLEGEN="${LLVM_TABLEGEN:-$LLVM_NATIVE_TOOL_DIR/llvm-tblgen}"
export MLIR_TABLEGEN="${MLIR_TABLEGEN:-$LLVM_NATIVE_TOOL_DIR/mlir-tblgen}"
export MLIR_LINALG_ODS_YAML_GEN="${MLIR_LINALG_ODS_YAML_GEN:-$LLVM_NATIVE_TOOL_DIR/mlir-linalg-ods-yaml-gen}"
export MLIR_PDLL="${MLIR_PDLL:-$LLVM_NATIVE_TOOL_DIR/mlir-pdll}"

# emscripten link flags (verbatim from eudsl scripts/llvm_wasm/pyproject.toml).
export EM_LINK_FLAGS="-sALLOW_TABLE_GROWTH -sASSERTIONS -sNO_DISABLE_EXCEPTION_CATCHING -sWASM_BIGINT"

mkdir -p "$WORK"

echo "[env] LLVM_SHA=$LLVM_SHA  EMSCRIPTEN=$EMSCRIPTEN_VERSION  WORK=$WORK"
