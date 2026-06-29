#!/usr/bin/env bash
# Post-install shim for the libraries-only wasm MLIR/LLVM install (after stage 2).
#
# WHY: stage 2 installs only library/header/cmake-export components (building LLVM
# executables to wasm is pointless). But `install-cmake-exports` still EXPORTS the
# tool targets (llvm-tblgen, mlir-opt, FileCheck, ... ~90 of them) as imported
# executables pointing at bin/<tool>.js files that were never built. Two
# consequences when a downstream project (StableHLO) runs find_package(MLIR):
#   1. LLVM/MLIR exports validate those files exist -> FATAL "references the file
#      .../bin/<tool>.js but this file does not exist".
#   2. MLIRConfig hard-sets MLIR_TABLEGEN_EXE="mlir-tblgen" (the imported wasm
#      target), so tablegen would try to run wasm via node instead of natively.
#
# FIX (contained, reversible, no rebuild):
#   * placeholder every referenced bin/<tool> so the existence check passes (those
#     imported targets are never invoked when only linking libraries),
#   * repoint MLIR_TABLEGEN_EXE / MLIR_PDLL_TABLEGEN_EXE at the NATIVE host tools
#     so downstream tablegen runs natively.
#
# The cleaner upstream-blessed alternative is to build the install via LLVM's
# distribution mechanism (LLVM_DISTRIBUTIONS=MlirDevelopment) whose cmake-exports
# include only the distribution's library components. This shim avoids re-running
# the multi-hour build.
#
# Usage: fixup_wasm_mlir_install.sh <WASM_LLVM_INSTALL> <NATIVE_TOOLS_BIN>
set -euo pipefail
INSTALL="${1:?usage: fixup_wasm_mlir_install.sh <install> <native-bin>}"
NATIVE="${2:?need native tools bin dir}"

mkdir -p "$INSTALL/bin"
n=0
while read -r rel; do
  p="$INSTALL/${rel#\$\{_IMPORT_PREFIX\}/}"
  if [ ! -e "$p" ]; then : > "$p"; chmod +x "$p"; n=$((n+1)); fi
done < <(grep -rhoE '\$\{_IMPORT_PREFIX\}/bin/[A-Za-z0-9._-]+' "$INSTALL/lib/cmake/" 2>/dev/null | sort -u)
echo "[fixup] placeholdered $n tool files in $INSTALL/bin"

mlc="$INSTALL/lib/cmake/mlir/MLIRConfig.cmake"
sed -i "s#set(MLIR_TABLEGEN_EXE \"mlir-tblgen\")#set(MLIR_TABLEGEN_EXE \"$NATIVE/mlir-tblgen\")#" "$mlc"
sed -i "s#set(MLIR_PDLL_TABLEGEN_EXE \"mlir-pdll\")#set(MLIR_PDLL_TABLEGEN_EXE \"$NATIVE/mlir-pdll\")#" "$mlc"
echo "[fixup] repointed MLIR tablegen EXE vars at native tools:"
grep -nE 'set\(MLIR_(TABLEGEN|PDLL_TABLEGEN)_EXE' "$mlc"
