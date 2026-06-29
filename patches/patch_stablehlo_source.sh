#!/usr/bin/env bash
# Prepare an openxla/stablehlo source tree for a wasm cross-build against a
# libraries-only MLIR install. Two changes, both needed because StableHLO assumes
# a full LLVM build with test tools and assumes its custom tablegen can read host
# files:
#
#   1. Inject dummy FileCheck/count/not targets. StableHLO unconditionally adds
#      its test subdirs (add_subdirectory(tests)); their add_lit_target calls
#      DEPENDS on these LLVM test tools, which a libraries-only install lacks.
#      The dummies satisfy the dependency without building/running tests.
#
#   2. Disable integrations/cpp/builder. It builds StableHLO's OWN tablegen
#      (mlir_builder_tblgen) to wasm; run via node it cannot read the host .td
#      files (no NODERAWFS), so generation fails. The linalg-conversion passes and
#      the C-API do not need the cpp builder, so we drop it.
#
# Usage: patch_stablehlo_source.sh /path/to/stablehlo
set -euo pipefail
S="${1:?usage: patch_stablehlo_source.sh /path/to/stablehlo}"
top="$S/CMakeLists.txt"
cppc="$S/stablehlo/integrations/cpp/CMakeLists.txt"

# 1. dummy lit-tool targets before the subdirs that need them
if ! grep -q 'jaxwasm-dummy-lit-tools' "$top"; then
  python3 - "$top" <<'PY'
import sys
f=sys.argv[1]; s=open(f).read()
anchor="add_subdirectory(stablehlo)"
inject=("# jaxwasm-dummy-lit-tools\n"
        "foreach(_t FileCheck count not)\n"
        "  if(NOT TARGET ${_t})\n    add_custom_target(${_t})\n  endif()\n"
        "endforeach()\n\n")
assert anchor in s, "anchor not found"
open(f,"w").write(s.replace(anchor, inject+anchor, 1))
print("patched dummy lit tools")
PY
else echo "dummy lit tools already present"; fi

# 2. disable the cpp builder integration
if grep -qE '^add_subdirectory\(builder\)' "$cppc"; then
  sed -i 's/^add_subdirectory(builder)/# add_subdirectory(builder)  # jaxwasm: disabled (custom wasm tblgen cannot read host files via node; unused)/' "$cppc"
  echo "disabled integrations/cpp/builder"
else echo "cpp/builder already disabled"; fi
