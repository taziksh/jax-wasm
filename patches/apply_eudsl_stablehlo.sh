#!/usr/bin/env bash
# Integrate StableHLO into eudsl's wasm bindings build (the L7 / pyodide path).
#
# This is the STEP-1 integration from the kit, applied to a checkout of
# llvm/eudsl. It does three things (all grounded in files read from eudsl +
# openxla/stablehlo):
#   1b. add the StableHLO libs to WasmCompilerLinkerLoaderCAPI LINK_LIBS
#   1c. #include StablehloPasses.h and call mlirRegisterAllStablehloPasses()
#       at nanobind module init, so PassManager.parse("builtin.module(
#       stablehlo-legalize-to-linalg)") resolves in the tab (L2).
#
# Stage 3 (scripts/04_build_stablehlo_wasm.sh) must have built the StableHLO
# wasm libs first, and the eudsl CMake must be told where to find them
# (MLIR_DIR already points at the same install; add StableHLO's include/lib).
#
# Usage:  patches/apply_eudsl_stablehlo.sh /path/to/eudsl
set -euo pipefail
EUDSL="${1:?usage: apply_eudsl_stablehlo.sh /path/to/eudsl}"
P="$EUDSL/projects/mlir-python-bindings-wasm"
CML="$P/CMakeLists.txt"
NB="$P/WasmExecutionEngine.cpp"
[ -f "$CML" ] && [ -f "$NB" ] || { echo "not an eudsl wasm-bindings tree: $P"; exit 1; }

# --- 1b. LINK_LIBS: add StableHLO libs after lldCommon ------------------------
if ! grep -q 'StablehloLinalgTransforms' "$CML"; then
  python3 - "$CML" <<'PY'
import sys
f = sys.argv[1]; s = open(f).read()
needle = "  lldWasm\n  lldCommon\n"
add = ("  lldWasm\n  lldCommon\n"
       "  # --- StableHLO (added by jax-wasm kit) ---\n"
       "  StablehloOps\n  ChloOps\n  StablehloLinalgTransforms\n  StablehloCAPI\n")
assert needle in s, "could not find lldWasm/lldCommon LINK_LIBS block"
open(f, "w").write(s.replace(needle, add, 1))
print("patched CMakeLists.txt LINK_LIBS")
PY
else
  echo "CMakeLists.txt already has StableHLO libs"
fi

# --- 1c. register the StableHLO passes at nanobind module init ----------------
if ! grep -q 'mlirRegisterAllStablehloPasses' "$NB"; then
  python3 - "$NB" <<'PY'
import sys
f = sys.argv[1]; s = open(f).read()
inc = '#include "WasmCompilerLinkerLoaderCAPI.h"'
assert inc in s
s = s.replace(inc, inc + '\n#include "stablehlo/integrations/c/StablehloPasses.h"', 1)
anchor = "NB_MODULE(_mlirWasmExecutionEngine, m) {\n"
assert anchor in s
s = s.replace(anchor, anchor +
    "  // Register StableHLO passes (incl. stablehlo-legalize-to-linalg) so the\n"
    "  // in-tab PassManager can parse the lowering pipeline. (jax-wasm kit)\n"
    "  mlirRegisterAllStablehloPasses();\n", 1)
open(f, "w").write(s)
print("patched WasmExecutionEngine.cpp registration")
PY
else
  echo "WasmExecutionEngine.cpp already registers StableHLO passes"
fi

cat <<'NOTE'

[done] eudsl patched. Remaining wiring for the pyodide build:
  * point the bindings CMake at the StableHLO wasm install
    (include dir + lib dir from scripts/04), e.g. via CMAKE_PREFIX_PATH or by
    extending target_include_directories / link_directories.
  * the python Context that parses the StableHLO text must have the stablehlo +
    chlo dialects loaded. mlirRegisterAllStablehloPasses() covers passes (L2);
    dialect loading is the remaining python-glue item (register the dialect
    handles mlirGetDialectHandle__stablehlo__ / __chlo__ into the context, or
    ship stablehlo's python dialect module). See pipeline/realtime_jit.py.
NOTE
