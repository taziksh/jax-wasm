#!/usr/bin/env bash
# Stage 4 (lite): build the in-wasm lowering harness and run it in node.
#
# Compiles tests/pipeline_runner.c to wasm32 against the stage-2 wasm MLIR install
# and the stage-3 StableHLO wasm build, then runs it under node on the fixtures.
# This proves, with NO pyodide/python in the loop:
#   L2  the StableHLO passes register in wasm (pipeline parses)
#   L3  the pipeline lowers to 0 leftover stablehlo./linalg./unrealized casts
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/env.sh"
source "$EMSDK/emsdk_env.sh" 2>/dev/null || true
export PATH="$EMSDK/upstream/emscripten:$EMSDK/upstream/bin:$PATH"

RUNNER_BUILD="${RUNNER_BUILD:-$WORK/runner-build}"
emcmake cmake -G Ninja \
  -S "$ROOT/tests" \
  -B "$RUNNER_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$WASM_LLVM_INSTALL/lib/cmake/mlir" \
  -DLLVM_DIR="$WASM_LLVM_INSTALL/lib/cmake/llvm" \
  -DSTABLEHLO_SOURCE_DIR="$STABLEHLO_SOURCE_DIR" \
  -DSTABLEHLO_BUILD_DIR="$STABLEHLO_WASM_BUILD"
ninja -C "$RUNNER_BUILD" pipeline_runner

NODE="$EMSDK/node/22.16.0_64bit/bin/node"
[ -x "$NODE" ] || NODE=node
RUNNER="$RUNNER_BUILD/pipeline_runner.js"

# The verified lowering pipeline (index width pinned to 32 for wasm32), incl. the
# call-ABI passes (buffer-results-to-out-params, llvm-request-c-wrappers) so this
# exercises the exact pipeline the in-tab runtime uses. Keep in sync with
# pipeline/pipeline.py PIPELINE.
PIPELINE='builtin.module(inline,func.func(stablehlo-legalize-to-linalg),one-shot-bufferize{bufferize-function-boundaries},buffer-results-to-out-params,func.func(convert-linalg-to-loops),convert-scf-to-cf,func.func(llvm-request-c-wrappers),finalize-memref-to-llvm{index-bitwidth=32},convert-func-to-llvm{index-bitwidth=32},convert-arith-to-llvm{index-bitwidth=32},convert-math-to-llvm,convert-cf-to-llvm{index-bitwidth=32},reconcile-unrealized-casts)'

run_one() {
  local mlir="$1" name="$2"
  echo "=== $name ==="
  out="$("$NODE" "$RUNNER" "$mlir" "$PIPELINE")" || { echo "  RUN FAILED"; return 1; }
  s=$(printf '%s' "$out" | grep -c 'stablehlo\.' || true)
  l=$(printf '%s' "$out" | grep -c 'linalg\.' || true)
  u=$(printf '%s' "$out" | grep -c 'unrealized_conversion_cast' || true)
  echo "  leftover stablehlo=$s linalg=$l unrealized=$u"
  [ "$s" = 0 ] && [ "$l" = 0 ] && [ "$u" = 0 ] && echo "  L3 PASS" || echo "  L3 FAIL"
}

# L3 lowering string only goes to linalg+loops cleanly; the full ->llvm tail is
# included so a PASS means the whole pipeline ran in wasm.
run_one "$ROOT/fixtures/smoke.stablehlo.mlir" "smoke (matmul+softmax)"
run_one "$ROOT/fixtures/gpt2_block.stablehlo.mlir" "gpt2 block"
