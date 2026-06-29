#!/usr/bin/env bash
# Native (x86) de-risk of the lowering pipeline using IREE's iree-opt, which is
# pip-installable and bundles upstream MLIR. iree-opt does NOT expose the bare
# upstream `stablehlo-legalize-to-linalg` (only IREE's variant) nor the upstream
# `*-to-llvm` finalization passes, so this validates the MIDDLE of the pipeline
# on real GPT-2 linalg — in particular one-shot-bufferize, which the handoff
# flagged as the most likely place to need tuning.
#
# Result observed this session on fixtures/gpt2_block.stablehlo.mlir:
#   stablehlo->linalg (IREE variant) : 0 leftover stablehlo, 106 linalg, 0 iree-private
#   one-shot-bufferize               : exit 0, 0 leftover tensor, 0 stray casts
#   linalg->loops, scf->cf, reconcile: 0 linalg, 0 scf, 0 unrealized; residual
#                                      = cf+arith+memref+math+func (the ->llvm domain)
#
#   pip install iree-base-compiler
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IN="${1:-$HERE/../fixtures/gpt2_block.stablehlo.mlir}"

command -v iree-opt >/dev/null || { echo "need: pip install iree-base-compiler"; exit 1; }

echo "== stablehlo -> linalg (IREE variant) =="
iree-opt --iree-stablehlo-to-iree-input "$IN" -o /tmp/_linalg.mlir
echo "  stablehlo=$(grep -c 'stablehlo\.' /tmp/_linalg.mlir) linalg=$(grep -cE 'linalg\.' /tmp/_linalg.mlir) iree-private=$(grep -cE 'flow\.|stream\.|hal\.|util\.' /tmp/_linalg.mlir)"

echo "== one-shot-bufferize (the finicky step) =="
iree-opt --one-shot-bufferize="bufferize-function-boundaries" /tmp/_linalg.mlir -o /tmp/_buf.mlir
echo "  linalg=$(grep -cE 'linalg\.' /tmp/_buf.mlir) tensor=$(grep -cE 'tensor\.' /tmp/_buf.mlir) memref=$(grep -cE 'memref\.' /tmp/_buf.mlir)"

echo "== linalg->loops, scf->cf, reconcile =="
iree-opt --convert-linalg-to-loops --convert-scf-to-cf --reconcile-unrealized-casts /tmp/_buf.mlir -o /tmp/_loops.mlir
echo "  linalg=$(grep -cE 'linalg\.' /tmp/_loops.mlir) scf=$(grep -cE 'scf\.' /tmp/_loops.mlir) unrealized=$(grep -c unrealized_conversion_cast /tmp/_loops.mlir)"
echo "  residual dialects:"; grep -oE '\b(cf|arith|memref|math|func)\.[a-z_]+' /tmp/_loops.mlir | sed -E 's/\..*//' | sort | uniq -c | sort -rn
