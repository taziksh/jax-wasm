#!/usr/bin/env bash
# Stage 0/1 prerequisites: fetch all sources + toolchain.
#
# NOTE on transport: this kit fetches sources as TARBALLS from codeload.github.com
# (and emsdk's toolchain from storage.googleapis.com) rather than `git clone`,
# because some locked-down networks allow codeload/raw/googleapis but block the
# github smart-HTTP git endpoint and github Pages (llvm.github.io). Tarballs do
# not carry submodule contents, which is exactly why we pin and fetch llvm-project
# separately at StableHLO's exact LLVM commit (the version anchor; see env.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

fetch_tar() { # url destdir stripped-top-dir-name
  local url="$1" dest="$2" top="$3" tgz="$WORK/$(basename "$dest").tar.gz"
  [ -d "$dest" ] && { echo "[fetch] $dest exists, skip"; return; }
  echo "[fetch] $url"
  curl -sSL --retry 4 --retry-delay 2 --max-time 900 -o "$tgz" "$url"
  tar xzf "$tgz" -C "$WORK"
  mv "$WORK/$top" "$dest"
  rm -f "$tgz"
}

# 1. llvm-project at the StableHLO-pinned commit (the anchor).
fetch_tar "https://codeload.github.com/llvm/llvm-project/tar.gz/$LLVM_SHA" \
          "$LLVM_SOURCE_DIR" "llvm-project-$LLVM_SHA"

# 2. StableHLO (must pin the SAME llvm commit; assert it).
fetch_tar "https://codeload.github.com/openxla/stablehlo/tar.gz/refs/heads/$STABLEHLO_REF" \
          "$STABLEHLO_SOURCE_DIR" "stablehlo-$STABLEHLO_REF"
pin="$(cat "$STABLEHLO_SOURCE_DIR/build_tools/llvm_version.txt")"
if [ "$pin" != "$LLVM_SHA" ]; then
  echo "WARNING: StableHLO pins llvm $pin but we built against $LLVM_SHA." >&2
  echo "         Set LLVM_SHA=$pin and re-run, or use a StableHLO ref that matches." >&2
fi

# 3. eudsl (the wasm CAPI + cmake cache files we adapt).
fetch_tar "https://codeload.github.com/llvm/eudsl/tar.gz/refs/heads/$EUDSL_REF" \
          "$EUDSL_SOURCE_DIR" "eudsl-$EUDSL_REF"

# 4. emsdk + emscripten toolchain.
if [ ! -d "$EMSDK" ]; then
  fetch_tar "https://codeload.github.com/emscripten-core/emsdk/tar.gz/refs/heads/main" \
            "$EMSDK" "emsdk-main"
fi
"$EMSDK/emsdk" install "$EMSCRIPTEN_VERSION"
"$EMSDK/emsdk" activate "$EMSCRIPTEN_VERSION"

echo "[fetch] all sources present under $WORK"
