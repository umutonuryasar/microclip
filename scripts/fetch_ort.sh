#!/usr/bin/env bash
# Vendor onnxruntime-web into the static demo. Run once; the files are gitignored.
set -euo pipefail
VERSION="1.26.0"
DEST="$(dirname "$0")/../demo/static/vendor/ort"
mkdir -p "$DEST"
for f in ort.wasm.bundle.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  curl -sSfL -o "$DEST/$f" "https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION}/dist/$f"
  echo "$f  $(du -h "$DEST/$f" | cut -f1)"
done
