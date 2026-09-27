#!/usr/bin/env bash
# Scene-model preview only. --export refreshes a running window; --render writes PNGs.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
cmake --build build --target export_model_preview
build/export_model_preview scenes/the_temple.owe out/model-review/temple
build/export_model_preview scenes/the_observatory.owe out/model-review/observatory
build/export_model_preview scenes/the_telescope.owe out/model-review/telescope
case "${1:-}" in
    --export) ;;
    --render) blender -b -noaudio --python-exit-code 1 --python tools/live_model_preview.py -- --render ;;
    '') blender -noaudio --python tools/live_model_preview.py ;;
    *) echo 'Usage: bash tools/review_models.sh [--export|--render]' >&2; exit 2 ;;
esac
