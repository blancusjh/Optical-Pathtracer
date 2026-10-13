#!/usr/bin/env bash
# Legacy gallery selection; archive under out/ and explicitly review/promote.
# Usage: tools/render_gallery.sh [NAME...] [--list|--dry-run|--promote ...]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 tools/render_gallery.py "$@"
