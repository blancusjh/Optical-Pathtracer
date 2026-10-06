#!/usr/bin/env bash
# Renders the canonical gallery into docs/gallery. Every image is accompanied by a JSON
# record (scene hash, integrator, seed, samples, statistics) that makes it reproducible.
# Usage: tools/render_gallery.sh [NAME...]   (no names: everything; ~3 h on four cores)
set -euo pipefail
cd "$(dirname "$0")/.."
OWE=${OWE:-build/owe}
OUT=docs/gallery
mkdir -p "$OUT"
ONLY=" $* "
r() {
  local name=$1; shift
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $name "* ]]; then return 0; fi
  echo "== $name"; "$OWE" render "$@" --out "$OUT/$name" 2>&1 | grep -E "^paths|^first"
}

r the_lens            scenes/the_lens.owe          --spp 512
r telescope_naked     scenes/the_telescope.owe     --detector Naked    --spp 256
r telescope_eyepiece  scenes/the_telescope.owe     --detector Eyepiece --spp 512
r telescope_beside    scenes/the_telescope.owe     --detector Beside   --spp 256
r prism_in_sunlight   scenes/prism_in_sunlight.owe --spp 1024 --backend gpu
r glass_of_water      scenes/glass/glass_of_water.owe --spp 512
r glass_in_sunlight   scenes/glass/glass_in_sunlight.owe --spp 1024 --backend gpu
r tea_glass           scenes/glass/tea_glass.owe   --spp 1024 --backend gpu
r the_shallows        scenes/the_shallows.owe      --spp 512 --backend gpu
r study_room          scenes/the_study.owe         --detector Room --spp 512 --backend gpu
r study_eyepiece      scenes/the_study.owe         --detector Eyepiece --spp 512 --backend gpu
r bench_sensor        scenes/optical_bench.owe     --spp 4000
r bench_overview      scenes/optical_bench.owe     --detector Eye --integrator hybrid --spp 256
r camera_obscura_wall scenes/camera_obscura.owe    --spp 512
r observatory_room    scenes/the_observatory.owe   --detector Room --spp 512
r observatory_desk    scenes/the_observatory.owe   --detector Desk --spp 512
r observatory_slit    scenes/the_observatory.owe   --detector Slit --spp 256
r observatory_saturn  scenes/the_observatory.owe   --detector SaturnEyepiece --spp 1024
r observatory_jupiter scenes/the_observatory.owe   --detector JupiterEyepiece --spp 1024
r observatory_moon    scenes/the_observatory.owe   --detector MoonEyepiece --spp 512
rm -f "$OUT"/*.pfm
