#!/usr/bin/env bash
# Renders the gallery into docs/gallery. Every image is accompanied by a JSON
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
r prism_in_sunlight   scenes/prism_in_sunlight.owe --spp 1024 --backend gpu
r glass_of_water      scenes/glass/glass_of_water.owe --spp 512
r glass_in_sunlight   scenes/glass/glass_in_sunlight.owe --spp 1024 --passes 8 --integrator vcm --backend gpu
r tea_glass           scenes/glass/tea_glass.owe   --spp 1024 --passes 8 --integrator vcm --backend gpu \
                      --set 'Eye.position=(0,-0.25,0.30)' --set 'Eye.look_at=(0,-0.08,0.02)' \
                      --set 'Eye.fov=62deg' --set 'Eye.focus=0.33'
r the_shallows        scenes/the_shallows.owe      --spp 512 --backend gpu
r study_room          scenes/the_study.owe         --detector Room --spp 512 --backend gpu
r study_eyepiece      scenes/the_study.owe         --detector Eyepiece --spp 512 --backend gpu
r bench_sensor        scenes/optical_bench.owe     --spp 4000
r bench_overview      scenes/optical_bench.owe     --detector Eye --integrator hybrid --spp 256
r camera_obscura_wall scenes/camera_obscura.owe    --spp 512
r observatory_room    scenes/the_observatory.owe   --detector Room --spp 4096 --passes 8 --integrator path --backend gpu
r observatory_desk    scenes/the_observatory.owe   --detector Desk --spp 4096 --passes 8 --integrator path --backend gpu
rm -f "$OUT"/*.pfm
