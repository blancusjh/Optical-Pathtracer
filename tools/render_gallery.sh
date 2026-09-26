#!/usr/bin/env bash
# Renders the canonical gallery into docs/gallery. Every image is accompanied by a JSON
# record (scene hash, integrator, seed, samples, statistics) that makes it reproducible.
# Usage: tools/render_gallery.sh [NAME...]   (no names: everything; ~3 h on four cores)
# The temple's statue is a museum scan fetched by tools/fetch_assets.sh.
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
r the_prism           scenes/the_prism.owe         --spp 512
r statue_direct       scenes/the_statue.owe        --detector Direct    --spp 256
r statue_close        scenes/the_statue.owe        --detector Close     --spp 256
r statue_magnifier    scenes/the_statue.owe        --detector Magnifier --spp 256
r statue_telescope    scenes/the_statue.owe        --detector Telescope --spp 512
r glass_of_water      scenes/glass_of_water.owe    --spp 512
r bench_sensor        scenes/optical_bench.owe     --spp 4000
r bench_overview      scenes/optical_bench.owe     --detector Eye --integrator hybrid --spp 256
r camera_obscura_wall scenes/camera_obscura.owe    --spp 512
r temple_wide         scenes/the_temple.owe        --spp 512
r temple_colonnade    scenes/the_temple.owe        --detector Colonnade --spp 512
r temple_cam_f2       scenes/the_temple.owe        --detector Cam --spp 1024
r temple_cam_f11      scenes/the_temple.owe        --detector Cam --spp 1024 --set Cam.f_number=11 --set "Cam.focus=25 m"
r observatory_room    scenes/the_observatory.owe   --detector Room --spp 512
r observatory_desk    scenes/the_observatory.owe   --detector Desk --spp 512
r observatory_slit    scenes/the_observatory.owe   --detector Slit --spp 256
r observatory_saturn  scenes/the_observatory.owe   --detector SaturnEyepiece --spp 1024
r observatory_jupiter scenes/the_observatory.owe   --detector JupiterEyepiece --spp 1024
r observatory_moon    scenes/the_observatory.owe   --detector MoonEyepiece --spp 512
rm -f "$OUT"/*.pfm
