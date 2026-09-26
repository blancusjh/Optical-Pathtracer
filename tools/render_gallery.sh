#!/usr/bin/env bash
# Renders the canonical gallery into docs/gallery. Every image is accompanied by a JSON
# record (scene hash, integrator, seed, samples, statistics) that makes it reproducible.
set -euo pipefail
cd "$(dirname "$0")/.."
OWE=${OWE:-build/owe}
OUT=docs/gallery
mkdir -p "$OUT"
r() { local name=$1; shift; echo "== $name"; "$OWE" render "$@" --out "$OUT/$name" 2>&1 | grep -E "^paths|^first"; }

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
rm -f "$OUT"/*.pfm
