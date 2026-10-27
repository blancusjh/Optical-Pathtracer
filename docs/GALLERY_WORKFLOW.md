# Gallery staging and review

Promotion requires `rights: "cleared"` in the manifest, recorded zero leaks, and at
most one region inconsistency per million paths (each stops its path, so at most that
share of paths loses its light; `MAX_INCONSISTENCY_RATE` in `tools/render_gallery.py`).
Rare float32 contact cases at sub-millimetre bodies far from the eye made exactly zero
unreachable over the ~600M paths of a gallery capture. All selected entries pass these
checks before any copy.
A clean counter record still requires visual review for noise and composition. Unresolved targets may be
rendered into local archives for diagnosis, but cannot be promoted. See [current status](STATUS.md).

`tools/render_gallery.sh` retains its thirteen legacy scene selections and render
arguments. Rendering now stages a raw archive under `out/gallery/` and leaves
existing `docs/gallery/` images intact. The new scan views are explicit opt-ins in
[the manifest](gallery/manifest.json). No render or existing raw PFM is deleted.
Use the same Python environment as the benchmark runner (NumPy is required by
its shared identity helpers).

```sh
bash tools/render_gallery.sh --list                  # the legacy default set
bash tools/render_gallery.sh --list --scans          # include scan views
bash tools/render_gallery.sh kunstkammer_room --dry-run --device 1
bash tools/render_gallery.sh kunstkammer_room peacock_lamp_room \
  --device 1 --binary build/owe --archive out/gallery/review_20261008
```

The archive retains native PNG, raw linear RGB PFM and JSON, stdout/stderr logs, exact command,
transport and process wall time, binary/source hashes, recursive scene/asset
hashes and host/device/environment identity. `--reuse` requires an exact matching
identity and unchanged artifacts; a mismatch never overwrites an earlier archive.
Choose a fresh directory after changing inputs. Scene/assets and executable are
also checked again after rendering to detect edits during capture. Unsupported
integrator fallback, wrong detector, resolution or sample count rejects the
candidate. Profiling state is captured in the environment; these gallery timings
are not automatically performance certification.
SPPM photon budgets are checked against native metadata. Native VCM/BDPT records
omit that field; when `--photons` is explicit, the archive retains the command and
records this verification limitation rather than claiming an independent budget check.
The shared source hash is explicitly scoped to the current capture workspace; the
executable hash identifies the frozen implementation, whose build revision must be
preserved separately when it differs from that workspace.

```sh
bash tools/render_gallery.sh kunstkammer_room \
  --archive out/gallery/review_20261008 --check
bash tools/render_gallery.sh kunstkammer_room \
  --archive out/gallery/review_20261008 --promote \
  --review-note 'Opened native image; complete objects framed, daylight/materials checked, contact statistics reviewed.'
```

Promotion requires an explicit set of names and a written review note. Open each
native image and assess composition, noise, material response and contact/transport
statistics first; file existence is not acceptance. Promotion checks the entire
set before copying PNG, native JSON and a review/provenance sidecar into
`docs/gallery/`. Raw PFM and logs remain in the archive. Check and promotion verify
the manifest entry hash, so changing the requested detector or render arguments
rejects an older archive even when its scene and asset files are unchanged.
Changed scenes/assets or
modified image files require a new render; do not manufacture provenance for old
renders that lack these sidecars. Historical reviewed outputs under `out/scenes/`
remain useful visual evidence, but their native scene hash alone does not prove
unchanged external assets or executable.

Permission for the invisible-window target photograph was confirmed on 26 October
2026, so both entries are `cleared` in the manifest; their historical sidecars still
record the earlier `unresolved` status. Their promotion now waits on clean
diagnostics and review. See [asset credits](ASSET_CREDITS.md). The manifest does not declare any scan image
visually accepted; a root review and the promotion note provide that decision.
New README image links should be added only after promotion.

## Curated scene asset setup

The four scan scenes require external downloads under the ignored `assets/` directory.
A clone includes the scene recipes, provenance and gallery images, but not those large
original assets. Setup from the repository root:

1. Download each original **glTF** archive listed in
   [the saved Sketchfab manifest](assets/sketchfab.json), using its source page and
   model ID. Login or author download access may be required; stable public archive
   endpoints were not recorded. Extract each archive so its `scene.gltf`, `scene.bin`
   and `textures/` occupy the recorded `assets/sketchfab/<slug>/` directory. Keep the
   saved licence/source texts in `docs/assets/sketchfab/`. Do not substitute a different
   format or revised scan and assume the same capture identity.
2. Fetch the original Poly Haven files to the exact `path` values in
   [the saved download manifest](assets/polyhaven.json). Its `url` values are public
   file URLs; the eight saved `docs/assets/polyhaven/<id>/files.json` records retain
   provider MD5 checksums and glTF included-file paths. Preserve subdirectories such
   as `textures/`. Availability has not been rechecked; this package performs no
   automatic network downloads.
3. Save the original lunar JPEG from the URL in
   [lunar source notes](assets/lunar_SOURCE.txt) as `assets/lunar/lroc_color_2k.jpg`.
   Install NumPy and Pillow, then run `python3 tools/scene_assets/derive_textures.py`
   and `python3 tools/scene_assets/classify_armour.py`. These helpers create surface
   derivatives and the armour material partition without rewriting any scene.
   Existing derivatives are preserved. The dark table uses the original
   `round_wooden_table_01_8k.gltf` with an explicit scene material that references
   generated `textures/dark_walnut_8k.jpg`; no dark-table glTF derivative is required. [Recipe provenance](assets/recipe_provenance.json)
   records the available dependency versions and reconstructed formulas; JPEG encoder
   differences and missing original steel/lunar helpers mean byte-identical output is
   not yet established. The classification preserves source triangles and coordinates.
4. Generate the two daytime HDRs using Blender with Cycles (CUDA when available,
   CPU fallback otherwise):
   `blender --background --factory-startup --python tools/scene_assets/generate_daylight.py`.
   [Daylight notes](assets/daylight_SOURCE.txt) and [saved parameters](assets/daylight.json)
   describe the original CUDA sky bake, solar alignment and calibration. Add `-- --cpu`
   to request CPU explicitly. Existing generated HDRs are preserved. Different devices
   can change encoded bytes. The Blender version used
   for the original capture was not recorded. This writes only local generated assets;
   it retains the scene's analytic Sun as the sole solar disc.
5. Run `python3 tools/scene_assets/check_assets.py` to inspect scene paths and recursive
   glTF buffer/image dependencies without hashing large assets. When no timed capture
   is running, `--integrity` additionally checks original Poly Haven files against
   available provider MD5s. Missing assets or a mismatch return a failing exit code.
   Exact scene/asset identity for comparison remains the gallery archive's hashes;
   this checker does not certify reconstructed derivatives against an old capture.

The frozen PMMA mesh and product JSON under `models/caustic/` are already included.
Rendering the invisible window does not require the target photograph or inverse
solver. The package does not redistribute that source image; the permission recorded
in the asset credits covers the derived renders. The tracked Peacock glTF
uses the original downloaded scan buffer and images at their recorded relative paths.
Do not run the historical `assets/curation/*/build_study.py` scripts: those construction
helpers replace scene fragments and are not an installation procedure.
