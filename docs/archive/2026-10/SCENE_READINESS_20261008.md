Historical record, retained for provenance. Current state: [STATUS.md](../../STATUS.md)

# Scene and historical-baseline audit, 8 October 2026

## Completed gallery milestone — 2026-10-10

Kunstkammer joinery and Window materials/framing are adopted. The final engine passes
157/157 hardware and 21/21 targeted software checks. Kunstkammer, Peacock and all three
Gregorian images and both Window views are locally promoted after native review.
Window Room SPPM 64 retains 168 inconsistencies and 21 leaks; Wall SPPM 128 retains
243 and 34. Their native structure is readable despite residual grain and optical
softness; these diagnostics remain unresolved. All seven archives are verified with
finite raw images and current provenance sidecars. See the
[portable gallery record](../../../benchmarks/results/20261010_gallery.json).
The Naked eye moved
122.829 mm laterally after CPU rays
proved its old sightline hit the housing; optics and exposure remain unchanged.
See [sightline evidence](../../../benchmarks/results/20261010_gregorian_sightline.json) and
[current milestone status](NEXT_ITERATION.md).

## Historical measurement update — 2026-10-09

All seven planned fixed benchmark sweeps completed with exit code zero. Canonical historical/
current results, both reference seed pairs and retained production diagnostics are summarized
in [PERFORMANCE_STATUS.md](PERFORMANCE_STATUS.md), with a portable JSON record under
`benchmarks/results/20261009_corrected_checkpoint.json`. This supersedes pending-capture
wording below. The 4× target is unmet. No new final gallery captures or promotion occurred;
The notes below retain the earlier measurement inputs; the current section above
supersedes their scene-adoption and gallery status.


This is the state before new final captures and art changes. The production
benchmark remains eight canonical views at 256×192 and four fixed scan views at
960×600 from `benchmarks/suites.json`. Gallery views can differ from benchmark
views without changing that workload.

## Historical build and fair comparison

The annotated `v1.0` tag resolves to commit
`20ff59a9afe0d111d5356bfa25caea3fe1fdf121`. An isolated `git archive` source tree is
configured successfully, offline, at `out/performance_20261008/v1_source/build`.
Its `historical-build.json` and `configuration.log` retain the revision and command.
The configuration uses Release, GPU on, viewer off, GCC 16.2.1 and existing pinned
Slang 2026.18.3, Vulkan headers/volk 1.4.357.0, stb and cgltf dependencies. The isolated historical CLI then built successfully with `-j1` in359.1seconds.
Its SHA-256 is `f00b549d96ca54828c6c2c600c41628c80ec2cca560956b454179a3cdb1b3107`;
the monitored minimum available host memory was14.79GiB. This establishes a
reproducible executable, not a historical performance result. The eight canonical
inputs pass a feature/file compatibility audit; measurements against corrected
common references were still pending at this build audit; the completed historical
measurements are recorded in the dated update above.

The historical material loader lacks image normal/roughness maps, and its
conductor branch ignores base-colour textures. The current three museum scenes
use those features. The historical image representation also predates the default
texture budget and can exhaust host memory with the full-resolution assets.
Therefore a successful historical render of the current scan file still does not
prove the same expected image. A saved pre-budget Kunstkammer review was reportedly
killed at 20.6 GB; this is a reported historical incident, not a new measured v1
benchmark. Do not deliberately reproduce uncontrolled host OOM.

Compare historical canonical cases only where features, materials, settings and
expected image agree. For incompatible scans, report the specific unsupported
features and any safely bounded failure; do not assign infinite speedup or remove
the case from a purported full-suite geometric mean. A compatibility derivative
that disables maps or reduces inputs needs a separate named experiment and cannot
certify the fixed current production workload. The current-feature frozen baseline
supports forward optimization measurement, but it does not redefine the original
fourfold target relative to v1.0.

## Observatory cabinet correctness repair

Contact validation exposed an actual overlap between the generated cabinet shelves
and stiles, including coincident material backfaces. The shelf width in
`tools/build_scene_models.py` changed from 1.42 m to 1.355 m to fit between the stile
inner faces at x=-2.7775 and -1.4225. Only
`models/crafted/the_observatory_Walnut_panelling_1.obj` was regenerated. This is an
asset correctness repair, not an engine speedup or a change in art direction.

| Asset version | SHA-256 |
|---|---|
| Original generated mesh | `cfea8b0586aec93d6ad80596240d0b82217339cab632d55334c76dd729f6b6ef` |
| Repaired mesh | `d6ecfec8b29fc13972fe2ab8ff279b992fd414aa2635fd574e493f3f639220aa` |

Both versions are preserved under `out/performance_20261008/asset_repairs/`, with
the reason and hashes in `observatory_walnut1_repair.json`. The archived original
matches the deterministic generator's previous output. The external OBJ diff
changes 128 vertex-position lines; face topology and other records are unchanged.

The new `observatory_saturn` comparison must use the identical repaired mesh for
both historical v1.0 and the corrected binary, loaded from the current repository
root. Its earlier full-suite capture retains the old asset identity and is
incompatible with this repaired workload: preserve it as historical evidence,
without reusing its reference or rescoring its raw image into the new comparison.
Capture fresh common references and measurements for both binaries. The other
canonical cases are unaffected by this mesh repair.

## Gregorian trim and Window joinery correctness repairs

The Gregorian's left/right architrave height changed from 1.20 m to 1.15 m,
with centre height 1.47 m to 1.445 m. Their bottom remains 0.87 m; their top now
meets the header underside at 2.02 m. This removes a verified trim overlap while
preserving the instrument optics, source powers, Moon map and detector framing.

The Invisible Window's doorway jambs now stop below the lintel: height 1.78 m,
centre height 0.89 m. The lintel spans the outer jamb faces with width
1.365616480 m and centre X 0.622808240 m. The lower retaining slip centre changed
from 0.650390608 m to 0.6503906076 m to remove a 0.4 nm overlap. The panel, Sun,
heliostat, receiving surface, materials and detector framing remain unchanged.
Art/material/framing variants under `out/curation_20261008/` remain separate.

| Main scene version | SHA-256 |
|---|---|
| Gregorian before trim fit | `c6adb7f264b853d8ec5898241c11b1c8f6d8f2c001b8780e53488e4f560aaeec` |
| Gregorian after trim fit | `947a5f369e16f4701acde5f9614fb9324c484a98b212303417bd7fbba0001868` |
| Window before joinery fit | `94d96761b3b71d3250f454e13f676d7e7a46433e09c83a110aa33f21d69d7e46` |
| Window after joinery fit | `2ede8a627c87d03c85946f8e577a9652bfe29e524fb29d89021a587084a03c36` |

Original and repaired scene snapshots, narrow diffs and repair records are in
`out/performance_20261008/asset_repairs/`. The original scene SHA-256 values
match their archived full-suite reference identities. Preserve those captures
as historical evidence; their raw images and references cannot be reused or
rescored as the repaired workloads. New common references and all comparable
binary runs must load the identical repaired inputs. This does not resolve the
historical v1.0 scan feature incompatibility or establish a speedup.

Root's four 960×600, 16-spp path previews each trace 9,216,000 paths, with zero
leaks and 1–11 remaining inconsistencies. The Gregorian preview uses Eyepiece;
the Window preview uses Room from a curation derivative, so these are not the
fixed production benchmark views. The remaining previews use Kunstkammer Room
and lamp Room. The [pilot record](../../../benchmarks/evidence/performance_20261008/production_collinear_gpu_pilots.json)
retains the diagnostics and visual notes. They ran during host builds, before
the final endpoint corrections, and establish neither final gallery acceptance
nor controlled performance.

## Historical Kunstkammer candidate and staged 512-spp reviews

The frozen native-owner path previews under
`out/gallery/path_contact_review_20261009/` are provisional and unpromoted.
Each view traced 294,912,000 paths at 512 spp, with zero leaks. Kunstkammer Room
reported 411 inconsistencies, first at LeftSillWall; Gregorian Beside reported
44, first at WestWall; Gregorian Eyepiece reported 72, first carrying L2 glass
state into RamsdenScan. Native visual notes and diagnostics are retained in
[the staged review](../../../benchmarks/evidence/gallery/path_contact_review_20261009/native_reviews.json).
These captures ran during host compilation and are not timing comparisons or
final aesthetic acceptance.

A static audit confirmed positive-volume Kunstkammer window overlaps between
rails/jambs, plaster backing and the limestone sill. The separate
[fitted candidate audit](../../../benchmarks/evidence/performance_20261008/kunstkammer_window_joinery_review_20261009/README.md)
partitions the backing around the original stone cap, clips buried rail/jamb
overlaps and moves Bead2 outward by 3 mm. It preserves the aperture, sources,
materials, detectors and exposed sill. The exact-decimal scoped box audit finds
17 original overlap pairs and none in the candidate; this is not a render or
visual validation. Other scene geometry remains outside that audit.

The fixed benchmark Kunstkammer input had SHA-256
`235e033276e93746eb3b2b357782f45df77a656113ebee564e408eb409484c81`.
The separate candidate was
`out/performance_20261008/kunstkammer_window_joinery_review_20261009/the_kunstkammer_joinery_candidate.owe`,
SHA-256 `3a40b61275e5bf85acc849b00d6cc6e0190c3cbc0a61b1fe109d91bbc30abfe6`.
The benchmark sweeps retained that fixed input and its diagnostics. The fitted joinery
has since been adopted with portable paths; its adopted source SHA-256 is
`54546f437edbd173e834f2b3f84b3148a2bf01d0913bb02670334bd38c27e424`.
This is an authored source identity, not a promise that downloaded or reconstructed
assets reproduce historical bytes. Final gallery images require fresh source/asset/entry
identities and archives and cannot substitute for the historical benchmark inputs.

Integrated candidate validation was pending when this audit was written. The isolated
native-owner six-test hardware proof did not replace that step. Current engine validation
and gallery status are tracked in [NEXT_ITERATION.md](NEXT_ITERATION.md).

## Existing final-view evidence

All listed originals retain raw PFM alongside PNG/JSON under `out/scenes/`.
Native scene hashes cover source text, not external assets or engine version;
matching them alone is insufficient for reference/cache reuse.

| Scene | Existing views | Archived native source hash | Remaining completion work |
|---|---|---|---|
| Kunstkammer | Room 2048; Table/Armour 2048; Book 1024, GPU path | Room: `469db8305ef4ad57`; details: older `85cb82c8264de249` | Earlier room review accepted the daylight composition. New 512-spp provisional capture has zero leaks but 411 inconsistencies; validate the separate window-fit candidate after fixed benchmark captures. Refresh details after sky change. Preserve scans and curated maps. |
| Peacock lamp | Room/Shade 512, GPU VCM | Both: `056e69c67cc6642c` | Prior review accepted both, zero leaks. Review a new engine capture before promotion; imported smooth normals are intentional and original file normals reproduce a known dark-patch defect. |
| Gregorian | Beside 2048 path; Eyepiece 256 VCM; Naked 64 VCM | Beside: `8a3e0a49011fffeb`; optical views: older `dd166cbb2d64c660` | Prior review explicitly provisional: bright room speckles, one room leak, spectral grain and central bright optical ghost. The Naked view is only a true-angular-scale comparison, not the art hero. Reassess estimator and contact correctness before final imagery. |
| Invisible window | Room 128, GPU SPPM, 16 photons/pixel/iteration | `db33fb0629e7cd7d` | Broader art/material pass unfinished; 253 reported leaks, first inconsistency WallR.surface. Room shows the complete caustic/window context; benchmark Wall is an intentional oblique detail view. Preserve the frozen PMMA product and diagnose contacts/convergence separately. Photo clearance unresolved. |

The Gregorian keeps its original optics, 38° optical field and real mapped Moon.
Its scan's posed height disagrees with one museum catalogue dimension; the review
records this rather than applying nonuniform deformation. The Wawel chest's
independent dimensions are unverified. These are attribution/scale limitations,
not permission to resize an instrument or change its prescription for a benchmark.

## Historical reference and completion sequence

1. Finish immutable full-suite baseline and candidate capture before art edits.
   Keep logs, failed cases and all raw reference images. Reuse the earlier bounded
   lens/prism/lamp references only when their frozen measurement identities match.
2. Raise the inexpensive canonical reference budget to 8192 and capture an
   independent seed. Check reference-to-reference full-image relative MSE, mean
   luminance and relevant spatial blocks; rare bright paths can make 256-sample
   references misleading. Retain nonmonotonic errors rather than extrapolating.
3. For scan references, inspect 256 first and raise only the cases whose finite
   reference variance changes the time-to-error conclusion: staged 512/1024/2048
   path references and 512/1024 VCM or SPPM references, plus an independent seed.
   Estimate costs from the isolated current measurements. Existing shared-GPU
   final-view timings are planning evidence, not controlled benchmark timings.
   No budget by itself establishes convergence; require reference disagreements
   comfortably below the candidate error being compared, including blocks and
   mean luminance. If this condition fails, mark the equal-error result provisional.
4. Use only observed budgets meeting a fixed error target against the same
   independent reference. Report transport and process wall comparisons, peak
   host RSS, incomplete targets and full-suite geometric mean separately.
   A single-seed finite-reference result cannot certify a fourfold expected-quality
   claim; SPPM/VCM merging bias and spectral convergence also need examination.
5. Diagnose the Gregorian grain/ghost and invisible-window contacts before art
   changes. Then render the four hero views and optional details through the
   gallery staging workflow, open native images, and document the visual/physical
   review. Do not denoise, paint over artifacts or regenerate the frozen caustic
   product as part of routine scene iteration.
6. Promote reviewed local outputs with their metadata and credits;
   add the new README gallery links and finish scene notes. The target-photo
   provenance investigation is set aside at the user's direction; local work
   continues with unknown rights honestly recorded. Local evidence establishes
   only that they supplied `cristo-vidriera.jpg` on 6 October 2026, not its licence.

The old gallery contains eleven PNG/JSON pairs; its legacy render list contains
thirteen entries, including absent `the_shallows` and `study_eyepiece`. The new
[manifest and archive workflow](../../GALLERY_WORKFLOW.md) retain those default selections,
add explicit scan views and preserve all existing raw evidence.
