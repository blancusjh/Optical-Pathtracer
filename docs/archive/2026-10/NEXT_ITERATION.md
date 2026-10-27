Historical record, retained for provenance. Current state: [STATUS.md](../../STATUS.md)

# Next iteration

Written 2026-10-06, after version 1.0. Two goals:
- **scenes worth looking at**;
- **a 4× gain in rendering performance**, measured as time to equal quality.

Engine features come second to these two.

## Gallery and engine milestone — completed 2026-10-10

The completed deliverable is the reviewed seven-view gallery plus tested engine improvements.
The full 4× equal-quality target is follow-up work, not a gate for this milestone.
The earlier correctness/measurement checkpoint alone did not close this work.

Contact traversal now reuses an already-proved first interval, walks native mesh BVHs
without a private stack, and shares the analytic solver. Negative-distance slab bounds
and GPU finite-endpoint ambiguity handling are corrected. The final implementation passes 157/157 hardware and 21/21 targeted software tests,
with four viewer/layering checks. Two frozen-executable repeats measure 2.258× Window
transport, 1.319–1.327× Saturn, 1.094–1.095× lens and 1.069–1.071× Peacock against the
corrected checkpoint; see [recorded results](../../../benchmarks/results/20261009_gallery_engine.json).
The earlier 25/25 hardware and 21/21 software results below belong to the prior checkpoint.

The final gallery comprises Kunstkammer Room; Gregorian Beside, Eyepiece and Naked;
Peacock Room; and Invisible Window Room and Wall. Fitted Kunstkammer joinery is adopted,
and the Gregorian capture settings preserve the optical prescription and common field.
Window materials/framing are adopted with its physical optical system unchanged.
All seven views—Kunstkammer Room, Peacock Room, Gregorian Beside/Eyepiece/Naked and
Window Room/Wall—are locally promoted after native review. The Gregorian set was recaptured after moving only the Naked
eye 122.829 mm sideways: CPU rays proved its old position hit the scanned housing,
and fresh path tracing now shows the true-scale Moon. See the
[sightline proof](../../../benchmarks/results/20261010_gregorian_sightline.json).
Window Room SPPM 64 with 16 photons is reviewed and promoted: its complete panel and
projection fit, with readable plaster, timber and floor joints despite residual grain.
Window Wall SPPM 128 is also reviewed and promoted: the whole oblique lancet, portrait,
hand, garment and ribbon structure are readable despite optical softness and grain;
lettering is not legible. Room retains 168 inconsistencies and 21 leaks, Wall 243 and 34.
The [portable gallery record](../../../benchmarks/results/20261010_gallery.json) retains all
seven capture identities, actual statistics and reviews. Raw images are finite and
archives verified. No further capture is required for this milestone.
Every promotion retains diagnostics, raw archive identity and a review sidecar.

The seven historical benchmark sweeps remain complete, including independent reference
pairs. Their [results](PERFORMANCE_STATUS.md) and
[portable record](../../../benchmarks/results/20261009_corrected_checkpoint.json) preserve the
previous regression and noisy-reference limitations. They are not measurements of the
final optimization or of subsequent gallery scene revisions.

## Historical checkpoint status — 2026-10-09

This section records the earlier corrected checkpoint, before the contact performance
changes and adopted gallery inputs. Its validation counts, pending proposals and scene
status are historical; the milestone section above owns current status.
The baseline and original diagnosis below describe v1.0; they are retained as history.
The byte-backed texture storage, bounded sequential image loading/upload and GPU normal/roughness
maps are now implemented. Curated environment assets have been downloaded and the four scan
scenes have substantial local revisions. October 7 Kunstkammer notes report a 960×600 render at
about 4.4 GB peak host memory. Memory and map support are no longer the next implementation tasks.

At that checkpoint, the broader objective was **fast, verified, beautiful optical worlds**.
The following records progress and outstanding work at that date:

1. **Implemented and previously validated:** transactional VCM cache overflow recovery,
   preserving the guide and sampling schedule. Forced overflow and bounded-allocation recovery
   tests passed with both hardware and software traversal on the RTX 4060. The earlier
   integrated geometry/VCM checkpoint passed 7/7 in each mode, including textured diffuse
   agreement with path tracing; this predates the contact/endpoint changes now under validation.
2. **Full twelve-case baseline captured:** eight canonical views at 256×192 and four scan views
   at 960×600 completed with a frozen integrated executable, references at 256 spp and
   measurements at 16/64 spp. Raw images, logs, scene/asset identities, wall/transport timing
   and host RSS are retained; see [BENCHMARK_BASELINE_20261008.md](BENCHMARK_BASELINE_20261008.md).
   These references have finite noise and unresolved transport diagnostics, so this completed
   capture is not converged truth or a full-suite equal-quality result. The runner supports
   frozen references and optional GPU timestamps; see [BENCHMARKS.md](../../BENCHMARKS.md).
   A first 960×600 Peacock profile attributes 79.15% of kernel time to camera paths and 20.76%
   to light paths. Its unprofiled render took 18.393 s for 16 spp. This is a current-scene
   diagnostic, not a v1.0 speedup comparison; see [PERFORMANCE_STATUS.md](PERFORMANCE_STATUS.md).
3. **Measured camera optimizations:** cache the diffuse spectral response within each VCM camera
   hit. The earlier combined checkpoint, including the planetary fix, reduces a 960×600 Peacock
   16-spp render from 18.3959 s to 16.1294 s with profiling disabled (1.141×, about 12% less
   transport time), with identical accepted path/segment counts and similar raw error against
   the same 256-spp reference. The isolated camera patch measured 15.501 s with profiling on;
   details and limitations are in [PERFORMANCE_STATUS.md](PERFORMANCE_STATUS.md). This
   single-scene comparison is not a full-suite equal-quality result. A separate CIE cache now
   reuses the packet's colour-matching values while retaining its current wavelength PDFs.
   The unprofiled paired 64-spp lamp capture fell from 55.2653 s to 53.2466 s: **3.65% less
   transport time**, relative to the newer integrated baseline. These are bounded measurements
   with different checkpoints, not additive percentages or a demonstrated 4× result. Further
   choices must follow measurements; wavefront decomposition, specialization, GPU film
   accumulation and asynchronous submissions remain candidates.
4. **Planetary precision, contacts and finite endpoints validated at the targeted checkpoint:** near-surface
   sphere intersections
   retain small clearances in CPU arithmetic and GPU pole-relative coordinates. GPU near and
   distant hits reconstruct on the same full analytic sphere; hidden-sheet, far-ray illumination,
   and observer-rebasing regressions passed at the earlier hardware/software checkpoint. The planetary
   fixture's inconsistencies drop from 6,206 on CPU and 149 on GPU to zero on CPU and both GPU
   traversal modes, with matching final path/segment counts and displayed mean brightness. Small
   [regression fixtures](../../../benchmarks/regressions/README.md) preserve the original observations
   and controls. New corrections retain contact anchors, primitive identity and region ordering,
   and trace finite detector connections without skipping near-observer or source-end blockers.
   The current [CPU-only checkpoint](../../../benchmarks/evidence/performance_20261008/contact_endpoint_cpu_full_tests.log)
   passes **157/157 registered tests**, with GPU transport unavailable/skipped. Curved roots,
   mesh hit residuals, thin-film attenuation, sensor gaps and finite-light endpoint controls pass.
   An earlier full hardware run passed 156/157: smooth glass retained five region inconsistencies.
   Native-coordinate replay traced that failure to querying a departing mesh owner with the
   lagged world ray instead of its corrected contact anchor. A complete native owner query
   passes the unchanged smooth-glass assertion with zero inconsistencies and five focused
   controls. The integrated hardware build passes 25/25 targeted checks, including all eight
   VCM checks, finite endpoints and planetary geometry. Software traversal passes 21/21 targeted
   checks. The bounded software SPPM smoke completes all 16 passes with zero inconsistencies
   or leaks (162.05 s cold-inclusive wall, 1.53 s reported transport). The long software
   smooth-glass diagnostic was terminated without a conformance result; a bounded smoke
   capture is recorded separately. This coverage is not a second full-suite run. Do not silence diagnostics or change the frozen caustic panel to
   hide failures. The executable and source record are now
   [frozen](../../../benchmarks/evidence/performance_20261008/corrected_final_manifest_20261009.json), and corrected
   reference captures completed; see the closure record above.
5. **Historical comparison prepared:** the exact `v1.0` revision now builds offline in isolation.
   All eight current canonical inputs pass the feature/file audit; their historical/corrected
   comparison still needs common corrected 8192-spp references and an independent seed check.
   The historical loader lacks the museum scans' normal/roughness maps and conductor colour
   textures. Those inputs cannot establish equal expected images, even if a render finishes;
   reported historical OOM is separate evidence. Unsupported scans remain explicit and cannot
   be removed from a purported full-suite 4× geometric mean. See
   [SCENE_READINESS_20261008.md](SCENE_READINESS_20261008.md).
6. **Gallery completion pending:** complete physical and artistic scene reviews. Peacock Lamp
   has reviewed finals; Kunstkammer's
   Room has a newer daylight render than its close-ups. Gregorian remains provisional and the
   invisible window has unresolved transport diagnostics and an unfinished broader material pass.
   The user deferred the photograph-provenance
   investigation; local rendering, review and promotion continue with unknown rights honestly
   recorded. The [gallery workflow](../../GALLERY_WORKFLOW.md) stages explicit scan views, preserves
   raw images and requires a review note before local promotion. The refreshed four-scene
   gallery has not been accepted. A file named `final` is not sufficient acceptance.

Use the same scene/assets, detector, resolution, texture budget, device/driver and fixed reference
images on both sides of a performance comparison. Report load/setup/export overhead separately
from transport and record peak host RSS. Changed art direction must not count as an engine speedup.
Retain unmasked raw-image error as the primary metric; masked glint error is a diagnostic only.
An absent or out-of-memory baseline is an unresolved comparison, not an infinite speedup. If
v1.0 cannot render a member, report that separately and establish a feasible baseline for that
member without claiming it proves the original full-suite 4× target.

The broader capability sequence and project invariants are in [ROADMAP.md](ROADMAP.md).

## 1. Performance: 4×

### Historical baseline (RTX 4060 laptop GPU, v1.0)

| Measure | v1.0 |
|---|---|
| `owe bench --backend gpu` (canonical suite, 256×192, 16 spp) | 6.37 Mpaths/s, 0.987 s |
| `the_peacock_lamp`, 960×600, 16 VCM iterations | 123 s (7.7 s per iteration) |
| `the_gregorian`, 960×600, 16 iterations | 24 s |
| `the_kunstkammer`, 960×600 | **killed by the system (out of memory)** |
| Cold start of VCM (no caches) | 61 s, the largest kernel |

The original estimate was 256–1024 iterations and half an hour or more for the lamp.
These historical timings and budget estimates are not current controlled measurements;
no sample budget alone establishes convergence.

### Target and how it is measured

- **Benchmark set:** the canonical suite plus the four scan scenes at 960×600.
- **Time to equal error:** time to reach the same relMSE against a long reference render, per scene. Raw Mpaths/s is reported too, but it is not the target: an estimator that is faster and noisier does not count.
- **Correctness:** the expected value must not change. The statistical agreement tests (`compare`, the conformance suite) must still pass. Image digests may change.
- **Goal:** the geometric mean over the set is at least 4× better than v1.0, and every scene renders at production resolution.

### Original optimization candidates (measure before choosing)

1. **Memory first** (the Kunstkammer crash).
   - `ImageRGB` keeps every 8-bit texture twice on the host: linear float RGB (12 B per texel) and the sRGB bytes (4 B per texel). Six 8192² and six 4096² textures come to about 8 GB on the host alone, before the flattened copy and the GPU buffer.
   - Keep only the sRGB bytes and decode with a lookup table where the CPU needs a texel.
   - Free the host copies once uploaded.
   - Consider BC7 compression or mip levels on the GPU.
2. **Profile before optimizing.**
   - An Nsight trace per kernel.
   - `OWE_GPU_STATS=1` for registers and spills.
   - A per-stage timer in the renderer (traversal, shading, NEE, VCM cache and grid, readback).
   - Decide the next items from the measurements, not from this list.
3. **Wavefront instead of megakernels** (Laine, Karras & Aila 2013).
   - The kernels are 1.1–2.4 MB of SPIR-V, and their register pressure limits occupancy.
   - Split each integrator into stages joined by queues: generate, intersect, shade and sample, shadow rays, media. Sort or bin by material where it pays.
   - This usually gives the largest single gain, and smaller kernels also cut compile time.
4. **Specialization per scene.**
   - Turn off the features a scene does not use (media, relief, stained glass, image textures, material maps) with specialization constants.
   - Mark rare paths `[noinline]`.
5. **Host overhead.**
   - Accumulate films on the GPU, and read back only when the viewer or the output needs an image (every pass today copies the whole pixel state and adds it on the CPU).
   - Replace the wait after each submission with double-buffered command buffers and a timeline semaphore.
6. **Texture bandwidth.** The lamp's 8192² atlases are a suspect for its 7.7 s per iteration. Use mip selection from ray footprints, or compressed formats.
7. **Acceleration structures.**
   - Merge many small instances into fewer bottom-level structures, and compact them.
   - Analytic optical surfaces stay analytic (the guiding rule), but purely decorative primitives may be triangulated for hardware traversal.
8. **VCM costs.**
   - Light vertex cache size, the per-iteration grid build by sorting, and the number of light subpaths per pixel.
   - Verify the "VCM cache drops vertices" report from the invisible-window agent.
9. **Cold start.** Find what inflates the largest kernel (unrolled hero-wavelength loops, BSDF and texture switches inlined at every call site). Items 3 and 4 address most of it.

Denoising does not count toward the 4×. Measurements stay unfiltered by the guiding rule; a denoiser could only be a labelled preview in the viewer, if ever.

## 2. Scenes worth looking at

### Historical scene diagnosis (v1.0)

- **The Kunstkammer leaves its hero, the armour, in a dark corner.** The walls are too flat and the window is plain.
- **No curated environment assets were downloaded.** The asset shortlist of 2026-09-28 ("Asset Shortlist for the Engine": Poly Haven plaster, wood floor, oak, marble, fabric, book sets, a lantern, a bust) was never fetched. Only the hero scans were.
- **The engine reads no glTF normal or roughness maps,** so even good texture sets would render flat.
- **Light was not designed around the hero of each scene.**
- **Review was blind.** The agents judged 320×200 noisy previews, some without a GPU in their sandbox. My review checked that scenes rendered, not how they looked.

### Original scene plan (implementation progress is recorded above)

1. **Engine:**
   - read glTF `normalTexture` and `metallicRoughnessTexture`, and Poly Haven normal and roughness maps for primitives;
   - use triplanar or uv mapping with a real tangent frame.
2. **Assets:**
   - fetch the Poly Haven texture sets and objects from the shortlist;
   - add period interior scans (panelling, furniture, historic interiors), downloaded through the user's logged-in browser as before;
   - write each into `assets/sketchfab/MANIFEST.json` with its licence.
3. **Light design:**
   - for each scene, decide where the hero is and how light reaches it before building anything (Vermeer's window, Wright of Derby's single source);
   - the Kunstkammer's sun must fall on the armour.
4. **Review protocol:**
   - previews at 960×600 or larger, run long enough to read (possible once §1 lands);
   - the agent writes three critiques per preview;
   - I judge each scene's aesthetics before calling it reviewed, against its references.

## 3. Original order of work (superseded by the status update)

1. The memory fix, so the Kunstkammer renders at production resolution.
2. Profiling and the baseline per stage.
3. Wavefront kernels, specialization constants and host overhead, until the 4× target is met on the benchmark set.
4. glTF normal and roughness maps.
5. Curated assets, then the scenes rebuilt with designed light, reviewed by the protocol above.

The engine work remaining from v1.0 is in `docs/ROADMAP.md`: stained-glass roughness, folded mirrors, volume vertices in VCM, sun extraction, star maps, heterogeneous media, manifold NEE. It comes after these.
