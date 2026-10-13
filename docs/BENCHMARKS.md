# Reproducible rendering benchmarks

`tools/run_benchmarks.py` measures raw linear PFM images against a longer independent-seed
reference. Full-image relative MSE is the performance target. Display exposure and tone mapping
have no effect on this metric; no denoising is applied. NumPy is the only Python dependency.

The original command still selects the twelve transport scenes in `benchmarks/`, with their
scene-selected detector/reference estimator and all five transport estimators. The additional
suites are explicit in `benchmarks/suites.json`:

| suite | cases | resolution | default candidate estimator |
|---|---|---|---|
| `transport` | twelve estimator transport tests | original scene resolution | scene estimator plus SPPM, VCM, BDPT, hybrid and path |
| `canonical` | eight scene/detector pairs used by `owe bench` | 256×192 | path |
| `scans` | Kunstkammer/Room, Gregorian/Beside, Peacock lamp/Room, invisible window/Wall | 960×600 | path, path, VCM, SPPM respectively |
| `performance` | canonical plus scans | as above | as above |

`--include-scans` also appends the scan cases to another suite. `--only` selects manifest case
names; an unknown name is an error. `--list` (also `--dry-run`) prints selected settings and
budgets without opening assets or running the renderer. Scan detector, estimator, depth and
photon count are fixed in the manifest. The original scene files supply other settings and are
never edited. Canonical depth is fixed to 64. Seeds default to 1 for candidates and 1000 for
references; passes default to 1; CPU thread count 0 means hardware concurrency. Use explicit
thread/device choices for a baseline. `--resolution` overrides resolution; nonstandard canonical
or scan resolutions are ineligible for the fixed production goal. `--quick` reduces budgets to 16/64 and reference
samples to 256, **without reducing resolution**. It overrides `--reference-spp`; explicit
`--budgets` overrides the quick candidate budgets.

List the production cases first:

```sh
python3 tools/run_benchmarks.py --suite performance --list
```

Capture a baseline using a preserved baseline executable and the exact scene/assets that will
be compared. This is a long, deliberately opt-in workload:

```sh
python3 tools/run_benchmarks.py --suite performance --binary /path/to/v1/owe \
  --backend gpu --device 0 --output out/baseline-v1 \
  --reference-spp 8192 --budgets 16 64 256 1024
```

Compare a candidate executable against **the same frozen reference images**:

```sh
python3 tools/run_benchmarks.py --suite performance --binary build/owe \
  --backend gpu --device 0 --output out/candidate \
  --reference-dir out/baseline-v1 --reference-spp 8192 \
  --budgets 16 64 256 1024 --baseline out/baseline-v1/summary.json
```

`--reference-integrator` overrides the reference estimator; `--integrators` selects candidate
estimators. Unsupported estimator fallbacks are failures, rather than mislabeled measurements.
A frozen reference directory is read only: a missing or incompatible reference is reported and
never regenerated there. Reuse requires matching reference estimator, sample count, seed, scene,
assets, render settings and host/device environment. It intentionally permits a different
executable/source hash for a frozen reference, so an optimized build can be measured against the
baseline image. Preserve the baseline binary and summaries; the historical figures in
[the archived checkpoint](archive/2026-10/NEXT_ITERATION.md) alone cannot reconstruct a full equal-error baseline.

An inexpensive CPU command checks the runner end to end:

```sh
python3 tools/run_benchmarks.py --suite canonical --only the_lens --backend cpu \
  --threads 1 --resolution 4x4 --reference-spp 2 --budgets 1 2 --output /tmp/owe-bench-smoke
```

## Equal-error comparison

For each baseline scene, the default target is its smallest finite full-image relMSE over the
recorded runs. `--target-error E` uses an explicit common threshold instead. Baseline and
candidate times are each the fastest **observed** run meeting that threshold. There is no curve
extrapolation or interpolation, so a candidate that has not reached the target is explicitly
`candidate_target_missing`. Add larger budgets when needed. This distinguishes time to equal
error from equal samples or raw throughput. `--time wall` compares whole-process elapsed time;
the default compares the renderer's transport time.

`comparison.json` records per-scene targets, times, ratios and status. Its geometric mean is
available only when every case in the union of baseline and candidate selections matches.
An additional partial geometric mean is diagnostic and cannot satisfy the goal. `goal_met`
requires at least the requested speedup (default 4), all twelve manifest scene/detector pairs,
every canonical view at 256×192 and scan view at 960×600, compatible measurement identities,
and profiling disabled. A matched subset
can still report its geometric mean but cannot certify the production goal. Missing references,
failed processes (including OOM kills), changed assets/settings, unreached error targets and
legacy summaries remain visible. Rendering failures produce a nonzero runner exit status;
an incomplete speedup comparison is reported as data and does not itself change exit status.

The reference is a finite Monte Carlo estimate, not exact ground truth. These are single-seed
measurements, with no confidence intervals or repeated timing distributions. A production claim
needs adequate reference convergence and repeated independent seeds, saved in separate output
directories. Run the engine's statistical agreement/conformance checks as well: agreement with
one finite image does not establish unchanged expected value. Full-image relMSE is
`mean((image-reference)^2 / (reference^2 + 0.01*mean(reference)^2))` over pixels and channels.
The mean ratio and a secondary glint-masked metric help diagnose failures; they do not replace
the full-image target. The glint mask removes a four-pixel neighbourhood around reference
pixels exceeding 20× their 9×9 median and 10× the image mean, unless that mask contains at least
half of the reference's maximum-channel light. A completely black reference with a nonblack
candidate has an unavailable relative error rather than a fabricated finite score.

## Records, timing and safe reuse

Each output case has native `.pfm`, `.png` and `.json` render artifacts, stdout/stderr logs,
and a `.benchmark.json` sidecar. The sidecar persists process wall time and host peak RSS, which
were previously lost on rescoring. `--rescore` reuses a render only when its sidecar identity and
both PFM/native JSON content hashes match; it renders missing/stale outputs. Existing artifacts
without a sidecar are never accepted as a cache hit. A different code/binary invalidates normal
reuse. It is safest to use a separate output directory for each build.

`code.source_scope` is `capture_workspace`: `source_sha256` and `git_head` describe
the checkout observed at capture time, including current source files, build definition,
runner and suite manifest. They do **not** establish which source built a frozen executable.
`binary_sha256` identifies the implementation actually invoked. Retain a separate build
revision/configuration manifest for historical or externally frozen binaries. The prepared
v1.0 comparison links and hashes `v1_source/historical-build.json`; its archived revision
is separate from the current capture workspace. Older sidecars without this label remain
historical evidence and cannot satisfy the new strict candidate-cache identity.

Identities include SHA-256 of the executable and repository sources, git HEAD, scene text,
recursive external glTF/GLB buffers and images, OBJ material references, lens prescriptions,
and other quoted scene file paths. Missing dependencies are recorded explicitly, so adding an
optional asset invalidates reuse. Binary/image assets are streamed through the hash; hashes
are memoized only within an invocation. Keep sources, executable and assets unchanged during
an invocation. Hashing scan assets can take time and is outside the
per-render timer. Self-contained GLB content is covered by its file hash. Asset discovery
covers the current file-based scene syntax; new loader dependency formats must extend it.
Identities also capture platform, CPU thread availability, `owe backends` inventory, selected
backend/device, and `OWE_*`, `VK_*`, `VULKAN_*`, `CUDA_*`, `OMP_NUM_THREADS`, and `SLANG_DIR`
environment values. A read-only `nvidia-smi` query, when available, additionally records GPU
UUID, PCI bus, model and NVIDIA driver version; unavailable inventory is explicit. Vulkan API
version alone does not identify the installed NVIDIA driver build. This deliberately refuses cross-host/environment comparisons, including
changed texture budgets that could change the image. Keep those settings identical.

Each run reports:

- `seconds`: native `render_seconds`, time accumulated by the transport renderer;
- `wall_seconds`: elapsed time around the fresh renderer process;
- `non_transport_seconds`: wall minus transport, clamped at zero;
- `peak_host_rss_bytes`: child-process maximum RSS, or null when unavailable;
- paths, Mpaths/s, region inconsistencies, leaks, full/masked relMSE and mean/reference ratio.

Non-transport time includes scene loading, renderer setup, pipeline compilation, GPU upload,
resolve and output writing. It is **not an isolated scene-load timer**. GNU `time` is preferred;
when unavailable on Linux/macOS, a fresh Python helper launches exactly one renderer and reads
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`. A separate helper per run prevents an earlier
child's peak from contaminating later measurements. Linux reports KiB and macOS reports bytes;
other systems without a supported method report null. `rss_method` identifies the method.
Either wrapper adds process-launch overhead to wall measurements. Host RSS excludes the Python
scoring/helper process and is not GPU VRAM usage. Failed runs retain wall/RSS measurements where
available and log paths.
For cold-start work, use a fresh external pipeline cache directory and a separate output
experiment; normal speed comparisons should use the same cache policy and warmed shader cache.
The runner does not clear caches, control power/temperature, or claim kernel compilation is
warm. Record such experimental controls alongside saved outputs.

## GPU profiling

`OWE_GPU_PROFILE=1` enables an opt-in renderer timestamp diagnostic. Stderr lines beginning
`owe_gpu_profile` are retained in logs and copied into run records. Per-kernel records aggregate
GPU dispatch durations; per-submission records aggregate GPU durations including transfers and
CPU time around `vkQueueSubmit` plus its fence wait. Submission CPU time excludes command
recording and host readback processing. Kernels are nested inside submissions, so **do not sum
kernel and submission categories**. Traversal, shading and NEE inside a megakernel are not
separate stage measurements. Queues that cannot supply timestamps report why profiling is
unavailable.

Profiling inserts serialization around dispatches and perturbs execution. Profile runs are
useful for selecting work, but cannot satisfy the 4× throughput goal; run the baseline/candidate
comparisons with profiling disabled. The profiling environment participates in reuse and
measurement identity.

Run the focused Python checks with:

```sh
python3 -m unittest discover -s tools -p test_run_benchmarks.py -v
```
