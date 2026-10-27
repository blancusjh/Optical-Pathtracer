Historical record, retained for provenance. Current state: [STATUS.md](../../STATUS.md)

# Performance baselines — 2026-10-08

## Full production workload captured

A later frozen executable includes the diffuse-response cache and planetary precision fixes,
and precedes the CIE cache, contact ordering and finite-connection visibility corrections.
All twelve cases completed: eight canonical views at 256×192 and four scan views at 960×600,
each with a 256-spp reference (seed 1000) and 16/64-spp measurements (seed 1).
Executable SHA-256: `b08531232c341ba367a1b433fd4771851e944d399868b9a38455f28381abca7a`.

| Case | Estimator | Reference seconds | 16-spp seconds | 64-spp seconds | 64-spp full relMSE | 64-spp inconsistencies / leaks |
|---|---|---:|---:|---:|---:|---:|
| the_lens | path | 0.340 | 0.019 | 0.075 | 0.617421 | 5 / 2 |
| prism_in_sunlight | path | 0.323 | 0.021 | 0.082 | 0.0443613 | 6 / 4 |
| glass_of_water | path | 0.909 | 0.058 | 0.227 | 15.888 | 13 / 0 |
| study_magnifier | path | 1.881 | 0.119 | 0.471 | 0.528768 | 7 / 0 |
| the_telescope | path | 0.561 | 0.049 | 0.149 | 0.841032 | 24 / 0 |
| study_camera | path | 1.022 | 0.066 | 0.258 | 9.49023 | 6 / 0 |
| observatory_saturn | path | 1.121 | 0.098 | 0.319 | 22.1734 | 16 / 0 |
| camera_obscura | path | 0.624 | 0.040 | 0.159 | 6933.07 | 35 / 0 |
| the_kunstkammer | path | 36.964 | 2.368 | 9.291 | 0.50921 | 2042 / 0 |
| the_gregorian | path | 11.078 | 0.776 | 2.791 | 86.9917 | 1064 / 0 |
| the_peacock_lamp | vcm | 197.595 | 15.798 | 55.245 | 4.02149 | 466568 / 0 |
| the_invisible_window | sppm | 148.618 | 11.314 | 44.521 | 0.0115191 | 7177 / 49 |

These references are diagnostic, not converged truth. Rare bright paths cause severe and
sometimes nonmonotonic raw error, especially in the obscura and Gregorian views. Their
transport diagnostics also mean they cannot certify unchanged physical correctness.
Correctness repairs must first be checked against analytic/conformance oracles, then longer
references must be captured from the corrected implementation. A closer match to a known
incorrect image is not an optimization criterion.

Artifacts: [full summary](../../../benchmarks/evidence/performance_20261008/full_integrated_baseline256/summary.json)
and [invocation](../../../benchmarks/evidence/performance_20261008/full_integrated_baseline256/invocation.json).
The twelve references and twenty-four measurements retain all native metadata, raw images,
logs and workload identities. Profiling was disabled. Host builds overlapped a portion of
this capture, so process wall time and cold-start overhead are descriptive.

## Earlier three-case checkpoint
This checkpoint uses the frozen current-working-tree executable, not a reconstructed v1.0 binary. It establishes reproducible reference images for optimization experiments and makes no 4× claim. Only three of the twelve production cases are selected; each reference is a finite 256-sample estimate from seed 1000, with candidate budgets 16 and 64 from seed 1. No denoising or GPU profiling was used.

```sh
python3 tools/run_benchmarks.py --suite performance \
  --only the_lens prism_in_sunlight the_peacock_lamp \
  --binary out/performance_20261008/owe_baseline --backend gpu --device 1 \
  --reference-spp 256 --budgets 16 64 \
  --output out/performance_20261008/bounded_baseline
```

Frozen executable SHA-256: `fc909789c3b5b8dbcd3d11617a4a8ee79144f3398561c2c45f7ddeef2b589c6b`.

Host: Intel i7-12650H / 16 logical CPUs; NVIDIA GeForce RTX 4060 Laptop GPU, hardware ray queries, Vulkan API 1.4.351. NVIDIA driver **615.71.09**; UUID `GPU-827caec2-6f67-a1f6-4a68-4da0180a00c3`, PCI `00000000:01:00.0`. Vulkan device index is 1; `nvidia-smi` index is 0. These are different enumeration schemes.

The saved invocation records host/driver/environment identity. The only captured environment setting was `CUDA_PATH=/opt/cuda`; `OWE_GPU_PROFILE` was absent (disabled). Preserve that same environment for frozen-reference comparisons; explicitly adding `OWE_GPU_PROFILE=0` changes the strict environment identity. Existing shader pipeline caches were retained. No other team GPU workloads ran concurrently. Other agents edited sources and could build on the CPU while the frozen executable rendered, so whole-process startup/wall timings are descriptive rather than isolated host benchmarks.

The executable contains its compiled GPU kernels, so later shader source edits do not alter this capture. The source fingerprint is a snapshot of the workspace at invocation; the executable checksum identifies the compiled implementation actually measured. The separate `baseline_source.patch` preserves the root agent’s source checkpoint. Scene files and recursive assets were hashed before each case; sidecars retain those identities and hashes of both raw PFM and native JSON artifacts.

| Case / detector | Resolution | Estimator | Budget | Transport s | Wall s | Peak host MiB | Full relMSE | Mean/reference | Inconsistencies | Leaks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the_lens / Eye | 256x192 | path | 16 | 0.0178724 | 0.348563 | 322.148 | 0.0663009 | 0.933884 | 3 | 2 |
| the_lens / Eye | 256x192 | path | 64 | 0.0699306 | 0.410938 | 322.234 | 0.617421 | 0.933702 | 5 | 2 |
| prism_in_sunlight / Eye | 256x192 | path | 16 | 0.0193212 | 0.387734 | 321.719 | 0.178449 | 0.751501 | 5 | 0 |
| prism_in_sunlight / Eye | 256x192 | path | 64 | 0.0771207 | 0.452259 | 321.820 | 0.0443613 | 0.751501 | 6 | 4 |
| the_peacock_lamp / Room | 960x600 | vcm | 16 | 18.3959 | 40.9739 | 2621.195 | 0.935296 | 0.898889 | 116278 | 0 |
| the_peacock_lamp / Room | 960x600 | vcm | 64 | 63.2678 | 84.9762 | 2619.613 | 4.03054 | 0.886988 | 466568 | 0 |

| Reference case | Budget | Transport s | Wall s | Peak host MiB | PFM SHA-256 |
|---|---:|---:|---:|---:|---|
| the_lens | 256 | 0.331891 | 0.788856 | 322.535 | `b92afc4ef6ced75d81f4eef3c46e8565f61218fe8589bddad02a016e805d37b7` |
| prism_in_sunlight | 256 | 0.299155 | 0.673263 | 322.074 | `968e243b0a76c5241fa0eaa0f7e0d3ce0e3398464709e520d6f7835fbcf268c2` |
| the_peacock_lamp | 256 | 223.36 | 241.117 | 2654.320 | `e47006015eca65457c69f51e78cd61c4a575eb7b7fceed607b949c7c8970ae91` |

Candidate comparison using the same frozen images:

```sh
python3 tools/run_benchmarks.py --suite performance \
  --only the_lens prism_in_sunlight the_peacock_lamp \
  --binary build/owe --backend gpu --device 1 --reference-spp 256 --budgets 16 64 \
  --reference-dir out/performance_20261008/bounded_baseline \
  --baseline out/performance_20261008/bounded_baseline/summary.json \
  --output out/performance_20261008/bounded_candidate
```

The default target for each case is its best observed baseline full-image relMSE, not its largest sample budget. Lens and lamp errors are nonmonotonic here: their 64-sample runs are farther from the finite reference than their 16-sample runs. Sun glints and Monte Carlo noise make this a weak quality target; converged references and repeated independent seeds remain necessary for a production speedup claim. The diagnostic masked metric is not the goal.

Transport inconsistency/leak counters above are retained rather than hidden. These scenes are not clean geometry-conformance references, and this checkpoint does not establish unchanged expected value or satisfy the correctness gate. Kernel/statistical conformance checks still apply independently. RSS uses an isolated Python helper’s renderer-child maximum RSS, not GPU VRAM; wall-minus-transport is setup/output overhead, not a load-only timer.

Artifacts: [saved invocation](../../../benchmarks/evidence/performance_20261008/bounded_baseline/invocation.json), [summary JSON](../../../benchmarks/evidence/performance_20261008/bounded_baseline/summary.json), [summary table](../../../benchmarks/evidence/performance_20261008/bounded_baseline/summary.md). Per-render sidecars, raw images, native metadata and logs are under the case subdirectories. Preserve these local output artifacts separately; the output tree is ignored by Git.
