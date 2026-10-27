# Native-owner hardware fallback proof, 9 October 2026

All six explicitly selected tests pass, exit 0. Names in metadata are checked
against the raw log, including fresh scene-layout/native root/bitwise matrix
assertions, four contact controls and the unchanged smooth-glass assertion.
Smooth-glass block L1 printed errors are 0.0192 faceted / 0.0057 smooth, with
zero smooth inconsistencies. Contact controls preserve exact darkness, positive
resolved gaps and exp(-1)/exp(-3) attenuation; true overlap remains diagnosed.

The temporary binary overrides only gpu_scene.cpp.o and three hardware shaders
(path, SPPM camera, SPPM photon); production renderer and other kernels remain
existing archives. Both link maps verify original gpu_scene archive member
was not extracted. Fresh test_gpu.cpp object includes the owner's native BVH
root and bitwise transform assertions. No assertion was relaxed. Source/object/
SPV/library and binary hashes, exact commands and actual test names are in
metadata.json. This snapshot proves hardware only; later main source adds
software parity and needs its own integrated validation and build identity.

Frozen CLI: `/tmp/owe_native_owner_probe_20261009/owe_cli`. SHA-256:
`8b5f6b3fbbda1a21013ab22141380e8e5d61c6e3ecdb42c2a8cc2cc89811cee7`.
Root's subsequent path-only previews use `env -u OWE_GPU_PROFILE
OWE_GPU_RAY_QUERY=1`, with `--backend gpu --device 1 --integrator path`; they
are staged diagnostics, not performance/gallery acceptance. Other embedded
kernels are not this native-owner snapshot.

Peak test-child host RSS 6,222,928 KiB (5.93 GiB), minimum sampled available
host memory 13.30 GiB; wall 308.11 seconds includes driver compilation and
test execution. No memory abort or OOM occurred. No performance claim or
canonical-reference cost forecast follows from this six-test run. Test images
are resolved in memory; raw logs preserve the actual comparison, not fabricated
native image files. Main/shared build files were not edited by this probe.
