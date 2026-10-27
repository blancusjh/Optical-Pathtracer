# Maintenance validation, 12–13 October 2026

[Current project status](../../../docs/STATUS.md) is authoritative. This directory
retains the logs and numeric checks underlying the maintenance changes.

- [Validation manifest](validation.json): full hardware, selected software,
  CPU-only build, tooling/viewer and final film-layout checks.
- [Four-case comparison](../../results/20261013_maintenance.json) and
  [many-pass film workload](../../results/20261013_film_workload.json): commands,
  binary/asset identities, native diagnostics, warmups, three alternating timed
  repeats, median ratios and raw-image comparisons. All five cases retain the
  same path, segment, inconsistency and leak counts before and after.
- [Initial rejected candidate](initial_performance.json), its
  [particle profile](initial_profile_after.log), and the
  [saved-binary profile](profile_before.log) expose the aggregate-parameter
  compiler regression. The [scalar adapter profile](profile_scalar_adapter.log)
  and [warm comparison](warm_light_check.json) document its correction.
- [Final output profile](film_layout_profile.log) records one readback per pass;
  the output metadata uses the already resolved image.
- [Panel topology](panel_topology.json): all indexed edges have two oppositely
  oriented incident triangles; vertices are finite and no triangle has zero
  area. This does not check self-intersections or floating-point traversal.

The baseline was a preserved executable from the pre-maintenance workspace,
identified by its SHA-256; it is not a clean tagged-source rebuild or v1.0.
The comparison records identify the committed candidate source. Raw images and
saved executables remain local scratch; these portable records do not replace
those files when rerunning the experiments. The tests establish the stated
coverage, not that the production scenes are free of transport errors.
