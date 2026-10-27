"""Fast benchmark metric/cache/CLI checks; fake renderer avoids production renders."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_benchmarks as bench


def write_pfm(path, values, scale=-1):
    values = np.asarray(values, dtype="<f4" if scale < 0 else ">f4")
    with Path(path).open("wb") as f:
        f.write(f"PF\n{values.shape[1]} {values.shape[0]}\n{scale}\n".encode())
        f.write(values.tobytes())


class ErrorTests(unittest.TestCase):
    def test_raw_metric_and_mean_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.pfm"
            ref = np.ones((2, 2, 3))
            write_pfm(path, 2 * ref)
            got = bench.errors(path, ref, np.zeros((2, 2), dtype=bool))
            self.assertAlmostEqual(got["rel_mse"], 1 / 1.01)
            self.assertEqual(got["mean_ratio"], 2)

    def test_pfm_scale_and_endian(self):
        with tempfile.TemporaryDirectory() as d:
            for scale in (-2, 2):
                path = Path(d) / "a.pfm"
                write_pfm(path, np.ones((2, 3, 3)), scale)
                np.testing.assert_array_equal(bench.read_pfm(path), 2 * np.ones((2, 3, 3)))

    def test_dark_reference_has_no_nan_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.pfm"
            ref = np.zeros((1, 1, 3))
            write_pfm(path, ref)
            result = bench.errors(path, ref, np.zeros((1, 1), dtype=bool))
            self.assertEqual(result["rel_mse"], 0)
            self.assertIsNone(result["mean_ratio"])
            write_pfm(path, np.ones_like(ref))
            self.assertIsNone(bench.errors(path, ref, np.zeros((1, 1), dtype=bool))["rel_mse"])

    def test_shape_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.pfm"
            write_pfm(path, np.full((1, 1, 3), float("nan")))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                bench.errors(path, np.ones((1, 1, 3)), np.zeros((1, 1), dtype=bool))
            write_pfm(path, np.ones((1, 1, 3)))
            with self.assertRaisesRegex(ValueError, "dimensions"):
                bench.errors(path, np.ones((2, 2, 3)), np.zeros((2, 2), dtype=bool))


class ComparisonTests(unittest.TestCase):
    def entry(self, name, seconds, error, identity="same", **extra):
        return {"scene": name, "measurement_id": identity, "runs": [
            {"seconds": seconds, "wall_seconds": seconds + 2, "rel_mse": error}], **extra}

    def test_time_to_equal_error_not_equal_spp_or_throughput(self):
        baseline = self.entry("a", 20, .1)
        candidate = self.entry("a", 10, .2)
        candidate["runs"].append({"seconds": 5, "wall_seconds": 7, "rel_mse": .08})
        comparison = bench.compare_baseline([candidate], [baseline])
        self.assertEqual(comparison["geomean_speedup"], 4)
        self.assertTrue(comparison["goal_met"])

    def test_no_extrapolation_for_unreached_error(self):
        got = bench.compare_baseline([self.entry("a", 1, .2)], [self.entry("a", 20, .1)])
        self.assertFalse(got["complete"])
        self.assertFalse(got["goal_met"])
        self.assertIsNone(got["geomean_speedup"])
        self.assertEqual(got["scenes"][0]["status"], "candidate_target_missing")

    def test_missing_and_incompatible_cases_remain_in_denominator(self):
        old = [self.entry("a", 16, .1), self.entry("b", 20, .1), self.entry("c", 20, .1)]
        new = [self.entry("a", 1, .1), self.entry("c", 1, .1, identity="changed_scene")]
        got = bench.compare_baseline(new, old)
        self.assertEqual(got["matched_cases"], 1)
        self.assertEqual(got["required_cases"], 3)
        self.assertAlmostEqual(got["partial_geomean_speedup"], 16)
        self.assertIsNone(got["geomean_speedup"])
        self.assertFalse(got["goal_met"])

    def test_geomean_and_wall_selection(self):
        old = [self.entry("a", 8, .1), self.entry("b", 2, .1)]
        new = [self.entry("a", 2, .1), self.entry("b", 2, .1)]
        got = bench.compare_baseline(new, old)
        self.assertEqual(got["geomean_speedup"], 2)
        wall = bench.compare_baseline(new, old, time_key="wall_seconds")
        self.assertAlmostEqual(wall["geomean_speedup"], 2.5 ** .5)

    def test_profiled_runs_cannot_claim_goal(self):
        got = bench.compare_baseline([self.entry("a", 1, .1, profiled=True)], [self.entry("a", 8, .1)])
        self.assertFalse(got["goal_met"])
        self.assertTrue(got["profiled"])

    def test_bad_errors_and_times_cannot_meet_target(self):
        runs = [{"seconds": 0, "rel_mse": 0}, {"seconds": 1, "rel_mse": float("nan")},
                {"seconds": 1, "rel_mse": -1}, {"seconds": float("inf"), "rel_mse": 0}]
        self.assertIsNone(bench.fastest_at_error(runs, .1))


class IdentityTests(unittest.TestCase):
    def test_source_identity_explicitly_describes_capture_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for folder in ('src', 'apps', 'benchmarks'):
                (root / folder).mkdir()
            for name in ('src/current.cpp', 'apps/current.cpp', 'CMakeLists.txt', 'benchmarks/suites.json'):
                (root / name).write_text('current workspace')
            binary = root / 'frozen_historical_binary'
            binary.write_bytes(b'compiled historical implementation')
            with patch.object(bench, 'ROOT', root):
                identity = bench.code_identity(binary)
            self.assertEqual(identity['source_scope'], 'capture_workspace')
            self.assertIn("not the executable's build provenance", identity['source_description'])
            self.assertEqual(identity['binary_sha256'], bench.file_digest(binary))
            self.assertTrue(identity['source_sha256'])

    def test_nvidia_identity_records_driver_version_and_uuid(self):
        result = type("Result", (), {"returncode": 0, "stderr": "", "stdout":
            "0, GPU-test-uuid, NVIDIA GeForce RTX 4060 Laptop GPU, 580.95.05, 00000000:01:00.0\n"})()
        with patch.object(bench.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             patch.object(bench.subprocess, "run", return_value=result) as run:
            identity = bench.nvidia_driver_identity()
        self.assertEqual(identity["status"], "available")
        self.assertEqual(identity["devices"][0]["driver_version"], "580.95.05")
        self.assertEqual(identity["devices"][0]["uuid"], "GPU-test-uuid")
        self.assertIn("--query-gpu=index,uuid,name,driver_version,pci.bus_id", run.call_args.args[0])
        self.assertNotIn("temperature", str(run.call_args))

    def test_nvidia_identity_unavailable_is_explicit(self):
        with patch.object(bench.shutil, "which", return_value=None):
            self.assertEqual(bench.nvidia_driver_identity(), {"status": "unavailable"})

    def test_dependencies_include_lens_recursive_gltf_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "scene.owe").write_text('body Lens { file = "a.lens" }\nbody Model { file = "a.gltf" }\n')
            (root / "a.lens").write_text("optical prescription")
            (root / "a.gltf").write_text(json.dumps({"buffers": [{"uri": "a.bin"}], "images": [{"uri": "missing.png"}]}))
            (root / "a.bin").write_bytes(b"mesh")
            first = bench.asset_identity(root / "scene.owe")
            self.assertEqual(len(first), 5)
            self.assertIn("missing", first.values())
            (root / "a.lens").write_text("different prescription")
            self.assertNotEqual(first, bench.asset_identity(root / "scene.owe"))
            (root / "missing.png").write_bytes(b"new texture")
            self.assertNotIn("missing", bench.asset_identity(root / "scene.owe").values())

    def test_glb_external_images_are_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            doc = json.dumps({"images": [{"uri": "a.png"}]}).encode()
            doc += b" " * (-len(doc) % 4)
            (root / "a.glb").write_bytes(struct.pack("<4sIII", b"glTF", 2, len(doc) + 20, len(doc)) + struct.pack("<I", 0x4E4F534A) + doc)
            (root / "a.png").write_bytes(b"image")
            self.assertEqual(len(bench.asset_identity(root / "a.glb")), 2)

    def test_cache_requires_matching_identity_and_intact_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "render"
            write_pfm(str(out) + ".pfm", np.ones((1, 1, 3)))
            Path(str(out) + ".json").write_text("{}")
            identity = {"schema": 2, "measurement": {"config": "fixed", "host": "fixed"}, "code": "old"}
            side = {"identity": identity, "record": {"wall_seconds": 3},
                    "artifacts": {ext: bench.file_digest(str(out) + ext) for ext in (".pfm", ".json")}}
            Path(str(out) + ".benchmark.json").write_text(json.dumps(side))
            self.assertEqual(bench.cached_record(out, identity)["wall_seconds"], 3)
            changed_code = {**identity, "code": "new"}
            self.assertIsNone(bench.cached_record(out, changed_code))
            self.assertIsNotNone(bench.cached_record(out, changed_code, frozen=True))
            changed_host = {**identity, "measurement": {"config": "fixed", "host": "changed"}}
            self.assertIsNone(bench.cached_record(out, changed_host, frozen=True))
            Path(str(out) + ".pfm").write_bytes(b"modified")
            self.assertIsNone(bench.cached_record(out, identity))

    def test_legacy_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "render"
            Path(str(out) + ".json").write_text("{}")
            write_pfm(str(out) + ".pfm", np.ones((1, 1, 3)))
            self.assertIsNone(bench.cached_record(out, {}))


class CliTests(unittest.TestCase):
    def test_original_default_and_scan_opt_in(self):
        args = bench.parser().parse_args([])
        cases = bench.select_cases(args)
        self.assertEqual(len(cases), 12)
        self.assertTrue(all(c["suite"] == "transport" for c in cases))
        args = bench.parser().parse_args(["--suite", "performance"])
        cases = bench.select_cases(args)
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(c["suite"] == "canonical" for c in cases), 8)
        self.assertTrue(all(c["resolution"] == "960x600" for c in cases if c["suite"] == "scans"))

    def test_plan_has_fixed_detector_without_needing_binary(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = bench.main(["--suite", "canonical", "--only", "study_camera", "--list", "--binary", "/missing",
                                 "--device", "2", "--budgets", "1", "2"])
        self.assertEqual(result, 0)
        case = json.loads(output.getvalue())[0]
        self.assertEqual(case["settings"]["detector"], "Cam")
        self.assertEqual(case["settings"]["device"], 2)
        self.assertEqual(case["budgets"], [1, 2])

    def test_invalid_selection_and_budgets_fail(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for args in (["--only", "typo", "--list"], ["--budgets", "0", "--list"],
                         ["--resolution", "0x600", "--list"], ["--target-error", "nan", "--list"]):
                with self.assertRaises(SystemExit) as e:
                    bench.main(args)
                self.assertEqual(e.exception.code, 2)

    def test_command_preserves_selection_and_settings(self):
        config = {"scene": "scene.owe", "detector": "Cam", "integrator": "path", "spp": 3,
                  "seed": 5, "backend": "cpu", "device": 2, "resolution": "8x6", "passes": 1, "threads": 1}
        cmd = bench.render_command(Path("/custom/owe"), Path("/tmp/out"), config)
        self.assertEqual(cmd[0], "/custom/owe")
        for flag, value in (("--detector", "Cam"), ("--device", "2"), ("--resolution", "8x6"), ("--seed", "5")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_seed_participates_in_measurement_settings(self):
        first = bench.parser().parse_args(["--suite", "canonical", "--seed", "1"])
        second = bench.parser().parse_args(["--suite", "canonical", "--seed", "2"])
        case = bench.select_cases(first)[0]
        self.assertNotEqual(bench.digest(bench.run_settings(case, first)),
                            bench.digest(bench.run_settings(case, second)))

    def test_production_eligibility_requires_all_standard_manifest_views(self):
        args = bench.parser().parse_args(["--suite", "performance"])
        records = [{"scene": case["name"], "suite": case["suite"], "settings": bench.run_settings(case, args)}
                   for case in bench.select_cases(args)]
        self.assertTrue(bench.performance_suite_eligible(records))
        self.assertFalse(bench.performance_suite_eligible(records[:-1]))
        records[0]["settings"]["resolution"] = "4x4"
        self.assertFalse(bench.performance_suite_eligible(records))
        records[0]["settings"]["resolution"] = "256x192"
        records[-1]["settings"]["resolution"] = "4x4"
        self.assertFalse(bench.performance_suite_eligible(records))
        records[-1]["settings"]["resolution"] = "960x600"
        records[0]["settings"]["detector"] = "another_camera"
        self.assertFalse(bench.performance_suite_eligible(records))


class FakeRenderTests(unittest.TestCase):
    def test_resource_fallback_selection_and_platform_units(self):
        with patch.object(bench, "gnu_time_available", return_value=False), patch.object(bench.sys, "platform", "linux"):
            self.assertEqual(bench.rss_method(), "resource")
        self.assertEqual(bench.rss_bytes(123, "linux"), 123 * 1024)
        self.assertEqual(bench.rss_bytes(123, "darwin"), 123)
        self.assertIsNone(bench.rss_bytes(123, "unknown"))

    def test_timings_persist_and_rescore_reuses_without_running(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "render"
            record = {"render_seconds": .1, "integrator": "path", "samples_per_pixel": 2, "seed": 1}
            def fake_run(*args, **kwargs):
                write_pfm(str(out) + ".pfm", np.ones((1, 1, 3)))
                Path(str(out) + ".json").write_text(json.dumps(record))
                return type("Result", (), {"returncode": 0, "stdout": "output", "stderr": "owe_gpu_profile kind=kernel name=path calls=1 gpu_ms=1"})()
            kwargs = {"provenance": {"code": "v1", "host": "host"}, "assets": {"scene": "hash"}, "measure_rss": False}
            with patch.object(bench.subprocess, "run", side_effect=fake_run) as call:
                first = bench.render("scene.owe", out, "path", 2, "cpu", **kwargs)
                second = bench.render("scene.owe", out, "path", 2, "cpu", reuse=True, **kwargs)
                self.assertEqual(call.call_count, 1)
            self.assertEqual(first["wall_seconds"], second["wall_seconds"])
            self.assertIsNone(first["peak_host_rss_bytes"])
            self.assertEqual(first["gpu_profile"], ["owe_gpu_profile kind=kernel name=path calls=1 gpu_ms=1"])
            self.assertIn("owe_gpu_profile", Path(str(out) + ".stderr.log").read_text())

    def test_frozen_missing_reference_does_not_render(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(bench.subprocess, "run") as call:
                with self.assertRaisesRegex(RuntimeError, "frozen reference"):
                    bench.render("scene.owe", Path(d) / "reference", "path", 2, "cpu", frozen=True,
                                 provenance={"code": "new", "host": "host"}, assets={"scene": "hash"})
                call.assert_not_called()

    def test_fallback_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "render"
            def fake_run(*args, **kwargs):
                Path(str(out) + ".json").write_text(json.dumps({"integrator": "path"}))
                return type("Result", (), {"returncode": 0, "stdout": "", "stderr": "fallback"})()
            with patch.object(bench.subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "fallback"):
                    bench.render("scene.owe", out, "vcm", 2, "cpu", measure_rss=False,
                                 provenance={"code": "v1", "host": "host"}, assets={"scene": "hash"})
            self.assertFalse(Path(str(out) + ".benchmark.json").exists())

    def test_failed_child_retains_wall_peak_rss_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "render"
            def fake_run(*args, **kwargs):
                Path(str(out) + ".rss.txt").write_text("Command terminated by signal 9\n123\n")
                return type("Result", (), {"returncode": 137, "stdout": "", "stderr": "killed"})()
            with patch.object(bench.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(bench.RenderFailure) as error:
                    bench.render("scene.owe", out, "path", 2, "cpu", measure_rss=True,
                                 provenance={"code": "v1", "host": "host"}, assets={"scene": "hash"})
            self.assertEqual(error.exception.measurements["peak_host_rss_bytes"], 123 * 1024)
            self.assertGreaterEqual(error.exception.measurements["wall_seconds"], 0)
            self.assertEqual(Path(str(out) + ".stderr.log").read_text(), "killed")

    def test_isolated_resource_helper_reports_each_child_separately(self):
        with tempfile.TemporaryDirectory() as d:
            rss = Path(d) / "usage.json"
            child = [bench.sys.executable, "-c", "data = bytearray(16 * 1024 * 1024); print('child stdout')"]
            first = bench.subprocess.run([bench.sys.executable, "-c", bench.RESOURCE_HELPER, str(rss), *child],
                                         capture_output=True, text=True)
            self.assertEqual(first.returncode, 0)
            self.assertEqual(first.stdout.strip(), "child stdout")
            big = json.loads(rss.read_text())
            second = bench.subprocess.run([bench.sys.executable, "-c", bench.RESOURCE_HELPER, str(rss),
                                           bench.sys.executable, "-c", "raise SystemExit(3)"], capture_output=True, text=True)
            self.assertEqual(second.returncode, 3)
            small = json.loads(rss.read_text())
            self.assertEqual(small["returncode"], 3)
            self.assertGreater(bench.rss_bytes(big["max_rss"], big["platform"]), 0)
            self.assertLess(small["max_rss"], big["max_rss"])


if __name__ == "__main__":
    unittest.main()
