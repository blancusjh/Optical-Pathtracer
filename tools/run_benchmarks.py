#!/usr/bin/env python3
"""Raw-image error benchmarks; see docs/BENCHMARKS.md for reproducibility and comparison rules."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import struct
import sys
import time
from urllib.parse import unquote

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
OWE = ROOT / "build" / "owe"
SCHEMA = 2
METRIC = "rel_mse"
# A fresh helper has exactly one child, so RUSAGE_CHILDREN cannot retain an earlier
# render's high-water mark. Its own Python memory is not included in the child's RSS.
RESOURCE_HELPER = """
import json, resource, subprocess, sys
result = subprocess.run(sys.argv[2:])
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
with open(sys.argv[1], 'w') as f:
    json.dump({'max_rss': peak, 'platform': sys.platform, 'returncode': result.returncode}, f)
sys.exit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
"""


def read_pfm(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"PF":
            raise ValueError(f"{path}: expected RGB PFM")
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
    if not scale or w <= 0 or h <= 0 or data.size != w * h * 3:
        raise ValueError(f"{path}: malformed PFM")
    return data.reshape(h, w, 3).astype(np.float64) * abs(scale)


def glint_mask(ref):
    """Diagnostic mask near sun glints; full-image relMSE is the performance target."""
    m = ref.max(axis=2)
    median = np.median(sliding_window_view(np.pad(m, 4, mode="edge"), (9, 9)), axis=(2, 3))
    seeds = (m > 20 * median) & (m > 10 * m.mean())
    mask = sliding_window_view(np.pad(seeds, 4), (9, 9)).any(axis=(2, 3))
    return mask if m[mask].sum() < 0.5 * m.sum() else np.zeros_like(mask)


def errors(pfm, ref, mask):
    a = read_pfm(pfm)
    if a.shape != ref.shape or mask.shape != ref.shape[:2]:
        raise ValueError("render/reference dimensions differ")
    if not np.isfinite(a).all() or not np.isfinite(ref).all():
        raise ValueError("nonfinite raw image")
    mean_ref = ref.mean()
    denom = ref * ref + 0.01 * mean_ref * mean_ref
    # A completely dark reference has zero error only for an equally dark render.
    rel = np.divide((a - ref) ** 2, denom, out=np.zeros_like(a), where=denom > 0)
    if np.any((denom == 0) & (a != ref)):
        return {"rel_mse": None, "rel_mse_no_glints": None, "mean_ratio": None}
    return {"rel_mse": float(rel.mean()), "rel_mse_no_glints": float(rel[~mask].mean()) if (~mask).any() else None,
            "mean_ratio": float(a.mean() / mean_ref) if mean_ref else None}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_identity(scene, cache=None):
    """Content hashes of scene files and recursive external glTF/OBJ/image dependencies."""
    cache = {} if cache is None else cache
    found = {}
    pending = [Path(scene).resolve()]
    while pending:
        path = pending.pop()
        key = os.path.relpath(path, ROOT)
        if key in found:
            continue
        if not path.is_file():
            found[key] = "missing"
            continue
        if path not in cache:
            cache[path] = file_digest(path)
        found[key] = cache[path]
        suffix = path.suffix.lower()
        refs = []
        if suffix == ".owe":
            # Strip line comments; quoted paths in file/image/sky/model declarations.
            text = re.sub(r"(?m)#.*$", "", path.read_text())
            refs = [s for s in re.findall(r'"([^"\n]+)"', text)
                    if Path(s).suffix or (path.parent / s).is_file()]
        elif suffix in (".gltf", ".glb"):
            if suffix == ".gltf":
                doc = json.loads(path.read_text())
            else:
                with path.open("rb") as f:
                    magic, version, _ = struct.unpack("<4sII", f.read(12))
                    if magic != b"glTF" or version != 2:
                        raise ValueError(f"unsupported GLB: {path}")
                    length, kind = struct.unpack("<II", f.read(8))
                    if kind != 0x4E4F534A:
                        raise ValueError(f"missing GLB JSON chunk: {path}")
                    doc = json.loads(f.read(length))
            refs = [v["uri"] for group in ("buffers", "images") for v in doc.get(group, []) if "uri" in v]
        elif suffix == ".obj":
            refs = [line.split(None, 1)[1].strip() for line in path.read_text().splitlines() if line.startswith("mtllib ")]
        elif suffix == ".mtl":
            refs = [line.split()[-1] for line in path.read_text().splitlines() if line.startswith(("map_", "bump "))]
        for ref in refs:
            if not ref.startswith("data:"):
                pending.append((path.parent / unquote(ref)).resolve())
    return found


def code_identity(binary):
    files = [p for folder in (ROOT / "src", ROOT / "apps") for p in folder.rglob("*") if p.is_file()]
    files += [ROOT / "CMakeLists.txt", Path(__file__), ROOT / "benchmarks" / "suites.json"]
    return {"binary_sha256": file_digest(binary),
            "source_scope": "capture_workspace",
            "source_description": "source_sha256 hashes the current workspace sources, build definition, runner and suite manifest; it is not the executable's build provenance",
            "source_sha256": digest({os.path.relpath(p, ROOT): file_digest(p) for p in sorted(files)}),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}


def environment_identity(binary):
    result = subprocess.run([str(binary), "backends"], cwd=ROOT, capture_output=True, text=True)
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "backend_inventory": result.stdout + result.stderr,
            "nvidia_driver_inventory": nvidia_driver_identity(),
            "environment": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith(("OWE_", "VK_", "VULKAN_", "CUDA_")) or k in ("OMP_NUM_THREADS", "SLANG_DIR")}}


def nvidia_driver_identity():
    """Read only stable identity fields; exclude temperatures/utilization from reuse keys."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {"status": "unavailable"}
    fields = ("index", "uuid", "name", "driver_version", "pci.bus_id")
    try:
        result = subprocess.run([binary, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": str(error)}
    if result.returncode:
        return {"status": "unavailable", "reason": result.stderr.strip() or result.stdout.strip()}
    rows = [dict(zip(fields, (value.strip() for value in row))) for row in csv.reader(result.stdout.splitlines())
            if len(row) == len(fields)]
    return {"status": "available" if rows else "unavailable", "devices": rows}


def scene_settings(path):
    text = re.sub(r"(?m)#.*$", "", Path(path).read_text())
    block = re.search(r"\brender\s*\{([^}]*)\}", text, re.S)
    block = block.group(1) if block else ""
    def value(key, default):
        match = re.search(r"\b" + key + r"\s*=\s*([^\s}]+)", block)
        return match.group(1).strip('"') if match else default
    return {"detector": value("detector", ""), "integrator": value("integrator", "path")}


def select_cases(args):
    manifest = json.loads((ROOT / "benchmarks" / "suites.json").read_text())
    cases = []
    if args.suite == "transport":
        for path in sorted((ROOT / "benchmarks").glob("*.owe")):
            cases.append({"name": path.stem, "path": os.path.relpath(path, ROOT),
                          **scene_settings(path), "suite": "transport"})
    for suite in ("canonical", "scans"):
        if args.suite in (suite, "performance") or (suite == "scans" and args.include_scans):
            for item in manifest[suite]:
                cases.append({**item, "suite": suite,
                              "integrator": item.get("integrator", "path"),
                              "max_depth": item.get("max_depth", 64),
                              "resolution": "960x600" if suite == "scans" else "256x192"})
    if args.only:
        names = set(args.only)
        unknown = names - {c["name"] for c in cases}
        if unknown:
            raise ValueError("unknown case(s) for selected suite: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["name"] in names]
    for case in cases:
        if args.resolution:
            case["resolution"] = args.resolution
    return cases


def run_settings(case, args):
    return {"scene": case["path"], "detector": case["detector"], "backend": args.backend,
            "device": args.device, "threads": args.threads, "passes": args.passes, "seed": args.seed,
            **{k: case[k] for k in ("resolution", "max_depth", "photons") if k in case}}


def render_command(binary, out, config):
    cmd = [str(binary), "render", config["scene"], "--out", str(out)]
    for key, option in (("detector", "--detector"), ("integrator", "--integrator"), ("spp", "--spp"),
                        ("backend", "--backend"), ("seed", "--seed"), ("device", "--device"),
                        ("threads", "--threads"), ("passes", "--passes"), ("resolution", "--resolution"),
                        ("max_depth", "--max-depth"), ("photons", "--photons")):
        if key in config and config[key] is not None and config[key] != "":
            cmd += [option, str(config[key])]
    return cmd


def cached_record(out, identity, frozen=False):
    """No legacy or mutated artifact is accepted as a reproducible cache hit."""
    try:
        sidecar = json.loads(Path(str(out) + ".benchmark.json").read_text())
        expected = identity["measurement"] if frozen else identity
        actual = sidecar["identity"]["measurement"] if frozen else sidecar["identity"]
        if expected != actual:
            return None
        if any(file_digest(str(out) + ext) != sidecar["artifacts"][ext] for ext in (".pfm", ".json")):
            return None
        return sidecar["record"]
    except (OSError, ValueError, KeyError):
        return None


def gnu_time_available():
    path = Path("/usr/bin/time")
    if not path.exists():
        return False
    r = subprocess.run([str(path), "--version"], capture_output=True, text=True)
    return r.returncode == 0 and "GNU" in r.stdout


def rss_method():
    if gnu_time_available():
        return "gnu"
    if sys.platform in ("linux", "darwin") and importlib.util.find_spec("resource") is not None:
        return "resource"
    return None


def rss_bytes(value, system):
    if system == "linux":
        return int(value) * 1024
    if system == "darwin":
        return int(value)
    return None  # Do not assume ru_maxrss units on another operating system.


class RenderFailure(RuntimeError):
    def __init__(self, message, measurements):
        super().__init__(message)
        self.measurements = measurements


def render(scene, out, integrator, spp, backend, seed=1, reuse=False, *, binary=OWE,
           settings=None, provenance=None, assets=None, measure_rss=None, frozen=False):
    # The original positional call remains supported.
    out = Path(out)
    config = {**(settings or {}), "scene": str(scene), "integrator": integrator,
              "spp": spp, "backend": backend, "seed": seed}
    if provenance is None:
        provenance = {"code": code_identity(binary), "host": environment_identity(binary)}
    identity = {"schema": SCHEMA, "measurement": {"config": config, "host": provenance["host"],
                "assets": assets if assets is not None else asset_identity(ROOT / scene)}, **provenance}
    if reuse or frozen:
        hit = cached_record(out, identity, frozen=frozen)
        if hit is not None:
            return hit
        if frozen:
            raise RuntimeError(f"missing or incompatible frozen reference: {out}")
    cmd = render_command(binary, out, config)
    measured_cmd = cmd
    rss_path = Path(str(out) + ".rss.txt")
    if measure_rss is None:
        measure_rss = rss_method()
    elif measure_rss is True:
        measure_rss = "gnu"
    if measure_rss == "gnu":
        measured_cmd = ["/usr/bin/time", "-f", "%M", "-o", str(rss_path), "--", *cmd]
    elif measure_rss == "resource":
        measured_cmd = [sys.executable, "-c", RESOURCE_HELPER, str(rss_path), *cmd]
    t0 = time.perf_counter()
    result = subprocess.run(measured_cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    Path(str(out) + ".stderr.log").write_text(result.stderr)
    Path(str(out) + ".stdout.log").write_text(result.stdout)
    peak = None
    if measure_rss == "gnu":
        try:
            peak = int(rss_path.read_text().strip().splitlines()[-1]) * 1024  # GNU time reports KiB on Linux.
        except (OSError, ValueError):
            pass
    elif measure_rss == "resource":
        try:
            usage = json.loads(rss_path.read_text())
            peak = rss_bytes(usage["max_rss"], usage["platform"])
        except (OSError, ValueError, KeyError):
            pass
    method = {"gnu": "gnu_time_maxrss", "resource": "isolated_resource_children_maxrss"}.get(measure_rss) if peak is not None else None
    measurements = {"wall_seconds": wall, "peak_host_rss_bytes": peak,
                    "rss_method": method}
    if result.returncode:
        raise RenderFailure(f"render exited {result.returncode}; see {out}.stderr.log", measurements)
    with open(str(out) + ".json") as f:
        record = json.load(f)
    if record.get("integrator") != integrator:
        raise RuntimeError(f"requested {integrator}, renderer used {record.get('integrator')}; fallback is not a benchmark")
    if record.get("samples_per_pixel") != spp or record.get("seed") != seed:
        raise RuntimeError("renderer sample count or seed differs from the requested configuration")
    transport = record["render_seconds"]
    record.update(wall_seconds=wall, non_transport_seconds=max(0.0, wall - transport),
                  peak_host_rss_bytes=peak, rss_method=method)
    profile = [line for line in result.stderr.splitlines() if line.startswith("owe_gpu_profile ")]
    if profile:
        record["gpu_profile"] = profile
    sidecar = {"identity": identity, "command": cmd, "record": record,
               "artifacts": {ext: file_digest(str(out) + ext) for ext in (".pfm", ".json")}}
    Path(str(out) + ".benchmark.json").write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
    return record


def fastest_at_error(runs, target, time_key="seconds"):
    valid = [r for r in runs if isinstance(r.get(METRIC), (int, float)) and math.isfinite(r[METRIC])
             and 0 <= r[METRIC] <= target and isinstance(r.get(time_key), (int, float))
             and math.isfinite(r[time_key]) and r[time_key] > 0]
    return min(valid, key=lambda r: r[time_key]) if valid else None


def compare_baseline(current, baseline, *, target_error=None, time_key="seconds", goal=4.0):
    """Observed time to equal error. Never extrapolate, silently omit, or claim partial success."""
    old = {e["scene"]: e for e in baseline}
    now = {e["scene"]: e for e in current}
    rows = []
    # Union ensures a missing requested baseline case cannot improve the geometric mean.
    for name in sorted(old.keys() | now.keys()):
        row = {"scene": name, "status": "missing_case"}
        b, c = old.get(name), now.get(name)
        if b is not None and c is not None:
            if not b.get("measurement_id") or b["measurement_id"] != c.get("measurement_id"):
                row["status"] = "incompatible_measurement"
            else:
                values = [r[METRIC] for r in b["runs"] if isinstance(r.get(METRIC), (int, float))
                          and math.isfinite(r[METRIC]) and r[METRIC] >= 0]
                target = target_error if target_error is not None else min(values, default=None)
                row["target_error"] = target
                br = fastest_at_error(b["runs"], target, time_key) if target is not None else None
                cr = fastest_at_error(c["runs"], target, time_key) if target is not None else None
                row["status"] = "baseline_target_missing" if br is None else "candidate_target_missing" if cr is None else "matched"
                if br is not None:
                    row["baseline_seconds"] = br[time_key]
                if cr is not None:
                    row["candidate_seconds"] = cr[time_key]
                if br is not None and cr is not None:
                    row["speedup"] = br[time_key] / cr[time_key]
        rows.append(row)
    ratios = [r["speedup"] for r in rows if r["status"] == "matched"]
    partial = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
    complete = bool(rows) and len(ratios) == len(rows)
    profiled = any(e.get("profiled", False) for e in [*current, *baseline])
    return {"metric": METRIC, "time_key": time_key, "method": "fastest_observed_meeting_error_no_interpolation",
            "goal_speedup": goal, "matched_cases": len(ratios), "required_cases": len(rows), "complete": complete,
            "geomean_speedup": partial if complete else None, "partial_geomean_speedup": partial,
            "profiled": profiled, "goal_met": complete and not profiled and partial >= goal, "scenes": rows}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--suite", choices=("transport", "canonical", "scans", "performance"), default="transport")
    ap.add_argument("--include-scans", action="store_true")
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--binary", type=Path, default=OWE)
    ap.add_argument("--output", "--out", type=Path, default=ROOT / "out" / "benchmarks")
    ap.add_argument("--reference-dir", type=Path, help="read frozen references from a previous output directory")
    ap.add_argument("--reference-spp", type=int, default=8192)
    ap.add_argument("--reference-integrator")
    ap.add_argument("--integrators", nargs="+", choices=("path", "light", "sppm", "vcm", "bdpt", "hybrid"))
    ap.add_argument("--budgets", nargs="+", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reference-seed", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--resolution", help="override suite resolution (WxH), including in quick mode")
    ap.add_argument("--quick", action="store_true", help="16/64 budgets and a 256-spp reference; resolution is unchanged")
    ap.add_argument("--rescore", action="store_true", help="reuse only verified matching renders; render missing/stale ones")
    ap.add_argument("--baseline", type=Path, help="compare against a previous summary.json")
    ap.add_argument("--target-error", type=float, help="fixed full-image relMSE target, otherwise each baseline's best error")
    ap.add_argument("--time", choices=("transport", "wall"), default="transport")
    ap.add_argument("--goal-speedup", type=float, default=4.0)
    ap.add_argument("--list", "--dry-run", action="store_true", help="print cases/settings without loading assets or rendering")
    return ap


def performance_suite_eligible(records):
    """Certification uses every fixed manifest view at its standard resolution."""
    manifest = json.loads((ROOT / "benchmarks" / "suites.json").read_text())
    actual = {entry["scene"]: entry for entry in records}
    for suite, cases in manifest.items():
        resolution = "960x600" if suite == "scans" else "256x192"
        for case in cases:
            entry = actual.get(case["name"], {})
            settings = entry.get("settings", {})
            if entry.get("suite") != suite or any(settings.get(key) != value for key, value in
                (("scene", case["path"]), ("detector", case["detector"]), ("resolution", resolution))):
                return False
    return True


def write_reports(out_root, summary, comparison=None):
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    text = ["Raw linear PFM; full-image relMSE is the target. No denoising.", "",
            "Non-transport time includes scene loading, compilation, upload, resolve and output; it is not a load-only timer.", "",
            "| scene | estimator | spp | transport s | wall s | non-transport s | peak host MiB | relMSE | masked relMSE | mean/ref | Mpaths/s | inconsistencies | leaks |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(v):
        return f"{v:.5g}" if isinstance(v, (int, float)) else "unavailable"
    for e in summary:
        if e.get("failure"):
            stats = e.get("failure_measurements", {})
            peak = stats.get("peak_host_rss_bytes")
            text.append(f"| {e['scene']} | FAILED: {e['failure']} | | | {fmt(stats.get('wall_seconds'))} | | "
                        f"{fmt(peak / 1048576 if peak is not None else None)} | | | | | | |")
        for r in e["runs"]:
            values = [r["integrator"], r["spp"], r["seconds"], r.get("wall_seconds"), r.get("non_transport_seconds"),
                      r["peak_host_rss_bytes"] / 1048576 if r.get("peak_host_rss_bytes") is not None else None,
                      r[METRIC], r["rel_mse_no_glints"], r["mean_ratio"], r.get("mpaths_per_second"),
                      r.get("inconsistencies"), r.get("leaks")]
            text.append("| " + e["scene"] + " | " + " | ".join(v if isinstance(v, str) else fmt(v) for v in values) + " |")
        for failure in e.get("failures", []):
            peak = failure.get("peak_host_rss_bytes")
            text.append(f"| {e['scene']} | FAILED: {failure['integrator']} {failure['spp']}: {failure['reason']} | | | "
                        f"{fmt(failure.get('wall_seconds'))} | | {fmt(peak / 1048576 if peak is not None else None)} | | | | | | |")
    if comparison is not None:
        (out_root / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
        text += ["", f"Equal-error comparison ({comparison['time_key']}); matched {comparison['matched_cases']}/{comparison['required_cases']} cases.",
                 f"Full-set geometric mean: {fmt(comparison['geomean_speedup'])}×. Goal met: {comparison['goal_met']}.",
                 f"Partial geometric mean: {fmt(comparison['partial_geomean_speedup'])}× (diagnostic only).", "",
                 "| scene | target relMSE | baseline s | candidate s | speedup | status |", "|---|---|---|---|---|---|"]
        for row in comparison["scenes"]:
            text.append(f"| {row['scene']} | {fmt(row.get('target_error'))} | {fmt(row.get('baseline_seconds'))} | "
                        f"{fmt(row.get('candidate_seconds'))} | {fmt(row.get('speedup'))} | {row['status']} |")
    (out_root / "summary.md").write_text("\n".join(text) + "\n")


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    budgets = args.budgets or ([16, 64] if args.quick else [16, 64, 256, 1024])
    ref_spp = 256 if args.quick else args.reference_spp
    if any(n <= 0 for n in [*budgets, ref_spp, args.passes]) or args.threads < 0 or args.device < -1 or min(args.seed, args.reference_seed) < 0:
        ap.error("budgets, reference spp and passes must be positive; threads/seeds nonnegative; device >= -1")
    if args.target_error is not None and (not math.isfinite(args.target_error) or args.target_error < 0):
        ap.error("target error must be finite and nonnegative")
    if not math.isfinite(args.goal_speedup) or args.goal_speedup <= 0:
        ap.error("goal speedup must be finite and positive")
    if args.resolution and not re.fullmatch(r"[1-9]\d*x[1-9]\d*", args.resolution):
        ap.error("resolution must be positive WxH")
    try:
        cases = select_cases(args)
    except ValueError as e:
        ap.error(str(e))
    if args.list:
        print(json.dumps([{**case, "settings": run_settings(case, args), "budgets": budgets,
                           "reference_spp": ref_spp} for case in cases], indent=2))
        return 0
    args.binary = args.binary.resolve()
    if not args.binary.is_file():
        ap.error(f"binary not found: {args.binary}")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    provenance = {"code": code_identity(args.binary), "host": environment_identity(args.binary)}
    (out_root / "invocation.json").write_text(json.dumps({"schema": SCHEMA, "provenance": provenance,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": cases, "budgets": budgets, "reference_spp": ref_spp}, indent=2) + "\n")
    rss = rss_method()
    summary, asset_cache = [], {}
    for case in cases:
        name = case["name"]
        d = out_root / name
        d.mkdir(exist_ok=True)
        settings = run_settings(case, args)
        assets = asset_identity(ROOT / case["path"], asset_cache)
        ref_est = args.reference_integrator or case["integrator"]
        ref_out = (args.reference_dir.resolve() if args.reference_dir else out_root) / name / "reference"
        entry = {"scene": name, "suite": case["suite"], "settings": settings, "runs": [], "failures": [],
                 "profiled": os.environ.get("OWE_GPU_PROFILE", "0") not in ("", "0")}
        print(f"{name}: reference {ref_est} × {ref_spp}", flush=True)
        try:
            ref = render(case["path"], ref_out, ref_est, ref_spp, args.backend, seed=args.reference_seed,
                         reuse=args.rescore, binary=args.binary, settings=settings, provenance=provenance,
                         assets=assets, measure_rss=rss, frozen=args.reference_dir is not None)
            ref_image = read_pfm(str(ref_out) + ".pfm")
            if not np.isfinite(ref_image).all():
                raise ValueError("nonfinite reference image")
            mask = glint_mask(ref_image)
            entry["reference"] = {"integrator": ref_est, "spp": ref_spp, "seconds": ref["render_seconds"],
                "mean_Y": ref["mean_Y"], "glint_fraction": float(mask.mean()), "pfm_sha256": file_digest(str(ref_out) + ".pfm"),
                **{k: ref.get(k) for k in ("wall_seconds", "non_transport_seconds", "peak_host_rss_bytes", "rss_method")}}
            entry["measurement_id"] = digest({"schema": SCHEMA, "settings": settings, "assets": assets,
                "host": provenance["host"], "reference_sha256": entry["reference"]["pfm_sha256"], "metric": METRIC})
            own = case["integrator"]
            estimators = args.integrators or ([own] + [e for e in ("sppm", "vcm", "bdpt", "hybrid", "path") if e != own]
                                              if case["suite"] == "transport" else [own])
            for est in estimators:
                for spp in sorted(set(budgets)):
                    out = d / f"{est}_{spp}"
                    try:
                        rec = render(case["path"], out, est, spp, args.backend, seed=args.seed, reuse=args.rescore,
                                     binary=args.binary, settings=settings, provenance=provenance, assets=assets, measure_rss=rss)
                        err = errors(str(out) + ".pfm", ref_image, mask)
                    except (RuntimeError, OSError, ValueError) as e:
                        entry["failures"].append({"integrator": est, "spp": spp, "reason": str(e),
                                                 **getattr(e, "measurements", {})})
                        print(f"  {est} {spp}: FAILED ({e})", flush=True)
                        break
                    stats = rec["statistics"]
                    seconds = rec["render_seconds"]
                    run = {"integrator": est, "spp": spp, "seconds": seconds, "mean_Y": rec["mean_Y"],
                           "paths": stats["paths"], "inconsistencies": stats["region_inconsistencies"],
                           "leaks": stats.get("leaks"),
                           "mpaths_per_second": stats["paths"] / seconds / 1e6 if seconds > 0 else None,
                           **{k: rec.get(k) for k in ("wall_seconds", "non_transport_seconds", "peak_host_rss_bytes", "rss_method")}, **err}
                    if "gpu_profile" in rec:
                        run["gpu_profile"] = rec["gpu_profile"]
                    entry["runs"].append(run)
                    print(f"  {est:6s} {spp:5d}: {seconds:7.2f} s  relMSE {err['rel_mse']}", flush=True)
        except (RuntimeError, OSError, ValueError) as e:
            entry["failure"] = str(e)
            entry["failure_measurements"] = getattr(e, "measurements", {})
            print(f"  reference FAILED ({e})", flush=True)
        summary.append(entry)
        write_reports(out_root, summary)
    comparison = compare_baseline(summary, baseline, target_error=args.target_error,
        time_key="wall_seconds" if args.time == "wall" else "seconds", goal=args.goal_speedup) if baseline is not None else None
    if comparison is not None:
        # A successful selected subset is useful, but cannot certify the production goal.
        comparison["performance_suite_complete"] = all(performance_suite_eligible(records)
                                                       for records in (summary, baseline))
        comparison["goal_met"] = comparison["goal_met"] and comparison["performance_suite_complete"]
    write_reports(out_root, summary, comparison)
    return 1 if any(e.get("failure") or e["failures"] for e in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
