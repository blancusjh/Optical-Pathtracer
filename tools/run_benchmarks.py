#!/usr/bin/env python3
"""Runs the transport benchmarks in benchmarks/ and writes out/benchmarks/summary.{json,md}.

For each benchmark scene: a long reference (the scene's own integrator, default sppm, many
iterations), then each estimator at increasing sample budgets. Every run is a separate `owe render`
from scratch, so its record's time is the estimator's cost at that budget. Errors are measured on
the raw linear images (PFM, never the display): relative MSE against the reference
(Σ (x − r)² / (r² + 0.01·mean(r)²) over pixels and channels), and the image mean's ratio. A second
relative MSE leaves out the sun's glints: pixels within 4 of one the reference shows 20× brighter
than its 9 × 9 median (and 10× the image mean). A glint is the sun seen through or in glass or water,
a path eye → specular⁺ → sun with no diffuse vertex; every estimator here samples it only by chance
(it is the same camera path in each), so at small budgets it dominates the error without saying
anything about how the estimator transports caustics. (Where such points hold most of the reference's
light, as in a lens's image of a small source, they are the measurement and nothing is masked.)
Display images use the scene's fixed exposure, identical for every estimator.

  tools/run_benchmarks.py [--only NAME ...] [--backend gpu] [--reference-spp 8192] [--quick] [--rescore]

--rescore measures the renders already in out/benchmarks again (and renders only those missing).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWE = os.path.join(ROOT, "build", "owe")


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
    return data.reshape(h, w, 3).astype(np.float64)


def render(scene, out, integrator, spp, backend, seed=1, reuse=False):
    if reuse and os.path.exists(out + ".pfm") and os.path.exists(out + ".json"):
        with open(out + ".json") as f:
            return json.load(f)
    cmd = [OWE, "render", scene, "--integrator", integrator, "--spp", str(spp), "--backend", backend,
           "--seed", str(seed), "--out", out]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    with open(out + ".json") as f:
        record = json.load(f)
    record["wall_seconds"] = time.time() - t0
    return record


def glint_mask(ref):
    """Pixels near the sun's glints in the reference (see the module's note)."""
    m = ref.max(axis=2)
    median = np.median(sliding_window_view(np.pad(m, 4, mode="edge"), (9, 9)), axis=(2, 3))
    seeds = (m > 20 * median) & (m > 10 * m.mean())
    mask = sliding_window_view(np.pad(seeds, 4), (9, 9)).any(axis=(2, 3))
    # Where the bright points are most of the image (a lens's or mirror's image of a small source),
    # they are the measurement, not glints: nothing is masked.
    return mask if m[mask].sum() < 0.5 * m.sum() else np.zeros_like(mask)


def errors(pfm, ref, mask):
    a = read_pfm(pfm)
    mean_ref = ref.mean()
    rel = (a - ref) ** 2 / (ref * ref + 0.01 * mean_ref * mean_ref)
    return {"rel_mse": float(rel.mean()), "rel_mse_no_glints": float(rel[~mask].mean()),
            "mean_ratio": float(a.mean() / mean_ref) if mean_ref else float("nan")}


def scene_integrator(path):
    integrator = "path"
    with open(path) as f:
        for line in f:
            if line.startswith("render") and "integrator" in line:
                integrator = line.split("integrator")[1].split("=")[1].split()[0].strip("}")
    return integrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--backend", default="gpu")
    ap.add_argument("--reference-spp", type=int, default=8192)
    ap.add_argument("--quick", action="store_true", help="small budgets, for checking the runner")
    ap.add_argument("--rescore", action="store_true", help="measure existing renders again")
    args = ap.parse_args()
    budgets = [16, 64] if args.quick else [16, 64, 256, 1024]
    ref_spp = 256 if args.quick else args.reference_spp
    out_root = os.path.join(ROOT, "out", "benchmarks")
    os.makedirs(out_root, exist_ok=True)
    scenes = sorted(f for f in os.listdir(os.path.join(ROOT, "benchmarks")) if f.endswith(".owe"))
    if args.only:
        scenes = [s for s in scenes if s[:-4] in args.only]
    summary = []
    for s in scenes:
        name = s[:-4]
        path = os.path.join("benchmarks", s)
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        own = scene_integrator(os.path.join(ROOT, path))
        print(f"{name}: reference {own} × {ref_spp}", flush=True)
        ref = render(path, os.path.join(d, "reference"), own, ref_spp, args.backend, seed=1000, reuse=args.rescore)
        ref_image = read_pfm(os.path.join(d, "reference.pfm"))
        mask = glint_mask(ref_image)
        entry = {"scene": name, "reference": {"integrator": own, "spp": ref_spp, "seconds": ref["render_seconds"],
                                               "mean_Y": ref["mean_Y"], "glint_fraction": float(mask.mean())},
                 "runs": []}
        estimators = [own] + [e for e in ("sppm", "vcm", "bdpt", "hybrid", "path") if e != own]
        for est in estimators:
            for spp in budgets:
                out = os.path.join(d, f"{est}_{spp}")
                try:
                    rec = render(path, out, est, spp, args.backend, reuse=args.rescore)
                except RuntimeError as e:
                    print(f"  {est} {spp}: skipped ({str(e).splitlines()[-1]})", flush=True)
                    break
                err = errors(out + ".pfm", ref_image, mask)
                run = {"integrator": est, "spp": spp, "seconds": rec["render_seconds"], "mean_Y": rec["mean_Y"],
                       "paths": rec["statistics"]["paths"], "inconsistencies": rec["statistics"]["region_inconsistencies"],
                       **err}
                entry["runs"].append(run)
                print(f"  {est:6s} {spp:5d}: {run['seconds']:7.2f} s  relMSE {run['rel_mse']:.4g}  "
                      f"(no glints {run['rel_mse_no_glints']:.4g})  mean/ref {run['mean_ratio']:.4f}", flush=True)
        summary.append(entry)
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("| scene | estimator | spp | seconds | relMSE | relMSE, glints masked | mean / reference |\n"
                "|---|---|---|---|---|---|---|\n")
        for e in summary:
            for r in e["runs"]:
                f.write(f"| {e['scene']} | {r['integrator']} | {r['spp']} | {r['seconds']:.2f} | {r['rel_mse']:.4g} | "
                        f"{r['rel_mse_no_glints']:.4g} ({100 * e['reference']['glint_fraction']:.2g}% masked) | "
                        f"{r['mean_ratio']:.4f} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
