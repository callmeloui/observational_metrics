#!/usr/bin/env python3
"""
run_per_token_ablation_multi_layer_olmoe.py -- per-token ablation, sweep across 5 OLMoE layers
======================================================================
Invokes per_token_ablation_olmoe.py once per layer in [0, 4, 7, 11, 15] and copies
each per-layer JSON output into a single consolidated directory.

The worker script (per_token_ablation_olmoe.py) is invoked as a subprocess with
--layer and --device passed on the command line; no source-level patching.

Usage:
    python run_per_token_ablation_multi_layer_olmoe.py            # default: cuda
    python run_per_token_ablation_multi_layer_olmoe.py --device cpu
    python run_per_token_ablation_multi_layer_olmoe.py --layers 0 7 15

Output:
    olmoe_per_token_ablation_all_layers/
        layer_00_results.json
        layer_04_results.json
        layer_07_results.json
        layer_11_results.json
        layer_15_results.json
        run_summary.json

Run from the repo root so the worker script (per_token_ablation_olmoe.py) is found
at the expected relative path.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LAYERS = [0, 4, 7, 11, 15]

REPO_ROOT    = Path(__file__).resolve().parent.parent
WORKER       = REPO_ROOT / "per_token_ablation_olmoe.py"
BASE_RESULTS = REPO_ROOT / "olmoe_per_token_ablation_results"
OUT_DIR      = REPO_ROOT / "olmoe_per_token_ablation_all_layers"


def run_layer(layer_idx: int, device: str) -> bool:
    print(f"\n{'#' * 70}")
    print(f"#  PER-TOKEN ABLATION (OLMoE) -- LAYER {layer_idx}")
    print(f"#  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(WORKER),
         "--layer",  str(layer_idx),
         "--device", device],
        cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - start

    success = result.returncode == 0
    json_src = BASE_RESULTS / "per_token_ablation_results.json"

    if success and json_src.exists():
        dest = OUT_DIR / f"layer_{layer_idx:02d}_results.json"
        shutil.copy2(json_src, dest)
        print(f"\n  Saved: {dest.name}")
    elif not success:
        print(f"\n  WARNING: Layer {layer_idx} failed (exit code {result.returncode})")
    else:
        print(f"\n  WARNING: Layer {layer_idx} succeeded but {json_src} not found")

    status = "SUCCESS" if success else "FAILED"
    print(f"  Layer {layer_idx}: {status} in {elapsed / 60:.1f} minutes")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    print("=" * 70)
    print("PER-TOKEN ABLATION FULL LAYER SWEEP -- OLMoE-1B-7B")
    print(f"Worker  : {WORKER.name}")
    print(f"Layers  : {args.layers}")
    print(f"Device  : {args.device}")
    print(f"Output  : {OUT_DIR.name}/")
    print("=" * 70)

    if not WORKER.exists():
        print(f"ERROR: {WORKER} not found.")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    run_results = {}
    total_start = time.time()
    for layer in args.layers:
        run_results[layer] = run_layer(layer, args.device)

    total_elapsed = time.time() - total_start
    summary = {
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":         "allenai/OLMoE-1B-7B-0924",
        "layers_run":    args.layers,
        "device":        args.device,
        "results":       {str(l): ("success" if s else "failed")
                          for l, s in run_results.items()},
        "total_hours":   round(total_elapsed / 3600, 2),
    }
    with open(OUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("ALL LAYERS COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time : {total_elapsed / 3600:.1f} hours")
    print(f"  Output dir : {OUT_DIR}/")
    for layer in args.layers:
        s = "+" if run_results.get(layer) else "X"
        print(f"  {s}  Layer {layer:2d}: layer_{layer:02d}_results.json")


if __name__ == "__main__":
    main()
