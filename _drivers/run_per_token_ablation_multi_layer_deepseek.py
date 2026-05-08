#!/usr/bin/env python3
"""
run_per_token_ablation_multi_layer_deepseek.py -- per-token ablation, sweep across 5 layers of
DeepSeek-V2-Lite
======================================================================
Invokes per_token_ablation_deepseek.py once per layer in [1, 7, 13, 20, 26] and
copies each per-layer JSON output into a single consolidated directory.

DeepSeek-V2-Lite has 27 layers; layer 0 is dense, layers 1-26 are MoE.
The chosen layer set spans the MoE range:
    Layer  1 -- earliest MoE layer
    Layer  7 -- early-mid
    Layer 13 -- middle
    Layer 20 -- late-mid
    Layer 26 -- final layer

Usage:
    python run_per_token_ablation_multi_layer_deepseek.py
    python run_per_token_ablation_multi_layer_deepseek.py --layers 1 13 26

Output:
    deepseek_per_token_ablation_all_layers/
        layer_01_results.json
        layer_07_results.json
        layer_13_results.json
        layer_20_results.json
        layer_26_results.json
        run_summary.json

Run from the repo root so the worker script is found at the expected
relative path.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LAYERS = [1, 7, 13, 20, 26]

REPO_ROOT    = Path(__file__).resolve().parent.parent
WORKER       = REPO_ROOT / "per_token_ablation_deepseek.py"
BASE_RESULTS = REPO_ROOT / "deepseek_per_token_ablation_results"
OUT_DIR      = REPO_ROOT / "deepseek_per_token_ablation_all_layers"


def run_layer(layer_idx: int, device: str) -> bool:
    print(f"\n{'#' * 70}")
    print(f"#  PER-TOKEN ABLATION (DeepSeek-V2-Lite) -- LAYER {layer_idx}")
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
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("=" * 70)
    print("PER-TOKEN ABLATION FULL LAYER SWEEP -- DeepSeek-V2-Lite")
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
        "model":         "deepseek-ai/DeepSeek-V2-Lite",
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
