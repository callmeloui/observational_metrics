#!/usr/bin/env python3
"""
run_redistribution_all_layers.py
=================================
Runs redistribution_olmoe.py across multiple layers and produces a
cross-layer summary table when all runs are complete.

Usage (from repo root):
  # Default (paper layers): 0 4 7 8 11 15
  python _drivers/run_redistribution_all_layers.py --device cuda

  # Subset for testing:
  python _drivers/run_redistribution_all_layers.py --device cuda --layers 0 8 15

  # Mac CPU overnight:
  python _drivers/run_redistribution_all_layers.py --device cpu --layers 0 7 8 15

Output (written to --output_dir, default repo root):
  redistribution_v2_layer{L}_wikitext.json  -- per layer (from inner script)
  redistribution_v2_layer{L}_wikitext.md    -- per layer (from inner script)
  redistribution_v2_summary.md              -- cross-layer table
  redistribution_v2_summary.json            -- cross-layer data

This driver runs the worker as a subprocess; both must be in the repo tree.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+",
                    default=[0, 4, 7, 8, 11, 15],
                    help="Layer indices to run (default: 0 4 7 8 11 15)")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--corpus", type=str, default="wikitext")
parser.add_argument("--samples", type=int, default=200)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--script", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "redistribution_olmoe.py"),
                    help="Path to the per-layer redistribution worker script")
args = parser.parse_args()

OUT_DIR = Path(args.output_dir)
SCRIPT  = Path(args.script)

if not SCRIPT.exists():
    print(f"ERROR: {SCRIPT} not found. Run this from the same directory.")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Multi-layer redistribution runner")
print(f"  Layers : {args.layers}")
print(f"  Device : {args.device}")
print(f"  Corpus : {args.corpus}")
print(f"  Samples: {args.samples}")
print(f"{'='*60}\n")

results_per_layer = {}
failed_layers = []
total_start = time.time()

for layer in args.layers:
    print(f"\n{'─'*50}")
    print(f"  Starting layer {layer}  ({args.layers.index(layer)+1}/{len(args.layers)})")
    print(f"{'─'*50}")

    layer_start = time.time()
    cmd = [
        sys.executable, str(SCRIPT),
        "--layer",      str(layer),
        "--device",     args.device,
        "--corpus",     args.corpus,
        "--samples",    str(args.samples),
        "--output_dir", str(OUT_DIR),
    ]

    result = subprocess.run(cmd, text=True)

    elapsed = time.time() - layer_start
    if result.returncode != 0:
        print(f"  ✗ Layer {layer} FAILED after {elapsed:.0f}s")
        failed_layers.append(layer)
        continue

    print(f"  ✓ Layer {layer} done in {elapsed:.0f}s")

    # Load the JSON output
    json_path = OUT_DIR / f"redistribution_v2_layer{layer}_{args.corpus}.json"
    if json_path.exists():
        with open(json_path) as f:
            results_per_layer[layer] = json.load(f)
    else:
        print(f"  ✗ Expected output not found: {json_path}")
        failed_layers.append(layer)

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"All layers complete in {total_elapsed/60:.1f} min")
if failed_layers:
    print(f"  Failed layers: {failed_layers}")
print(f"{'='*60}\n")

if not results_per_layer:
    print("No results to summarise.")
    sys.exit(1)

# ── Cross-layer summary ───────────────────────────────────────────────────────

import numpy as np

def layer_summary(data):
    experts = data["experts"]
    n = len(experts)
    rw  = np.array([experts[str(i)]["mean_routing_weight"]  for i in range(n)])
    gap = np.array([experts[str(i)]["mean_gap_norm"]         for i in range(n)])
    cos = np.array([experts[str(i)]["compensation_cosine"]   for i in range(n)])
    rel = np.array([experts[str(i)]["relative_compensation"] for i in range(n)])
    cov = np.array([experts[str(i)]["token_coverage"]        for i in range(n)])
    c   = data["correlations"]
    return {
        "n_tokens":              data["n_tokens_total"],
        "mean_gap_norm":         float(gap.mean()),
        "std_gap_norm":          float(gap.std()),
        "mean_rel_compensation": float(rel.mean()),
        "mean_comp_cosine":      float(cos.mean()),
        "mean_routing_weight":   float(rw.mean()),
        "coverage_cv":           float(cov.std() / cov.mean()),  # coefficient of variation
        "rho_rw_gap":            c["routing_weight_vs_gap_norm"]["rho"],
        "p_rw_gap":              c["routing_weight_vs_gap_norm"]["p"],
        "rho_gap_comp":          c["gap_norm_vs_relative_compensation"]["rho"],
        "p_gap_comp":            c["gap_norm_vs_relative_compensation"]["p"],
        "rho_rw_comp":           c["routing_weight_vs_relative_compensation"]["rho"],
        "p_rw_comp":             c["routing_weight_vs_relative_compensation"]["p"],
        "rho_cos_gap":           c["compensation_cosine_vs_gap_norm"]["rho"],
        "p_cos_gap":             c["compensation_cosine_vs_gap_norm"]["p"],
    }

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

summaries = {layer: layer_summary(data)
             for layer, data in sorted(results_per_layer.items())}

# Save summary JSON
summary_json = OUT_DIR / "redistribution_v2_summary.json"
with open(summary_json, "w") as f:
    json.dump(summaries, f, indent=2)
print(f"Saved: {summary_json}")

# ── Markdown summary ──────────────────────────────────────────────────────────

md = [
    f"# Redistribution v2 — Cross-Layer Summary",
    f"\n**Corpus**: {args.corpus}  |  **Samples**: {args.samples}  |  **Layers**: {sorted(summaries.keys())}\n",

    "## Functional Importance by Layer\n",
    "Mean gap norm = mean ||baseline_output - ablated_output|| across tokens where expert was active.",
    "Higher = experts contribute larger, more distinctive vectors.\n",
    "| Layer | Tokens | Mean gap norm | Std gap norm | Mean rel_comp | Mean comp_cos |",
    "|-------|--------|--------------|--------------|---------------|---------------|",
]
for layer, s in summaries.items():
    md.append(
        f"| {layer:2d} | {s['n_tokens']:,} | {s['mean_gap_norm']:.4f} | "
        f"{s['std_gap_norm']:.4f} | {s['mean_rel_compensation']:.4f} | "
        f"{s['mean_comp_cosine']:.4f} |"
    )

md += [
    "",
    "## Spearman Correlations by Layer\n",
    "Key question: does routing weight predict functional importance (rho_rw_gap)?",
    "Expected depth gradient: near-zero at early layers, positive at deep layers.\n",
    "| Layer | rw→gap ρ | sig | gap→comp ρ | sig | rw→comp ρ | sig | cos→gap ρ | sig |",
    "|-------|----------|-----|-----------|-----|----------|-----|----------|-----|",
]
for layer, s in summaries.items():
    md.append(
        f"| {layer:2d} "
        f"| {s['rho_rw_gap']:+.3f} | {sig_stars(s['p_rw_gap'])} "
        f"| {s['rho_gap_comp']:+.3f} | {sig_stars(s['p_gap_comp'])} "
        f"| {s['rho_rw_comp']:+.3f} | {sig_stars(s['p_rw_comp'])} "
        f"| {s['rho_cos_gap']:+.3f} | {sig_stars(s['p_cos_gap'])} |"
    )

md += [
    "",
    "## Depth Gradient Interpretation\n",
    "- **rw->gap increasing with depth**: routing weight becomes a better predictor of functional",
    "  importance at deeper layers. At early layers routing reflects load distribution,",
    "  not specialisation.",
    "- **gap->comp positive**: experts with larger functional contributions are more compensated.",
    "  Likely a geometric effect -- high-norm contributions are in directions already covered",
    "  by the ensemble, not active redistribution (the router cannot adapt at inference time).",
    "- **Absolute gap norm increasing with depth**: experts contribute larger, more differentiated",
    "  vectors at deeper layers. Consistent with the per-token ablation depth gradient.",
]

summary_md = OUT_DIR / "redistribution_v2_summary.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"Saved: {summary_md}")

# Print table to console
print("\n" + "\n".join(md[3:]))
