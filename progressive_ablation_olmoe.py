"""
Progressive Ablation Stress Test (OLMoE)
========================================
Research question: At what removal depth does ensemble redundancy break down?

Design:
- For each token position, progressively remove active experts one-by-one
  in descending routing-weight order (highest-weighted first)
- Record loss delta at each removal step (k=1..7 experts removed of 8)
- Run across layers 0, 1, 7, 8, 9, 15 to characterise depth gradient
- Reuses per-token ablation token positions for direct comparability

Compute estimate (RTX 3090):
  4 layers × 7 steps × 500 positions × 1 fwd pass ≈ 14,000 fwd passes
  Expected runtime: 2–4 hours with float16

Output:
  - progressive_ablation_results.json   (raw per-token curves)
  - progressive_ablation_summary.csv    (aggregated stats per layer × step)
  - progressive_ablation_curves.png     (visualisation)
  - progressive_ablation_analysis.md    (human-readable summary report)
"""

import os
import json
import csv
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAYERS          = [0, 1, 7, 8, 9, 15]    # depth gradient + Layer 8 anchor + neighbours
N_POSITIONS     = 500                     # token positions to sample
MIN_CONTEXT_LEN = 32                      # minimum context tokens before target
MAX_CONTEXT_LEN = 256                     # truncate long sequences
RANDOM_SEED     = 42
CHECKPOINT_EVERY = 50                     # save progress every N positions
RESULTS_DIR     = Path("progressive_ablation_results")

MODEL_ID = "allenai/OLMoE-1B-7B-0924"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("progressive_ablation_run.log"),
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using CUDA: {torch.cuda.get_device_name(0)}, "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")
    return device

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(device):
    log.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loading model from {MODEL_ID}")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    log.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Token position sampling (mirrors per-token ablation experiment)
# ---------------------------------------------------------------------------

def sample_token_positions(tokenizer, n_positions=500, seed=42):
    """
    Sample (input_ids, target_position) pairs from WikiText-2.
    Ensures minimum context before each target position.
    Returns list of dicts with keys: input_ids, target_pos, text_snippet
    """
    log.info("Loading WikiText-2 for position sampling...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    rng = random.Random(seed)
    positions = []
    doc_pool = [ex["text"] for ex in dataset if len(ex["text"].strip()) > 100]
    rng.shuffle(doc_pool)

    for text in doc_pool:
        if len(positions) >= n_positions:
            break

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_LEN,
        )
        seq_len = tokens["input_ids"].shape[1]

        if seq_len < MIN_CONTEXT_LEN + 1:
            continue

        # Sample a target position within [MIN_CONTEXT_LEN, seq_len-1]
        target_pos = rng.randint(MIN_CONTEXT_LEN, seq_len - 1)

        positions.append({
            "input_ids": tokens["input_ids"],           # shape [1, seq_len]
            "target_pos": target_pos,
            "seq_len": seq_len,
            "text_snippet": text[:80].replace("\n", " "),
        })

    log.info(f"Sampled {len(positions)} token positions from WikiText-2")
    return positions[:n_positions]

# ---------------------------------------------------------------------------
# Routing capture hook
# ---------------------------------------------------------------------------

class RoutingCapture:
    """
    Attaches hooks to OLMoE MoE layers to capture:
      - router_logits: raw logits before softmax
      - top_k_indices: which experts were selected
      - top_k_weights: routing weights (post-softmax, re-normalised)
    """
    def __init__(self):
        self.captures = {}   # layer_idx -> {indices, weights}
        self._hooks = []

    def attach(self, model, layers):
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            # OLMoE MoE block is at layer.mlp
            hook = layer.mlp.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Re-run gate on input hidden states to capture routing decisions.
            # Reshape to [batch*seq, hidden] before passing to gate.
            hidden = input[0]                             # [batch, seq, hidden]
            bs, seq, hid = hidden.shape
            hidden_2d = hidden.reshape(-1, hid)          # [batch*seq, hidden]

            with torch.no_grad():
                gate_out = module.gate(hidden_2d)

                # gate_out may be raw logits OR tuple (weights, indices)
                if isinstance(gate_out, tuple):
                    # OLMoE gate returns (routing_weights [B*S,k], selected_experts [B*S,k])
                    top_weights, top_indices = gate_out[0], gate_out[1]
                    top_weights = top_weights.float()
                    top_indices = top_indices.long()
                    top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)
                else:
                    # Raw logits — compute top-k ourselves
                    router_logits = gate_out.float()     # [batch*seq, n_experts]
                    weights = torch.softmax(router_logits, dim=-1)
                    top_weights, top_indices = torch.topk(weights, k=8, dim=-1)
                    top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

            # Store as explicitly typed tensors — long for indices, float for weights
            self.captures[layer_idx] = {
                "top_indices": top_indices.detach().cpu().long(),   # [batch*seq, 8]
                "top_weights": top_weights.detach().cpu().float(),  # [batch*seq, 8]
            }
        return hook_fn

    def clear(self):
        self.captures = {}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_token_loss(model, input_ids, target_pos, device):
    """
    Compute cross-entropy loss at target_pos given preceding context.
    Returns scalar loss value.
    """
    input_ids = input_ids.to(device)
    outputs = model(input_ids, labels=input_ids)

    # Extract per-token loss at target position
    # outputs.loss is mean; we need per-token
    logits = outputs.logits                              # [1, seq, vocab]
    shift_logits = logits[0, :-1, :]                    # [seq-1, vocab]
    shift_labels = input_ids[0, 1:]                     # [seq-1]

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(shift_logits, shift_labels)  # [seq-1]

    # target_pos indexes into input_ids; loss at position p is shift[p-1]
    loss_at_target = per_token_loss[target_pos - 1].item()
    return loss_at_target

# ---------------------------------------------------------------------------
# Expert ablation context manager
# ---------------------------------------------------------------------------

class AblateExperts:
    """
    Context manager that zeros the output of specified experts
    in a given MoE layer during the forward pass.
    Works by patching the expert forward methods temporarily.
    """
    def __init__(self, model, layer_idx, expert_indices):
        self.model = model
        self.layer_idx = layer_idx
        self.expert_indices = set(expert_indices)
        self._original_forwards = {}

    def __enter__(self):
        moe = self.model.model.layers[self.layer_idx].mlp
        for eidx in self.expert_indices:
            expert = moe.experts[eidx]
            orig = expert.forward
            self._original_forwards[eidx] = orig

            def make_zero_forward(orig_fn):
                def zero_forward(*args, **kwargs):
                    out = orig_fn(*args, **kwargs)
                    return torch.zeros_like(out)
                return zero_forward

            expert.forward = make_zero_forward(orig)
        return self

    def __exit__(self, *args):
        moe = self.model.model.layers[self.layer_idx].mlp
        for eidx, orig in self._original_forwards.items():
            moe.experts[eidx].forward = orig

# ---------------------------------------------------------------------------
# Core progressive ablation
# ---------------------------------------------------------------------------

def progressive_ablation_at_position(
    model, input_ids, target_pos, layer_idx, routing_capture, device
):
    """
    For a single token position and layer:
    1. Get baseline loss
    2. Get routing info (which 8 experts, in what weight order)
    3. Progressively remove experts highest-weight-first
    4. Record loss at each step k=1..7

    Returns dict:
        baseline_loss: float
        steps: list of dicts [{n_removed, expert_idx, cumulative_experts_removed,
                                loss, delta_loss, routing_weight}]
        expert_order: list of expert indices in removal order
        routing_weights: list of weights in removal order
    """
    # Baseline
    routing_capture.clear()
    baseline_loss = compute_token_loss(model, input_ids, target_pos, device)

    # Get routing for this position from the capture
    # Position in the flattened [batch*seq] tensor
    cap = routing_capture.captures.get(layer_idx)
    if cap is None:
        return None

    # target_pos - 1 because model sees all tokens to predict next
    flat_pos = target_pos - 1
    if flat_pos >= cap["top_indices"].shape[0]:
        flat_pos = cap["top_indices"].shape[0] - 1

    expert_order_tensor = cap["top_indices"][flat_pos]     # [8]
    weight_order_tensor = cap["top_weights"][flat_pos]     # [8]

    # Sort descending by weight (highest weight removed first)
    sorted_order = torch.argsort(weight_order_tensor, descending=True)
    # Explicitly convert to plain Python ints/floats — avoids unhashable tensor issues
    expert_order    = [int(x) for x in expert_order_tensor[sorted_order].tolist()]
    routing_weights = [float(x) for x in weight_order_tensor[sorted_order].tolist()]

    steps = []
    ablated_so_far = []

    for k in range(1, 8):   # remove 1..7 of the 8 active experts
        ablated_so_far.append(expert_order[k - 1])

        with AblateExperts(model, layer_idx, ablated_so_far):
            loss_k = compute_token_loss(model, input_ids, target_pos, device)

        steps.append({
            "n_removed": k,
            "expert_removed_this_step": expert_order[k - 1],
            "cumulative_experts_removed": list(ablated_so_far),
            "routing_weight_of_removed": routing_weights[k - 1],
            "loss": loss_k,
            "delta_loss": loss_k - baseline_loss,
            "pct_weight_removed": sum(routing_weights[:k]),
        })

    return {
        "baseline_loss": baseline_loss,
        "expert_order": expert_order,
        "routing_weights": routing_weights,
        "steps": steps,
    }

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(model, tokenizer, device, n_positions=500, resume=True):
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = RESULTS_DIR / "checkpoint.json"

    # Load checkpoint if resuming
    completed = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        log.info(f"Resuming from checkpoint: {len(completed)} positions done")

    # Sample token positions
    positions = sample_token_positions(tokenizer, n_positions=n_positions)

    # Set up routing capture (hooks always attached)
    routing_capture = RoutingCapture()
    routing_capture.attach(model, LAYERS)
    log.info(f"Routing hooks attached to layers {LAYERS}")

    # Warm-up pass to populate routing captures
    log.info("Warming up routing captures...")
    dummy = positions[0]
    _ = compute_token_loss(model, dummy["input_ids"], dummy["target_pos"], device)

    all_results = dict(completed)
    start_time = time.time()

    for pos_idx, pos_data in enumerate(positions):
        pos_key = str(pos_idx)
        if pos_key in all_results:
            continue

        input_ids  = pos_data["input_ids"]
        target_pos = pos_data["target_pos"]

        # Single baseline forward to populate routing captures for all layers
        routing_capture.clear()
        _ = compute_token_loss(model, input_ids, target_pos, device)

        pos_result = {
            "pos_idx": pos_idx,
            "target_pos": target_pos,
            "seq_len": pos_data["seq_len"],
            "text_snippet": pos_data["text_snippet"],
            "layers": {}
        }

        for layer_idx in LAYERS:
            try:
                layer_result = progressive_ablation_at_position(
                    model, input_ids, target_pos, layer_idx,
                    routing_capture, device
                )
                if layer_result is not None:
                    pos_result["layers"][str(layer_idx)] = layer_result
            except Exception as e:
                log.warning(f"Position {pos_idx}, Layer {layer_idx} failed: {e}")
                continue

        all_results[pos_key] = pos_result

        # Progress logging
        elapsed = time.time() - start_time
        done = pos_idx + 1 - len(completed)
        remaining = n_positions - len(all_results)
        rate = done / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else 0
        log.info(
            f"Position {pos_idx+1}/{n_positions} | "
            f"Elapsed: {elapsed/60:.1f}m | "
            f"ETA: {eta/60:.1f}m"
        )

        # Checkpoint
        if (pos_idx + 1) % CHECKPOINT_EVERY == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(all_results, f)
            log.info(f"Checkpoint saved ({len(all_results)} positions)")

    # Final save
    results_path = RESULTS_DIR / "progressive_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    routing_capture.remove()
    return all_results

# ---------------------------------------------------------------------------
# Analysis & summary
# ---------------------------------------------------------------------------

def analyse_results(all_results):
    """
    Aggregate per-token progressive curves into layer-level statistics.
    Returns structured summary suitable for plotting and the markdown report.
    """
    # Structure: layer -> step_k -> list of delta_loss values
    layer_step_deltas = defaultdict(lambda: defaultdict(list))
    layer_step_pct_weight = defaultdict(lambda: defaultdict(list))
    layer_baselines = defaultdict(list)

    for pos_key, pos_data in all_results.items():
        for layer_str, layer_data in pos_data.get("layers", {}).items():
            layer_idx = int(layer_str)
            layer_baselines[layer_idx].append(layer_data["baseline_loss"])

            for step in layer_data["steps"]:
                k = step["n_removed"]
                layer_step_deltas[layer_idx][k].append(step["delta_loss"])
                layer_step_pct_weight[layer_idx][k].append(step["pct_weight_removed"])

    summary = {}
    for layer_idx in LAYERS:
        summary[layer_idx] = {
            "baseline_loss_mean": float(np.mean(layer_baselines[layer_idx])),
            "baseline_loss_std":  float(np.std(layer_baselines[layer_idx])),
            "n_positions": len(layer_baselines[layer_idx]),
            "steps": {}
        }
        for k in range(1, 8):
            deltas = layer_step_deltas[layer_idx][k]
            pcts   = layer_step_pct_weight[layer_idx][k]
            if not deltas:
                continue
            arr = np.array(deltas)
            summary[layer_idx]["steps"][k] = {
                "n_removed": k,
                "delta_loss_mean":   float(np.mean(arr)),
                "delta_loss_std":    float(np.std(arr)),
                "delta_loss_median": float(np.median(arr)),
                "delta_loss_p25":    float(np.percentile(arr, 25)),
                "delta_loss_p75":    float(np.percentile(arr, 75)),
                "delta_loss_p95":    float(np.percentile(arr, 95)),
                "pct_positive":      float(np.mean(arr > 0)),
                "pct_weight_removed_mean": float(np.mean(pcts)),
                "n_samples":         len(deltas),
            }

    return summary

def find_cliff(step_data):
    """
    Identify the removal step where loss delta first exceeds 0.1 nats
    (i.e., the ensemble redundancy 'cliff').
    Returns the step index k, or None if no cliff found.
    """
    for k in range(1, 8):
        if k in step_data:
            if step_data[k]["delta_loss_mean"] > 0.1:
                return k
    return None

def save_summary_csv(summary, out_path):
    rows = []
    for layer_idx, layer_data in summary.items():
        for k, step_data in layer_data["steps"].items():
            rows.append({
                "layer": layer_idx,
                "n_removed": k,
                **{kk: vv for kk, vv in step_data.items()},
                "baseline_loss_mean": layer_data["baseline_loss_mean"],
            })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Summary CSV saved to {out_path}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(summary, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log.warning("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Progressive ablation: Progressive Expert Ablation — Ensemble Redundancy Stress Test\n"
        "OLMoE-1B-7B-0924 | Experts removed in descending routing-weight order",
        fontsize=12, fontweight="bold"
    )

    colors = {0: "#4e79a7", 1: "#59a14f", 7: "#f28e2b", 8: "#e15759", 9: "#edc948", 15: "#76b7b2"}
    layer_labels = {0: "Layer 0 (input)", 1: "Layer 1", 7: "Layer 7 (early-mid)", 8: "Layer 8 (ensemble)", 9: "Layer 9", 15: "Layer 15 (output)"}
    steps_x = list(range(1, 8))

    # Panel 1: Mean delta loss curve
    ax1 = axes[0]
    for layer_idx in LAYERS:
        if layer_idx not in summary:
            continue
        ys     = [summary[layer_idx]["steps"].get(k, {}).get("delta_loss_mean", np.nan) for k in steps_x]
        yerr   = [summary[layer_idx]["steps"].get(k, {}).get("delta_loss_std",  np.nan) for k in steps_x]
        ax1.plot(steps_x, ys, "o-", color=colors[layer_idx],
                 label=layer_labels[layer_idx], linewidth=2, markersize=6)
        ax1.fill_between(
            steps_x,
            [y - e for y, e in zip(ys, yerr)],
            [y + e for y, e in zip(ys, yerr)],
            alpha=0.15, color=colors[layer_idx]
        )

    ax1.axhline(0.1, color="gray", linestyle="--", linewidth=1, label="Cliff threshold (0.1 nats)")
    ax1.set_xlabel("Number of experts removed (of 8 active)", fontsize=11)
    ax1.set_ylabel("Mean Δ Loss (nats)", fontsize=11)
    ax1.set_title("Loss degradation vs. removal depth", fontsize=11)
    ax1.set_xticks(steps_x)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: P95 delta loss (stress-test view — worst-case positions)
    ax2 = axes[1]
    for layer_idx in LAYERS:
        if layer_idx not in summary:
            continue
        ys_p95 = [summary[layer_idx]["steps"].get(k, {}).get("delta_loss_p95", np.nan) for k in steps_x]
        ys_med = [summary[layer_idx]["steps"].get(k, {}).get("delta_loss_median", np.nan) for k in steps_x]
        ax2.plot(steps_x, ys_p95, "o--", color=colors[layer_idx],
                 label=f"Layer {layer_idx} P95", linewidth=1.5, markersize=5, alpha=0.7)
        ax2.plot(steps_x, ys_med, "o-", color=colors[layer_idx],
                 label=f"Layer {layer_idx} median", linewidth=2, markersize=6)

    ax2.axhline(0.1, color="gray", linestyle="--", linewidth=1, label="Cliff threshold (0.1 nats)")
    ax2.set_xlabel("Number of experts removed (of 8 active)", fontsize=11)
    ax2.set_ylabel("Δ Loss (nats)", fontsize=11)
    ax2.set_title("Median + P95 loss degradation (worst-case stress)", fontsize=11)
    ax2.set_xticks(steps_x)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved to {out_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(summary, all_results, out_path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_positions = len(all_results)

    lines = [
        f"# Progressive ablation: Progressive Ablation Stress Test",
        f"",
        f"**Generated:** {ts}  ",
        f"**Model:** OLMoE-1B-7B-0924  ",
        f"**Layers:** {LAYERS}  ",
        f"**Token positions:** {total_positions}  ",
        f"**Ablation order:** Descending routing weight (highest-weighted expert removed first)  ",
        f"",
        f"---",
        f"",
        f"## Research Question",
        f"",
        f"At what removal depth does ensemble redundancy break down? Single-expert "
        f"ablation showed near-zero functional impact at most layers in OLMoE "
        f"(|d| < 0.15 across all 20 metric-layer combinations in the per-token "
        f"ablation experiment). Progressive ablation tests how much redundancy "
        f"capacity the ensemble has -- removing experts one-by-one in "
        f"routing-weight descending order until the system fails.",
        f"",
        f"**Cliff threshold:** Δ Loss > 0.1 nats (mean across positions) is taken as the "
        f"operationalised failure point of the redundancy buffer.",
        f"",
        f"---",
        f"",
        f"## Results by Layer",
        f"",
    ]

    for layer_idx in LAYERS:
        if layer_idx not in summary:
            lines.append(f"### Layer {layer_idx}\n\n*No data collected.*\n")
            continue

        ld = summary[layer_idx]
        cliff = find_cliff(ld["steps"])
        cliff_str = f"k={cliff}" if cliff else "not reached (< 0.1 nats even at k=7)"

        lines += [
            f"### Layer {layer_idx}",
            f"",
            f"- **Baseline loss:** {ld['baseline_loss_mean']:.4f} ± {ld['baseline_loss_std']:.4f} nats",
            f"- **N positions:** {ld['n_positions']}",
            f"- **Cliff (first Δ Loss > 0.1):** {cliff_str}",
            f"",
            f"| k removed | Mean Δ Loss | Median Δ Loss | P95 Δ Loss | % Positions Positive | % Weight Removed |",
            f"|-----------|-------------|---------------|------------|----------------------|------------------|",
        ]

        for k in range(1, 8):
            if k not in ld["steps"]:
                continue
            s = ld["steps"][k]
            lines.append(
                f"| {k} | {s['delta_loss_mean']:+.4f} | {s['delta_loss_median']:+.4f} | "
                f"{s['delta_loss_p95']:+.4f} | {s['pct_positive']*100:.1f}% | "
                f"{s['pct_weight_removed_mean']*100:.1f}% |"
            )
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## Interpretation",
        f"",
        f"### Depth Gradient",
        f"",
    ]

    # Auto-generate cliff comparison
    cliff_summary = {}
    for layer_idx in LAYERS:
        if layer_idx in summary:
            cliff_summary[layer_idx] = find_cliff(summary[layer_idx]["steps"])

    for layer_idx in LAYERS:
        cliff = cliff_summary.get(layer_idx)
        cliff_str = f"cliff at k={cliff}" if cliff else "no cliff (full redundancy buffer intact at k=7)"
        lines.append(f"- **Layer {layer_idx}:** {cliff_str}")

    lines += [
        f"",
        f"### Connection to single-expert ablation",
        f"",
        f"The per-token ablation experiment established that single-expert ablation "
        f"yields |d| < 0.15 across all metric-layer combinations on OLMoE, "
        f"indicating near-complete redundancy within the top-k activated set at "
        f"most layers. Progressive ablation tests the outer limit: how much of "
        f"the ensemble can be removed before the redundancy buffer is exhausted.",
        f"",
        f"If the cliff appears late (k=6 or k=7), this confirms that OLMoE's "
        f"top-8 routing with load balancing creates deep functional redundancy -- "
        f"the model has learned more structure than it actively uses at any given "
        f"token position.",
        f"",
        f"If the depth gradient shows Layer 15 cliffing substantially earlier "
        f"than mid-layers, this corroborates the depth-concentrated functional "
        f"signal at Layer 15 reported in the per-token routing-weight control "
        f"experiment -- later layers rely on more specialised, less substitutable "
        f"experts.",
        f"",
        f"---",
        f"",
        f"*Report auto-generated by progressive_ablation_olmoe.py*",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Markdown report saved to {out_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Progressive Ablation Stress Test")
    parser.add_argument("--n-positions", type=int, default=N_POSITIONS,
                        help=f"Number of token positions to sample (default: {N_POSITIONS})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignoring any checkpoint")
    parser.add_argument("--analyse-only", action="store_true",
                        help="Skip data collection, just re-analyse existing results")
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS,
                        help="Layers to ablate (default: 0 7 8 15)")
    return parser.parse_args()

def main():
    args = parse_args()
    global LAYERS
    LAYERS = args.layers

    log.info("=" * 60)
    log.info("Progressive ablation: Progressive Ablation Stress Test")
    log.info(f"Layers: {LAYERS} | Positions: {args.n_positions} | ~{len(LAYERS)*7*args.n_positions:,} fwd passes")
    log.info("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    device = get_device()

    results_path = RESULTS_DIR / "progressive_ablation_results.json"

    if args.analyse_only:
        if not results_path.exists():
            log.error("No results file found. Run without --analyse-only first.")
            return
        log.info("Loading existing results for analysis...")
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        model, tokenizer = load_model_and_tokenizer(device)
        all_results = run_experiment(
            model, tokenizer, device,
            n_positions=args.n_positions,
            resume=not args.no_resume,
        )

    log.info("Analysing results...")
    summary = analyse_results(all_results)

    save_summary_csv(summary, RESULTS_DIR / "progressive_ablation_summary.csv")
    plot_curves(summary, RESULTS_DIR / "progressive_ablation_curves.png")
    generate_markdown_report(summary, all_results, RESULTS_DIR / "progressive_ablation_analysis.md")

    # Print quick summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY — Cliff Detection (first step where mean Δ Loss > 0.1 nats)")
    print("=" * 60)
    for layer_idx in LAYERS:
        if layer_idx not in summary:
            print(f"  Layer {layer_idx}: no data")
            continue
        cliff = find_cliff(summary[layer_idx]["steps"])
        k7_delta = summary[layer_idx]["steps"].get(7, {}).get("delta_loss_mean", float("nan"))
        cliff_str = f"k={cliff}" if cliff else f"none (k=7 mean={k7_delta:+.4f})"
        print(f"  Layer {layer_idx}: {cliff_str}")
    print("=" * 60)
    print(f"\nFull results in: {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
