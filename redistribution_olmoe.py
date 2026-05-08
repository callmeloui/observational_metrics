#!/usr/bin/env python3
"""
redistribution_olmoe.py
========================
Measures functional redistribution on OLMoE-1B-7B: how much do the remaining
experts' combined output shift to compensate when a single expert is ablated?

WHY THIS APPROACH (not gate weights):
  OLMoE's router is a pure feedforward gate -- it computes routing logits
  from the hidden state BEFORE any expert runs. Zeroing an expert's output
  therefore has zero effect on the gate's softmax distribution: there is no
  feedback loop from expert outputs to the router at inference time.

  The right level of analysis is the MoE LAYER OUTPUT: the weighted sum of
  all active expert contributions. When expert X is ablated, we ask:
    - How large is the output gap? (direct signal = functional importance)
    - Does the remaining top-(k-1) weighted sum shift toward filling that gap?
      (cosine similarity of the gap vector to the compensating shift vector)

  High compensation cosine similarity = other experts partially cover X's role
    -> functional redundancy with partial substitute
  Low compensation cosine similarity  = no expert covers X's role
    -> X is functionally unique / non-redundant

METRICS PER EXPERT (measured on tokens where that expert was in the top-k):
  - mean_freed_weight      : average routing weight the ablated expert held
  - output_gap_norm        : L2 norm of (baseline_output - ablated_output),
                             averaged across tokens. Equivalent to functional
                             importance (what per-token ablation's per-token delta-loss
                             measures indirectly).
  - compensation_cosine    : cosine similarity between the output gap vector
                             and the shift in the remaining experts' combined
                             output. Range [-1, 1]. High = compensating.
  - relative_compensation  : fraction of the gap norm recovered by compensation
                             = ||shift|| * cos(theta) / ||gap||.
                             Range [0, 1] if compensation is in the right
                             direction.

RESEARCH QUESTION:
  Does the gap between router confidence and expert importance correlate with
  redundancy? Here redundancy = relative_compensation: how much of the
  functional gap do remaining experts fill?

  Spearman correlation targets:
    freed_weight     vs output_gap_norm          (routing predicts importance?)
    output_gap_norm  vs relative_compensation    (important experts less redundant?)
    freed_weight     vs relative_compensation    (routing predicts redundancy?)

Usage (single layer):
  python redistribution_olmoe.py --layer 8 --device cuda
  python redistribution_olmoe.py --layer 8 --device cpu

Multi-layer driver:
  python _drivers/run_redistribution_all_layers.py --device cuda

Output:
  redistribution_v2_layer{L}_{corpus}.json
  redistribution_v2_layer{L}_{corpus}.md
"""

import argparse
import json
import math
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy import stats as scipy_stats

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
N_EXPERTS = 64
TOP_K = 8
MAX_SAMPLES = 200
MAX_LENGTH = 256
BATCH_SIZE = 8

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=8)
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--corpus", type=str, default="wikitext",
                    choices=["wikitext", "arxiv", "github"])
parser.add_argument("--samples", type=int, default=MAX_SAMPLES)
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()

LAYER = args.layer
DEVICE = args.device
CORPUS = args.corpus
N_SAMPLES = args.samples
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"Router Redistribution v2 — Functional Output Analysis")
print(f"  Layer  : {LAYER}")
print(f"  Corpus : {CORPUS}")
print(f"  Samples: {N_SAMPLES}")
print(f"  Device : {DEVICE}")
print(f"{'='*60}\n")

# ── Load model ───────────────────────────────────────────────────────────────

print("Loading model...")
dtype = torch.float16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map=DEVICE,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

moe_layer = model.model.layers[LAYER].mlp
print("Model loaded.\n")

# ── Load corpus ──────────────────────────────────────────────────────────────

def load_corpus(name, n):
    if name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 100]
    elif name == "arxiv":
        ds = load_dataset("EleutherAI/pile", "arxiv", split="test",
                          streaming=True, trust_remote_code=True)
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 100]
    elif name == "github":
        ds = load_dataset("EleutherAI/pile", "github", split="test",
                          streaming=True, trust_remote_code=True)
        texts = [r["text"] for r in ds if len(r["text"].strip()) > 100]
    return texts[:n]

print(f"Loading {CORPUS}...")
texts = load_corpus(CORPUS, N_SAMPLES)
print(f"  {len(texts)} texts loaded.")

encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                    padding=True, return_tensors="pt")
input_ids = encoded["input_ids"].to(DEVICE)
attention_mask = encoded["attention_mask"].to(DEVICE)
print(f"  Tokenized shape: {input_ids.shape}\n")

# ── Hook infrastructure ───────────────────────────────────────────────────────

class MoEOutputCapture:
    """
    Captures three things per forward pass at the target MoE layer:
      1. The full MoE layer output (weighted sum of all active experts + gate)
      2. Per-expert weighted contributions (expert_output * routing_weight)
      3. Which experts were selected (top-k indices) and their weights

    OLMoE MoE forward (simplified):
      router_logits = gate(hidden)               # (T, N_EXPERTS)
      topk_weights, topk_idx = topk(softmax(router_logits), k=8)
      output = sum_over_k(expert_k(hidden[tokens_for_k]) * weight_k)

    We hook:
      - moe.gate          → captures router logits (to get weights + indices)
      - each moe.expert_i → captures individual expert outputs before weighting
      - moe itself        → captures the final combined output
    """

    def __init__(self):
        self.moe_output = None        # (T, D) final output
        self.router_logits = None     # (T, N_EXPERTS)
        self.expert_outputs = {}      # expert_idx -> (n_tokens_for_expert, D)
        self.expert_token_indices = {}  # expert_idx -> which token positions it handled
        self._hooks = []

    def register(self, moe):
        # Hook the gate — OLMoE flattens (B,S,D)->(B*S,D) before gate,
        # so output is usually (B*S, N_EXPERTS). collect_baseline handles shape.
        def gate_hook(module, inp, out):
            self.router_logits = out.float().detach()
        self._hooks.append(moe.gate.register_forward_hook(gate_hook))

        # Hook each individual expert
        for i, expert in enumerate(moe.experts):
            def make_expert_hook(idx):
                def hook(module, inp, out):
                    # inp[0]: (n_tokens_assigned_to_this_expert, D)
                    self.expert_outputs[idx] = out.detach()
                return hook
            self._hooks.append(expert.register_forward_hook(make_expert_hook(i)))

        # Hook the MoE layer output
        def moe_output_hook(module, inp, out):
            if isinstance(out, tuple):
                self.moe_output = out[0].detach()  # (T, D)
            else:
                self.moe_output = out.detach()
        self._hooks.append(moe.register_forward_hook(moe_output_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear(self):
        self.moe_output = None
        self.router_logits = None
        self.expert_outputs = {}
        self.expert_token_indices = {}


class ExpertAblator:
    """Zeros the output of a single expert."""
    def __init__(self):
        self._hooks = []

    def ablate(self, moe, expert_idx):
        def zero_hook(module, inp, out):
            return torch.zeros_like(out)
        self._hooks.append(moe.experts[expert_idx].register_forward_hook(zero_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ── Per-expert data collection ────────────────────────────────────────────────

def run_forward_batch(model, input_ids, attention_mask, capture, start, end):
    """Single batched forward pass, results land in capture object."""
    capture.clear()
    with torch.no_grad():
        model(input_ids=input_ids[start:end],
              attention_mask=attention_mask[start:end])


def flatten_with_mask(tensor_3d, mask_2d):
    """
    tensor_3d : (batch, seq_len, hidden)  — MoE output or router logits
    mask_2d   : (batch, seq_len)          — attention mask (1=real, 0=pad)
    Returns   : (n_real_tokens, hidden)   — only real (non-padding) tokens
    """
    b, s, d = tensor_3d.shape
    flat = tensor_3d.reshape(b * s, d)           # (b*s, d)
    flat_mask = mask_2d.reshape(b * s).bool()    # (b*s,)
    return flat[flat_mask]                        # (n_real, d)


def collect_baseline(model, moe, input_ids, attention_mask):
    """
    Collect baseline MoE outputs and routing info across all batches.
    Returns flat (n_real_tokens, D) arrays — padding tokens excluded.

    Returns:
      baseline_outputs : list of (n_real_batch, D) np arrays, one per batch
      baseline_logits  : list of (n_real_batch, N_EXPERTS) np arrays
    """
    capture = MoEOutputCapture()
    capture.register(moe)

    all_outputs = []
    all_logits = []

    for start in range(0, input_ids.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, input_ids.shape[0])
        mask_batch = attention_mask[start:end]
        run_forward_batch(model, input_ids, attention_mask, capture, start, end)

        if capture.moe_output is not None:
            out = capture.moe_output.cpu().float()   # may be (B,S,D) or (B*S,D)
            if out.dim() == 3:
                out = flatten_with_mask(out, mask_batch.cpu())
            all_outputs.append(out.numpy())

        if capture.router_logits is not None:
            logits = capture.router_logits.cpu().float()  # may be (B,S,E) or (B*S,E)
            if logits.dim() == 3:
                logits = flatten_with_mask(logits, mask_batch.cpu())
            all_logits.append(logits.numpy())

    capture.remove()
    return all_outputs, all_logits


def collect_ablated(model, moe, input_ids, attention_mask, expert_idx):
    """Collect MoE outputs with expert_idx zeroed. Returns flat (n_real, D) arrays."""
    capture = MoEOutputCapture()
    capture.register(moe)
    ablator = ExpertAblator()
    ablator.ablate(moe, expert_idx)

    all_outputs = []

    for start in range(0, input_ids.shape[0], BATCH_SIZE):
        end = min(start + BATCH_SIZE, input_ids.shape[0])
        mask_batch = attention_mask[start:end]
        run_forward_batch(model, input_ids, attention_mask, capture, start, end)

        if capture.moe_output is not None:
            out = capture.moe_output.cpu().float()
            if out.dim() == 3:
                out = flatten_with_mask(out, mask_batch.cpu())
            all_outputs.append(out.numpy())

    capture.remove()
    ablator.remove()
    return all_outputs


# ── Redistribution metrics ────────────────────────────────────────────────────


def compute_expert_stats(
    base_out: np.ndarray,    # (N_tokens, D) flat
    abl_out: np.ndarray,     # (N_tokens, D) flat — same length guaranteed
    logits: np.ndarray,      # (N_tokens, N_EXPERTS) flat
    expert_idx: int,
    n_total_tokens: int,
) -> dict:
    """
    For tokens where expert_idx was in the top-k, compute:
      - gap vector        = base_out - abl_out  (expert's weighted contribution)
      - compensation      = cosine(gap, abl_out) — does remaining output point toward gap?
      - relative_comp     = ||abl_out|| * max(cos,0) / ||gap||
    All inputs are flat (N_tokens, D/E) — no batch dimension.
    """
    # Routing weights and top-k mask
    weights = torch.softmax(torch.tensor(logits), dim=-1).numpy()   # (N, 64)
    topk_indices = np.argsort(weights, axis=1)[:, -TOP_K:]           # (N, 8)
    active_mask = np.any(topk_indices == expert_idx, axis=1)         # (N,)
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    if n_active == 0:
        return {
            "mean_routing_weight": 0.0,
            "token_coverage": 0.0,
            "mean_gap_norm": 0.0,
            "std_gap_norm": 0.0,
            "compensation_cosine": 0.0,
            "relative_compensation": 0.0,
            "n_active_tokens": 0,
        }

    # Vectorised over active tokens
    gap       = base_out[active_idx] - abl_out[active_idx]   # (n_active, D)
    remaining = abl_out[active_idx]                           # (n_active, D)

    gap_norms  = np.linalg.norm(gap,       axis=1)           # (n_active,)
    rem_norms  = np.linalg.norm(remaining, axis=1)           # (n_active,)

    # Cosine similarity per token (vectorised dot product)
    dot = np.sum(gap * remaining, axis=1)                    # (n_active,)
    denom = gap_norms * rem_norms
    cos = np.where(denom > 1e-10, dot / denom, 0.0)         # (n_active,)

    rel = np.where(
        gap_norms > 1e-10,
        rem_norms * np.maximum(cos, 0.0) / gap_norms,
        0.0
    )

    rw = weights[active_idx, expert_idx]                     # (n_active,)

    return {
        "mean_routing_weight":           float(rw.mean()),
        "token_coverage":                float(n_active / n_total_tokens),
        "mean_gap_norm":                 float(gap_norms.mean()),
        "std_gap_norm":                  float(gap_norms.std()),
        "compensation_cosine":           float(cos.mean()),
        "relative_compensation":         float(rel.mean()),
        "n_active_tokens":               int(n_active),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

print("Collecting baseline outputs...")
baseline_outputs_list, baseline_logits_list = collect_baseline(
    model, moe_layer, input_ids, attention_mask)

# Concatenate into single flat arrays — this is the canonical token ordering.
# The ablation loop must produce the same token count; we enforce this below.
baseline_outputs_flat = np.concatenate(baseline_outputs_list, axis=0)   # (N_tokens, D)
baseline_logits_flat  = np.concatenate(baseline_logits_list,  axis=0)   # (N_tokens, 64)

total_tokens = baseline_outputs_flat.shape[0]
hidden_dim   = baseline_outputs_flat.shape[1]
print(f"Baseline: {total_tokens} tokens, hidden dim {hidden_dim}")
print(f"  outputs shape : {baseline_outputs_flat.shape}  (expect: (N_tokens, 2048))")
print(f"  logits shape  : {baseline_logits_flat.shape}   (expect: (N_tokens, 64))")
assert baseline_outputs_flat.ndim == 2
assert baseline_logits_flat.ndim == 2
print("  Shape checks passed.\n")

# Sanity: mean output norm
mean_output_norm = float(np.linalg.norm(baseline_outputs_flat, axis=1).mean())
print(f"Mean baseline output norm: {mean_output_norm:.4f}\n")

print(f"Running ablation loop for {N_EXPERTS} experts at layer {LAYER}...")
results = {}

for expert_idx in range(N_EXPERTS):
    ablated_list = collect_ablated(
        model, moe_layer, input_ids, attention_mask, expert_idx)
    ablated_flat = np.concatenate(ablated_list, axis=0)

    # Trim to exactly the same length as baseline (float16 rounding can
    # cause ±1 token difference in padding mask application).
    n = min(total_tokens, ablated_flat.shape[0])
    base_out = baseline_outputs_flat[:n]
    abl_out  = ablated_flat[:n]
    logits   = baseline_logits_flat[:n]

    stats = compute_expert_stats(base_out, abl_out, logits, expert_idx, n)
    results[expert_idx] = stats

    if (expert_idx + 1) % 8 == 0:
        print(f"  {expert_idx+1:2d}/64 | "
              f"coverage={stats['token_coverage']:.3f} | "
              f"gap_norm={stats['mean_gap_norm']:.4f} | "
              f"comp_cos={stats['compensation_cosine']:.4f} | "
              f"rel_comp={stats['relative_compensation']:.4f}")

print("\nDone.\n")

# ── Correlations ─────────────────────────────────────────────────────────────

routing_weights = np.array([results[i]["mean_routing_weight"] for i in range(N_EXPERTS)])
gap_norms = np.array([results[i]["mean_gap_norm"] for i in range(N_EXPERTS)])
comp_cosines = np.array([results[i]["compensation_cosine"] for i in range(N_EXPERTS)])
rel_comps = np.array([results[i]["relative_compensation"] for i in range(N_EXPERTS)])
coverages = np.array([results[i]["token_coverage"] for i in range(N_EXPERTS)])

corr_rw_gap = scipy_stats.spearmanr(routing_weights, gap_norms)
corr_gap_comp = scipy_stats.spearmanr(gap_norms, rel_comps)
corr_rw_comp = scipy_stats.spearmanr(routing_weights, rel_comps)
corr_cos_gap = scipy_stats.spearmanr(comp_cosines, gap_norms)

print("=== SPEARMAN CORRELATIONS ===")
print(f"  routing_weight vs gap_norm       : ρ={corr_rw_gap.statistic:.4f}  p={corr_rw_gap.pvalue:.4f}")
print(f"  gap_norm vs rel_compensation     : ρ={corr_gap_comp.statistic:.4f}  p={corr_gap_comp.pvalue:.4f}")
print(f"  routing_weight vs rel_compensation: ρ={corr_rw_comp.statistic:.4f}  p={corr_rw_comp.pvalue:.4f}")
print(f"  comp_cosine vs gap_norm          : ρ={corr_cos_gap.statistic:.4f}  p={corr_cos_gap.pvalue:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────

out_data = {
    "layer": LAYER,
    "corpus": CORPUS,
    "n_samples": N_SAMPLES,
    "n_tokens_total": total_tokens,
    "hidden_dim": hidden_dim,
    "correlations": {
        "routing_weight_vs_gap_norm": {
            "rho": corr_rw_gap.statistic, "p": corr_rw_gap.pvalue},
        "gap_norm_vs_relative_compensation": {
            "rho": corr_gap_comp.statistic, "p": corr_gap_comp.pvalue},
        "routing_weight_vs_relative_compensation": {
            "rho": corr_rw_comp.statistic, "p": corr_rw_comp.pvalue},
        "compensation_cosine_vs_gap_norm": {
            "rho": corr_cos_gap.statistic, "p": corr_cos_gap.pvalue},
    },
    "experts": results,
}

out_json = OUT_DIR / f"redistribution_v2_layer{LAYER}_{CORPUS}.json"
with open(out_json, "w") as f:
    json.dump(out_data, f, indent=2)
print(f"\nSaved: {out_json}")

# ── Markdown summary ──────────────────────────────────────────────────────────

sorted_by_gap = sorted(range(N_EXPERTS), key=lambda i: -gap_norms[i])
sorted_by_comp = sorted(range(N_EXPERTS), key=lambda i: -rel_comps[i])

md_lines = [
    f"# Router Redistribution v2 — Layer {LAYER} / {CORPUS}",
    f"\n**Tokens**: {total_tokens}  |  **Hidden dim**: {hidden_dim}  |  **Samples**: {N_SAMPLES}\n",
    "## Summary Statistics\n",
    "| Metric | Mean | Std | Min | Max |",
    "|--------|------|-----|-----|-----|",
    f"| Routing weight | {routing_weights.mean():.4f} | {routing_weights.std():.4f} | {routing_weights.min():.4f} | {routing_weights.max():.4f} |",
    f"| Gap norm (functional importance) | {gap_norms.mean():.4f} | {gap_norms.std():.4f} | {gap_norms.min():.4f} | {gap_norms.max():.4f} |",
    f"| Compensation cosine | {comp_cosines.mean():.4f} | {comp_cosines.std():.4f} | {comp_cosines.min():.4f} | {comp_cosines.max():.4f} |",
    f"| Relative compensation | {rel_comps.mean():.4f} | {rel_comps.std():.4f} | {rel_comps.min():.4f} | {rel_comps.max():.4f} |",
    f"| Token coverage | {coverages.mean():.4f} | {coverages.std():.4f} | {coverages.min():.4f} | {coverages.max():.4f} |",
    "",
    "## Spearman Correlations\n",
    "| Comparison | ρ | p-value | Interpretation |",
    "|------------|---|---------|----------------|",
    f"| routing_weight → gap_norm | {corr_rw_gap.statistic:.4f} | {corr_rw_gap.pvalue:.4f} | Does routing predict functional importance? |",
    f"| gap_norm → rel_compensation | {corr_gap_comp.statistic:.4f} | {corr_gap_comp.pvalue:.4f} | Do important experts have less redundancy? |",
    f"| routing_weight → rel_compensation | {corr_rw_comp.statistic:.4f} | {corr_rw_comp.pvalue:.4f} | Does routing predict redundancy? |",
    f"| comp_cosine → gap_norm | {corr_cos_gap.statistic:.4f} | {corr_cos_gap.pvalue:.4f} | Do larger gaps get more directional compensation? |",
    "",
    "## Most Functionally Important Experts (highest gap norm)\n",
    "| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |",
    "|------|--------|---------------|----------|-------------|-----------------|----------------|",
]
for rank, idx in enumerate(sorted_by_gap[:10], 1):
    r = results[idx]
    md_lines.append(
        f"| {rank} | {idx} | {r['mean_routing_weight']:.4f} | "
        f"{r['mean_gap_norm']:.4f} | {r['compensation_cosine']:.4f} | "
        f"{r['relative_compensation']:.4f} | {r['token_coverage']:.4f} |"
    )

md_lines += [
    "",
    "## Most Redundant Experts (highest relative compensation)\n",
    "| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |",
    "|------|--------|---------------|----------|-------------|-----------------|----------------|",
]
for rank, idx in enumerate(sorted_by_comp[:10], 1):
    r = results[idx]
    md_lines.append(
        f"| {rank} | {idx} | {r['mean_routing_weight']:.4f} | "
        f"{r['mean_gap_norm']:.4f} | {r['compensation_cosine']:.4f} | "
        f"{r['relative_compensation']:.4f} | {r['token_coverage']:.4f} |"
    )

out_md = OUT_DIR / f"redistribution_v2_layer{LAYER}_{CORPUS}.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"Saved: {out_md}")
print("\n✅ Complete.")
