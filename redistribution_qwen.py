#!/usr/bin/env python3
"""
redistribution_qwen.py
=======================
Functional redistribution experiment on Qwen1.5-MoE-A2.7B; cross-architecture
replication of redistribution_olmoe.py with metrics defined identically so
results are directly comparable.

Metrics per expert (tokens where that expert was in the active top-4):
  - mean_routing_weight    : average routing weight the ablated expert held
  - mean_gap_norm          : L2 norm of (baseline_MoE_output - ablated_MoE_output)
                             = the expert's weighted contribution to the block output
  - compensation_cosine    : cosine(gap_vector, ablated_output)
                             does the remaining output point toward the gap?
  - relative_compensation  : ||ablated_output|| * max(cos,0) / ||gap||
                             fraction of the gap norm recovered by remaining experts

Definitions match redistribution_olmoe.py exactly:
  gap       = base_out - abl_out          (at MoE block level, NOT residual stream)
  remaining = abl_out                     (output with expert zeroed)
  cos       = cosine(gap, remaining)
  rel_comp  = ||remaining|| * max(cos,0) / ||gap||

Spearman correlations (matching OLMoE script):
  routing_weight vs gap_norm
  gap_norm vs relative_compensation
  routing_weight vs relative_compensation

Architecture (confirmed by runtime probe):
  - 24 layers
  - 60 routed experts (mlp.experts[0..59]), gate out_features=60
  - 1 fused shared expert (mlp.shared_expert, always active, separate module)
  - top-4 routing, auxiliary load-balancing loss at alpha=0.001
    (10x weaker than OLMoE's alpha=0.01)

Layers tested: [0, 6, 12, 18, 23]

Output:
  qwen_validation/redistribution_results.json        (summary, all layers)
  qwen_validation/redistribution_layer{L}_full.json  (per-expert detail)

Usage:
    python redistribution_qwen.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE      = "cuda"
DTYPE       = torch.bfloat16

LAYERS      = [0, 6, 12, 18, 23]
N_EXPERTS   = 60      # mlp.experts[0..59] — confirmed by probe
TOP_K       = 4       # num_experts_per_tok
N_SAMPLES   = 200     # WikiText-2 test sequences
MAX_LENGTH  = 256     # matches OLMoE script
BATCH_SIZE  = 8

RESULTS_DIR = Path("qwen_validation")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Hook infrastructure (mirrors OLMoE script) ─────────────────────────────────

class MoEOutputCapture:
    def __init__(self):
        self.moe_output    = None
        self.router_logits = None
        self._hooks        = []

    def register(self, moe):
        def gate_hook(module, inp, out):
            self.router_logits = out.float().detach()
        self._hooks.append(moe.gate.register_forward_hook(gate_hook))

        def moe_out_hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            self.moe_output = o.detach()
        self._hooks.append(moe.register_forward_hook(moe_out_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear(self):
        self.moe_output    = None
        self.router_logits = None


class ExpertAblator:
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


# ── Data helpers ───────────────────────────────────────────────────────────────

def flatten_with_mask(tensor, mask):
    """
    tensor : (B, S, D) or (B*S, D)
    mask   : (B, S)
    Returns: (n_real_tokens, D)  — padding excluded
    """
    if tensor.dim() == 3:
        b, s, d   = tensor.shape
        flat      = tensor.reshape(b * s, d)
        flat_mask = mask.reshape(b * s).bool()
        return flat[flat_mask]
    return tensor


def collect_baseline(model, moe, input_ids, attention_mask):
    capture = MoEOutputCapture()
    capture.register(moe)
    all_outputs, all_logits = [], []

    for start in range(0, input_ids.shape[0], BATCH_SIZE):
        end        = min(start + BATCH_SIZE, input_ids.shape[0])
        mask_batch = attention_mask[start:end].cpu()
        capture.clear()
        with torch.no_grad():
            model(input_ids=input_ids[start:end],
                  attention_mask=attention_mask[start:end])

        if capture.moe_output is not None:
            out = flatten_with_mask(capture.moe_output.cpu().float(), mask_batch)
            all_outputs.append(out.numpy())

        if capture.router_logits is not None:
            logits = flatten_with_mask(capture.router_logits.cpu().float(), mask_batch)
            all_logits.append(logits.numpy())

    capture.remove()
    return all_outputs, all_logits


def collect_ablated(model, moe, input_ids, attention_mask, expert_idx):
    capture = MoEOutputCapture()
    capture.register(moe)
    ablator = ExpertAblator()
    ablator.ablate(moe, expert_idx)
    all_outputs = []

    for start in range(0, input_ids.shape[0], BATCH_SIZE):
        end        = min(start + BATCH_SIZE, input_ids.shape[0])
        mask_batch = attention_mask[start:end].cpu()
        capture.clear()
        with torch.no_grad():
            model(input_ids=input_ids[start:end],
                  attention_mask=attention_mask[start:end])

        if capture.moe_output is not None:
            out = flatten_with_mask(capture.moe_output.cpu().float(), mask_batch)
            all_outputs.append(out.numpy())

    capture.remove()
    ablator.remove()
    return all_outputs


# ── Per-expert stats (identical logic to OLMoE script) ────────────────────────

def compute_expert_stats(base_out, abl_out, logits, expert_idx, n_total_tokens):
    """
    base_out : (N_tokens, D)
    abl_out  : (N_tokens, D)
    logits   : (N_tokens, N_EXPERTS) raw gate logits

    Exactly mirrors compute_expert_stats() in redistribution_olmoe.py.
    """
    weights      = torch.softmax(torch.tensor(logits), dim=-1).numpy()  # (N, 60)
    topk_indices = np.argsort(weights, axis=1)[:, -TOP_K:]              # (N, 4)
    active_mask  = np.any(topk_indices == expert_idx, axis=1)
    active_idx   = np.where(active_mask)[0]
    n_active     = len(active_idx)

    if n_active == 0:
        return {
            "mean_routing_weight":   0.0,
            "token_coverage":        0.0,
            "mean_gap_norm":         0.0,
            "std_gap_norm":          0.0,
            "compensation_cosine":   0.0,
            "relative_compensation": 0.0,
            "n_active_tokens":       0,
        }

    gap       = base_out[active_idx] - abl_out[active_idx]  # (n_active, D)
    remaining = abl_out[active_idx]                          # (n_active, D)

    gap_norms = np.linalg.norm(gap,       axis=1)
    rem_norms = np.linalg.norm(remaining, axis=1)

    dot   = np.sum(gap * remaining, axis=1)
    denom = gap_norms * rem_norms
    # Safe division: avoid numpy RuntimeWarning by masking before dividing
    safe_denom     = np.where(denom > 1e-10, denom, 1.0)
    cos            = np.where(denom > 1e-10, dot / safe_denom, 0.0)
    safe_gap_norms = np.where(gap_norms > 1e-10, gap_norms, 1.0)
    rel            = np.where(
        gap_norms > 1e-10,
        rem_norms * np.maximum(cos, 0.0) / safe_gap_norms,
        0.0
    )

    rw = weights[active_idx, expert_idx]

    return {
        "mean_routing_weight":   float(rw.mean()),
        "token_coverage":        float(n_active / n_total_tokens),
        "mean_gap_norm":         float(gap_norms.mean()),
        "std_gap_norm":          float(gap_norms.std()),
        "compensation_cosine":   float(cos.mean()),
        "relative_compensation": float(rel.mean()),
        "n_active_tokens":       int(n_active),
    }


# ── Per-layer run ──────────────────────────────────────────────────────────────

def run_layer(model, input_ids, attention_mask, layer_idx):
    moe = model.model.layers[layer_idx].mlp
    t0  = time.time()

    print(f"  Collecting baseline...")
    baseline_outputs_list, baseline_logits_list = collect_baseline(
        model, moe, input_ids, attention_mask)

    base_out = np.concatenate(baseline_outputs_list, axis=0)
    logits   = np.concatenate(baseline_logits_list,  axis=0)
    n_tokens = base_out.shape[0]
    print(f"  {n_tokens} real tokens, hidden dim {base_out.shape[1]}")

    results = {}
    print(f"  Ablating {N_EXPERTS} experts...")
    for expert_idx in range(N_EXPERTS):
        ablated_list = collect_ablated(model, moe, input_ids, attention_mask, expert_idx)
        abl_out      = np.concatenate(ablated_list, axis=0)
        n            = min(n_tokens, abl_out.shape[0])

        stats = compute_expert_stats(
            base_out[:n], abl_out[:n], logits[:n], expert_idx, n)
        results[expert_idx] = stats

        if (expert_idx + 1) % 10 == 0:
            print(f"    {expert_idx+1:2d}/{N_EXPERTS} | "
                  f"coverage={stats['token_coverage']:.3f} | "
                  f"gap_norm={stats['mean_gap_norm']:.4f} | "
                  f"comp_cos={stats['compensation_cosine']:.4f} | "
                  f"rel_comp={stats['relative_compensation']:.4f}")

    # ── Correlations ──────────────────────────────────────────────────────────
    routing_weights = np.array([results[i]["mean_routing_weight"]   for i in range(N_EXPERTS)])
    gap_norms_arr   = np.array([results[i]["mean_gap_norm"]         for i in range(N_EXPERTS)])
    comp_cosines    = np.array([results[i]["compensation_cosine"]   for i in range(N_EXPERTS)])
    rel_comps       = np.array([results[i]["relative_compensation"] for i in range(N_EXPERTS)])

    corr_rw_gap   = scipy_stats.spearmanr(routing_weights, gap_norms_arr)
    corr_gap_comp = scipy_stats.spearmanr(gap_norms_arr,   rel_comps)
    corr_rw_comp  = scipy_stats.spearmanr(routing_weights, rel_comps)
    corr_cos_gap  = scipy_stats.spearmanr(comp_cosines,    gap_norms_arr)

    correlations = {
        "routing_weight_vs_gap_norm":              {"rho": float(corr_rw_gap.statistic),   "p": float(corr_rw_gap.pvalue)},
        "gap_norm_vs_relative_compensation":       {"rho": float(corr_gap_comp.statistic), "p": float(corr_gap_comp.pvalue)},
        "routing_weight_vs_relative_compensation": {"rho": float(corr_rw_comp.statistic),  "p": float(corr_rw_comp.pvalue)},
        "compensation_cosine_vs_gap_norm":         {"rho": float(corr_cos_gap.statistic),  "p": float(corr_cos_gap.pvalue)},
    }

    elapsed = time.time() - t0
    print(f"\n  Layer {layer_idx} correlations:")
    print(f"    rw -> gap_norm       : ρ={corr_rw_gap.statistic:+.4f}  p={corr_rw_gap.pvalue:.2e}")
    print(f"    gap_norm -> rel_comp : ρ={corr_gap_comp.statistic:+.4f}  p={corr_gap_comp.pvalue:.2e}")
    print(f"    rw -> rel_comp       : ρ={corr_rw_comp.statistic:+.4f}  p={corr_rw_comp.pvalue:.2e}")
    print(f"    cos -> gap_norm      : ρ={corr_cos_gap.statistic:+.4f}  p={corr_cos_gap.pvalue:.2e}")
    print(f"  Done in {elapsed/60:.1f} min")

    return {
        "layer":        layer_idx,
        "n_tokens":     int(n_tokens),
        "correlations": correlations,
        "experts":      results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("QWEN1.5-MoE-A2.7B  REDISTRIBUTION EXPERIMENT")
    print("(mirrors redistribution_olmoe.py exactly)")
    print(f"  Layers     : {LAYERS}")
    print(f"  N_EXPERTS  : {N_EXPERTS}  TOP_K={TOP_K}")
    print(f"  N_SAMPLES  : {N_SAMPLES}  MAX_LENGTH={MAX_LENGTH}")
    print(f"  BATCH_SIZE : {BATCH_SIZE}")
    print("=" * 65)

    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE)
    model.eval()

    # ── Architecture probe ────────────────────────────────────────────────────
    _moe0 = model.model.layers[0].mlp
    assert len(_moe0.experts) == N_EXPERTS, \
        f"N_EXPERTS mismatch: script={N_EXPERTS}, model={len(_moe0.experts)}"
    assert model.config.num_experts_per_tok == TOP_K, \
        f"TOP_K mismatch: script={TOP_K}, model={model.config.num_experts_per_tok}"
    assert hasattr(_moe0, 'shared_expert'), "shared_expert module not found"
    print(f"[PROBE] experts={len(_moe0.experts)}, top_k={model.config.num_experts_per_tok}, "
          f"shared_expert=True, gate_out={_moe0.gate.out_features}  ✓\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading WikiText-2 ({N_SAMPLES} sequences)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts   = [t for t in dataset["text"] if len(t.strip()) > 100][:N_SAMPLES]

    encoded = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH,
        padding=True, return_tensors="pt", add_special_tokens=False)
    input_ids      = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    print(f"  Tokenized: {input_ids.shape}\n")

    all_results = {}

    for layer_idx in LAYERS:
        print(f"\n{'#'*65}")
        print(f"  LAYER {layer_idx}")
        print(f"{'#'*65}")

        layer_result = run_layer(model, input_ids, attention_mask, layer_idx)
        all_results[layer_idx] = layer_result

        # Save summary after each layer
        summary = {str(k): {kk: vv for kk, vv in v.items() if kk != "experts"}
                   for k, v in all_results.items()}
        with open(RESULTS_DIR / "redistribution_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Full per-expert detail
        with open(RESULTS_DIR / f"redistribution_layer{layer_idx}_full.json", "w", encoding="utf-8") as f:
            json.dump(layer_result, f, indent=2)

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"{'Layer':>6}  {'rw->gap_norm':>13}  {'gap->rel_comp':>14}  {'rw->rel_comp':>13}")
    print("-" * 52)
    for l in LAYERS:
        c = all_results[l]["correlations"]
        print(f"  {l:>4}  "
              f"{c['routing_weight_vs_gap_norm']['rho']:>+13.4f}  "
              f"{c['gap_norm_vs_relative_compensation']['rho']:>+14.4f}  "
              f"{c['routing_weight_vs_relative_compensation']['rho']:>+13.4f}")

    print(f"\nSaved to {RESULTS_DIR}/")
    print("DONE.")


if __name__ == "__main__":
    main()
