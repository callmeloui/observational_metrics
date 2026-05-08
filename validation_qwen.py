"""
validation_qwen.py
===================
Cross-architecture validation on Qwen1.5-MoE-A2.7B: replicates the OLMoE
per-token ablation + per-token routing-weight control design in a single combined
experiment, to test generalisability of the OLMoE null result.

This script bundles per-token ablation and the routing-weight control because both
share model loading, tokenisation, and the per-layer loop; splitting them
would require duplicating the setup boilerplate. The OLMoE and DeepSeek
versions of these two experiments are kept as separate files because
their entry points were already separate; the asymmetry is cosmetic.

Architecture differences from OLMoE:
  - 24 layers (vs 16)
  - 64 experts per layer: 4 shared (always active) + 60 routed (top-4 selected)
  - Auxiliary load-balancing loss at alpha=0.001
    (10x weaker than OLMoE's alpha=0.01)

Design choices:
  - Ablation targets ROUTED experts only (shared experts are always active
    and not ranked by observational metrics in the pruning literature)
  - Layers tested: 0, 6, 12, 18, 23  (5 layers spanning full depth, ~same
    relative positions as OLMoE layers 0, 4, 7, 11, 15)
  - n=200 token positions per layer per metric (matches OLMoE per-token ablation experiment)
  - Metrics: utilization_rate, activation_norm, mean_routing_weight_when_active,
             activation_std  (same 4 as the OLMoE per-token ablation experiment)
  - Control: per-token routing weight (same as OLMoE control experiment)

Output:
  qwen_validation/
    per_token_ablation_results.json          -- observational metrics (4 x 5 layers)
    routing_weight_control.json  -- per-token routing weight control (5 layers)

Usage:
    python validation_qwen.py
"""

import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE      = "cuda"
DTYPE       = torch.bfloat16
SEED        = 42

# 5 layers spanning depth of 24-layer model
# Approximate relative positions matching OLMoE 0/4/7/11/15 out of 16
LAYERS      = [0, 6, 12, 18, 23]

NUM_SAMPLES = 100    # WikiText-2 test samples
MAX_LEN     = 512
NUM_TESTS   = 200    # token positions per layer per metric

TOP_K_ROUTED = 4    # Qwen routes to top-4 from routed experts
NUM_EXPERTS  = 60   # mlp.experts ModuleList length; indexed 0..59
# Architecture: 1 fused shared_expert (separate module, ~4x intermediate size,
# always active via sigmoid gate) + 60-slot routed ModuleList (gate output dim=60).
# The blog's "4 shared experts" are folded into the single shared_expert module.

METRICS     = [
    "utilization_rate",
    "activation_norm",
    "mean_routing_weight_when_active",
    "activation_std",
]

RESULTS_DIR = Path("qwen_validation")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Corpus statistics (for observational metrics) ─────────────────────────────

def compute_corpus_metrics(model, tokenizer, texts, layer_idx: int) -> dict:
    """
    Forward pass over corpus to compute per-expert observational metrics
    for routed experts at layer_idx.
    Returns dict: metric_name -> np.array of shape (num_routed_experts,)
    """
    print(f"    Computing corpus metrics for layer {layer_idx}...")

    # Identify the MoE block
    # Qwen1.5-MoE: model.model.layers[i].mlp  (QWenMoE)
    moe_block = model.model.layers[layer_idx].mlp

    # Storage
    activation_counts   = defaultdict(int)       # expert_id -> count
    activation_norms    = defaultdict(list)      # expert_id -> [norms]
    routing_weights     = defaultdict(list)      # expert_id -> [weights]
    activation_outputs  = defaultdict(list)      # expert_id -> [output tensors]

    captured = []

    def hook_fn(module, input, output):
        # QWenMoE forward returns (hidden_states,) or (hidden_states, router_logits)
        # We need router weights and expert outputs
        # Capture via the gate/router output
        if hasattr(module, 'gate'):
            # gate output: (B*S, 60) routed experts, indexed 0..59
            with torch.no_grad():
                gate_input = input[0].view(-1, input[0].shape[-1])
                logits = module.gate(gate_input)           # (B*S, 60)
                
                routed_logits = logits     # (B*S, 60)
                weights = torch.softmax(routed_logits.float(), dim=-1)
                top_w, top_idx = torch.topk(weights, TOP_K_ROUTED, dim=-1)
                
                captured.append({
                    'top_idx': top_idx.cpu(),   # (B*S, top_k), indices 0..59
                    'top_w':   top_w.cpu(),                     # (B*S, top_k)
                })

    hook = moe_block.register_forward_hook(hook_fn)

    # Run corpus
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_LEN, padding=False,
            )
            if enc["input_ids"].shape[1] < 4:
                continue
            try:
                _ = model(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                )
            except Exception:
                continue

    hook.remove()

    # Aggregate
    for batch in captured:
        top_idx = batch['top_idx'].numpy()   # (T, top_k)
        top_w   = batch['top_w'].numpy()     # (T, top_k)
        for t in range(top_idx.shape[0]):
            for k in range(TOP_K_ROUTED):
                eid = int(top_idx[t, k])
                w   = float(top_w[t, k])
                activation_counts[eid]  += 1
                routing_weights[eid].append(w)

    # Build metric arrays over routed expert indices 0..59
    routed_ids = list(range(NUM_EXPERTS))
    total_activations = max(sum(activation_counts.values()), 1)

    utilization_rate = np.array([
        activation_counts[eid] / total_activations for eid in routed_ids
    ])
    mean_routing_weight = np.array([
        np.mean(routing_weights[eid]) if routing_weights[eid] else 0.0
        for eid in routed_ids
    ])

    # For activation_norm and activation_std we need expert output norms.
    # Re-run a subset with per-expert hooks.
    print(f"    Computing activation norms (subset pass)...")
    norms_per_expert   = defaultdict(list)
    outputs_per_expert = defaultdict(list)

    def make_expert_hook(eid):
        def h(module, inp, out):
            with torch.no_grad():
                flat = out.view(-1, out.shape[-1]).float()
                for row in flat:
                    norms_per_expert[eid].append(row.norm().item())
                    # keep first 32 dims only to save memory
                    outputs_per_expert[eid].append(row[:32].cpu().numpy())
        return h

    # Hook individual routed experts
    expert_hooks = []
    for eid in routed_ids:
        # Qwen1.5-MoE: mlp.experts[0..63] are all routed experts.
        # The shared expert is a separate mlp.shared_expert module — not in this list.
        exp_module = moe_block.experts[eid]
        expert_hooks.append(exp_module.register_forward_hook(make_expert_hook(eid)))

    sample_texts = texts[:30]
    with torch.no_grad():
        for text in sample_texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_LEN, padding=False,
            )
            if enc["input_ids"].shape[1] < 4:
                continue
            try:
                _ = model(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                )
            except Exception:
                continue

    for h in expert_hooks:
        h.remove()

    activation_norm = np.array([
        np.mean(norms_per_expert[eid]) if norms_per_expert[eid] else 0.0
        for eid in routed_ids
    ])
    activation_std = np.array([
        np.std([v for vals in outputs_per_expert[eid] for v in vals]) if outputs_per_expert[eid] else 0.0
        for eid in routed_ids
    ])

    return {
        "utilization_rate":               utilization_rate,
        "activation_norm":                activation_norm,
        "mean_routing_weight_when_active": mean_routing_weight,
        "activation_std":                 activation_std,
        "routed_ids":                     routed_ids,
    }


# ── Token-level loss ───────────────────────────────────────────────────────────

def compute_token_loss(model, input_ids, attention_mask, token_pos: int) -> float:
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
        )
    logits = out.logits[0]
    if token_pos + 1 >= logits.shape[0]:
        return float('nan')
    log_probs = torch.log_softmax(logits[token_pos].float(), dim=-1)
    target_id = input_ids[0, token_pos + 1].item()
    return -log_probs[target_id].item()


# ── Expert ablation hook ───────────────────────────────────────────────────────

class ExpertAblationHook:
    """Zero the output of a single routed expert for one forward pass."""

    def __init__(self, model, layer_idx: int, expert_id: int):
        self._hooks = []
        exp = model.model.layers[layer_idx].mlp.experts[expert_id]

        def h(module, inp, out):
            return torch.zeros_like(out)

        self._hooks.append(exp.register_forward_hook(h))

    def remove(self):
        for h in self._hooks:
            h.remove()


# ── per-token ablation: observational metrics ─────────────────────────────────────────────

def run_per_token_ablation_layer(model, tokenizer, tokenized_samples,
                     layer_idx: int, corpus_metrics: dict, rng) -> dict:
    """Run per-token ablation ablation for all 4 metrics at one layer."""
    routed_ids = corpus_metrics["routed_ids"]

    layer_results = {}
    for metric in METRICS:
        metric_scores = corpus_metrics[metric]   # shape (60,) for routed experts
        # Map global expert id -> metric rank
        score_map = {eid: metric_scores[i] for i, eid in enumerate(routed_ids)}

        comparisons = []
        attempts    = 0

        while len(comparisons) < NUM_TESTS and attempts < NUM_TESTS * 10:
            attempts += 1
            si  = rng.integers(0, len(tokenized_samples))
            ids, mask = tokenized_samples[si]
            seq_len = mask[0].sum().item()
            if seq_len < 3:
                continue
            token_pos = int(rng.integers(1, seq_len - 1))

            # Get active routed experts at this position via a forward pass with hook
            captured = []

            def cap_hook(module, inp, out):
                if hasattr(module, 'gate'):
                    with torch.no_grad():
                        gate_in = inp[0].view(-1, inp[0].shape[-1])
                        logits  = module.gate(gate_in)
                        routed  = logits
                        w       = torch.softmax(routed.float(), dim=-1)
                        tw, ti  = torch.topk(w, TOP_K_ROUTED, dim=-1)
                        captured.append({
                            'idx': (ti).cpu(),
                            'w':   tw.cpu(),
                        })

            hook = model.model.layers[layer_idx].mlp.register_forward_hook(cap_hook)
            with torch.no_grad():
                _ = model(
                    input_ids=ids.to(DEVICE),
                    attention_mask=mask.to(DEVICE),
                )
            hook.remove()

            if not captured:
                continue
            tok_idx = captured[0]['idx']   # (B*S, top_k)
            tok_w   = captured[0]['w']
            if token_pos >= tok_idx.shape[0]:
                continue

            active_experts  = tok_idx[token_pos].numpy()
            # Rank by observational metric score
            scores = np.array([score_map.get(int(e), 0.0) for e in active_experts])
            order  = np.argsort(scores)[::-1]
            high_exp = int(active_experts[order[0]])
            low_exp  = int(active_experts[order[-1]])

            if high_exp == low_exp:
                continue

            baseline = compute_token_loss(model, ids, mask, token_pos)
            if np.isnan(baseline):
                continue

            h_hook  = ExpertAblationHook(model, layer_idx, high_exp)
            high_loss = compute_token_loss(model, ids, mask, token_pos)
            h_hook.remove()

            l_hook  = ExpertAblationHook(model, layer_idx, low_exp)
            low_loss  = compute_token_loss(model, ids, mask, token_pos)
            l_hook.remove()

            comparisons.append({
                "token_pos":     token_pos,
                "high_expert":   high_exp,
                "low_expert":    low_exp,
                "high_score":    float(scores[order[0]]),
                "low_score":     float(scores[order[-1]]),
                "baseline_loss": float(baseline),
                "high_delta":    float(high_loss - baseline),
                "low_delta":     float(low_loss  - baseline),
            })

        # Stats
        n = len(comparisons)
        high_d = np.array([c["high_delta"] for c in comparisons])
        low_d  = np.array([c["low_delta"]  for c in comparisons])
        diffs  = high_d - low_d
        mean_d = float(np.mean(diffs))
        std_d  = float(np.std(diffs, ddof=1))
        cohens_d = mean_d / (std_d + 1e-9)
        t_stat, t_pval = stats.ttest_rel(high_d, low_d)
        w_stat, w_pval = stats.wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
        sig = "***" if t_pval < 0.001 else ("**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else "ns"))

        layer_results[metric] = {
            "n": n, "mean_diff": mean_d, "cohens_d": float(cohens_d),
            "t_stat": float(t_stat), "t_pval": float(t_pval), "sig": sig,
            "wilcoxon_stat": float(w_stat), "wilcoxon_pval": float(w_pval),
            "comparisons": comparisons,
        }
        print(f"      {metric:<35} d={cohens_d:+.3f}  p={t_pval:.4f}  [{sig}]")

    return layer_results


# ── Routing weight control ─────────────────────────────────────────────────────

def run_rw_control_layer(model, tokenizer, tokenized_samples,
                         layer_idx: int, rng) -> dict:
    """Per-token routing weight control for one layer."""
    comparisons = []
    attempts    = 0

    while len(comparisons) < NUM_TESTS and attempts < NUM_TESTS * 10:
        attempts += 1
        si  = rng.integers(0, len(tokenized_samples))
        ids, mask = tokenized_samples[si]
        seq_len = mask[0].sum().item()
        if seq_len < 3:
            continue
        token_pos = int(rng.integers(1, seq_len - 1))

        captured = []

        def cap_hook(module, inp, out):
            if hasattr(module, 'gate'):
                with torch.no_grad():
                    gate_in = inp[0].view(-1, inp[0].shape[-1])
                    logits  = module.gate(gate_in)
                    routed  = logits
                    w       = torch.softmax(routed.float(), dim=-1)
                    tw, ti  = torch.topk(w, TOP_K_ROUTED, dim=-1)
                    captured.append({
                        'idx': (ti).cpu(),
                        'w':   tw.cpu(),
                    })

        hook = model.model.layers[layer_idx].mlp.register_forward_hook(cap_hook)
        with torch.no_grad():
            _ = model(
                input_ids=ids.to(DEVICE),
                attention_mask=mask.to(DEVICE),
            )
        hook.remove()

        if not captured:
            continue
        tok_idx = captured[0]['idx']
        tok_w   = captured[0]['w']
        if token_pos >= tok_idx.shape[0]:
            continue

        experts = tok_idx[token_pos].numpy()
        weights = tok_w[token_pos].numpy()
        order   = np.argsort(weights)[::-1]
        high_exp = int(experts[order[0]])
        low_exp  = int(experts[order[-1]])
        high_wt  = float(weights[order[0]])
        low_wt   = float(weights[order[-1]])

        if high_exp == low_exp:
            continue

        baseline = compute_token_loss(model, ids, mask, token_pos)
        if np.isnan(baseline):
            continue

        h_hook    = ExpertAblationHook(model, layer_idx, high_exp)
        high_loss = compute_token_loss(model, ids, mask, token_pos)
        h_hook.remove()

        l_hook   = ExpertAblationHook(model, layer_idx, low_exp)
        low_loss  = compute_token_loss(model, ids, mask, token_pos)
        l_hook.remove()

        comparisons.append({
            "token_pos":     token_pos,
            "high_expert":   high_exp,
            "low_expert":    low_exp,
            "high_weight":   high_wt,
            "low_weight":    low_wt,
            "weight_ratio":  high_wt / (low_wt + 1e-9),
            "baseline_loss": float(baseline),
            "high_delta":    float(high_loss - baseline),
            "low_delta":     float(low_loss  - baseline),
        })

    n      = len(comparisons)
    high_d = np.array([c["high_delta"] for c in comparisons])
    low_d  = np.array([c["low_delta"]  for c in comparisons])
    diffs  = high_d - low_d
    wrats  = np.array([c["weight_ratio"] for c in comparisons])
    mean_d = float(np.mean(diffs))
    std_d  = float(np.std(diffs, ddof=1))
    cohens_d = mean_d / (std_d + 1e-9)
    t_stat, t_pval = stats.ttest_rel(high_d, low_d)
    w_stat, w_pval = stats.wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
    rho,  rho_p   = stats.spearmanr(wrats, diffs)
    sig = "***" if t_pval < 0.001 else ("**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else "ns"))

    return {
        "n": n, "mean_diff": mean_d, "cohens_d": float(cohens_d),
        "t_stat": float(t_stat), "t_pval": float(t_pval), "sig": sig,
        "wilcoxon_stat": float(w_stat), "wilcoxon_pval": float(w_pval),
        "spearman_rho": float(rho), "spearman_pval": float(rho_p),
        "comparisons": comparisons,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("QWEN1.5-MoE-A2.7B VALIDATION EXPERIMENT")
    print(f"  Layers  : {LAYERS}")
    print(f"  Metrics : {METRICS}")
    print(f"  n/cell  : {NUM_TESTS}")
    print(f"  Device  : {DEVICE}")
    print("=" * 65)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    # ── Architecture sanity probe ─────────────────────────────────────────────
    _moe0 = model.model.layers[0].mlp
    _experts_len = len(_moe0.experts)
    _gate_out    = _moe0.gate.out_features
    _top_k       = model.config.num_experts_per_tok
    _has_shared  = hasattr(_moe0, 'shared_expert')
    print(f"\n[PROBE] mlp.experts length      : {_experts_len}")
    print(f"[PROBE] gate out_features        : {_gate_out}")
    print(f"[PROBE] num_experts_per_tok      : {_top_k}")
    print(f"[PROBE] has shared_expert module : {_has_shared}")
    assert _experts_len == NUM_EXPERTS, (
        f"NUM_EXPERTS mismatch: script={NUM_EXPERTS}, model={_experts_len}. "
        f"Fix NUM_EXPERTS at top of script."
    )
    assert _top_k == TOP_K_ROUTED, (
        f"TOP_K_ROUTED mismatch: script={TOP_K_ROUTED}, model={_top_k}."
    )
    print("[PROBE] Constants match model architecture. Proceeding.\n")
    # ─────────────────────────────────────────────────────────────────────────

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"\nLoading WikiText-2 ({NUM_SAMPLES} samples)")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts   = [t for t in dataset["text"] if len(t.strip()) > 50][:NUM_SAMPLES]

    tokenized_samples = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_LEN, padding=False,
        )
        if enc["input_ids"].shape[1] >= 4:
            tokenized_samples.append((enc["input_ids"], enc["attention_mask"]))
    print(f"  {len(tokenized_samples)} usable samples")

    per_token_ablation_all   = {}
    rw_ctrl_all  = {}

    for layer_idx in LAYERS:
        t0 = time.time()
        print(f"\n{'#'*65}")
        print(f"  LAYER {layer_idx}")
        print(f"{'#'*65}")

        # Corpus metrics
        corpus_metrics = compute_corpus_metrics(model, tokenizer, texts, layer_idx)

        # per-token ablation
        print(f"\n  [per-token ablation] Observational metric ablation")
        pb = run_per_token_ablation_layer(
            model, tokenizer, tokenized_samples, layer_idx, corpus_metrics, rng
        )
        per_token_ablation_all[layer_idx] = pb

        # Routing weight control
        print(f"\n  [Control] Per-token routing weight")
        rw = run_rw_control_layer(
            model, tokenizer, tokenized_samples, layer_idx, rng
        )
        rw_ctrl_all[layer_idx] = rw
        print(f"    routing_weight  d={rw['cohens_d']:+.3f}  p={rw['t_pval']:.4f}  [{rw['sig']}]")

        elapsed = time.time() - t0
        print(f"\n  Layer {layer_idx} done in {elapsed/60:.1f} min")

        # Save after each layer
        def _strip(d):
            return {str(k): {m: {kk: vv for kk, vv in v.items() if kk != "comparisons"}
                              for m, v in vv.items()}
                    for k, vv in d.items()} if d else {}

        with open(RESULTS_DIR / "per_token_ablation_results.json", "w", encoding="ascii", errors="replace") as f:
            json.dump(_strip(per_token_ablation_all), f, indent=2)

        rw_save = {str(k): {kk: vv for kk, vv in v.items() if kk != "comparisons"}
                   for k, v in rw_ctrl_all.items()}
        with open(RESULTS_DIR / "routing_weight_control.json", "w", encoding="ascii", errors="replace") as f:
            json.dump(rw_save, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PER-TOKEN ABLATION SUMMARY (observational metrics)")
    print(f"{'='*65}")
    print(f"{'Layer':>6}  {'Metric':<35}  {'d':>7}  {'p':>8}  {'Sig':>4}")
    print("-" * 65)
    for layer_idx in LAYERS:
        for metric in METRICS:
            r = per_token_ablation_all[layer_idx][metric]
            print(f"  {layer_idx:>4}  {metric:<35}  {r['cohens_d']:>+7.3f}  {r['t_pval']:>8.4f}  {r['sig']:>4}")

    print(f"\n{'='*65}")
    print("ROUTING WEIGHT CONTROL SUMMARY")
    print(f"{'='*65}")
    print(f"{'Layer':>6}  {'d':>7}  {'p':>8}  {'Sig':>4}  {'Spearman rho':>13}")
    print("-" * 65)
    for layer_idx in LAYERS:
        r = rw_ctrl_all[layer_idx]
        print(f"  {layer_idx:>4}  {r['cohens_d']:>+7.3f}  {r['t_pval']:>8.4f}  {r['sig']:>4}  {r['spearman_rho']:>+13.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDONE.")


if __name__ == "__main__":
    main()
