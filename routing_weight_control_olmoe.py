"""
routing_weight_control_olmoe.py
================================
Per-token routing-weight control experiment on OLMoE-1B-7B-0924, run
across all five layers tested in the paper (0, 4, 7, 11, 15) at n=200 tokens per layer.

Design mirrors per-token ablation exactly:
  - For each token position, identify the highest- and lowest-routing-weight
    active expert (per-token, not global population rank)
  - Ablate each in turn; measure delta cross-entropy at that position
  - Paired t-test + Wilcoxon + Cohen's d + Spearman rho(weight_ratio, delta_diff)

Key distinction from the four observational metrics: routing weight here is the
per-token gate value the router assigns to each of the top-8 experts -- it
is conditioned on the specific hidden state, not aggregated over the corpus.
This is the strongest possible per-token predictor and serves as the
"ceiling" against which the null results for global metrics should be read.

Output: olmoe_routing_weight_control_layers/
  - routing_weight_control_all_layers.json       (summary)
  - routing_weight_control_all_layers_full.json  (full per-token data)

Usage:
    python routing_weight_control_olmoe.py
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME   = "allenai/OLMoE-1B-7B-0924"
DEVICE       = "cuda"
DTYPE        = torch.float32
SEED         = 42

LAYERS       = [0, 4, 7, 11, 15]   # must match per-token ablation
NUM_SAMPLES  = 100                  # WikiText-2 test samples
MAX_LEN      = 512
NUM_TESTS    = 200                  # token positions per layer (matches the per-token ablation experiment)
TOP_K        = 8                    # OLMoE top-k routing

RESULTS_DIR  = Path("olmoe_routing_weight_control_layers")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Router hook ────────────────────────────────────────────────────────────────

class RouterCaptureHook:
    """Captures top-k expert indices and weights for a single MoE layer."""

    def __init__(self, model, layer_idx: int):
        self.layer_idx = layer_idx
        self.expert_indices = None   # (seq, top_k)
        self.expert_weights = None   # (seq, top_k)
        self._hook = None
        self._register(model)

    def _register(self, model):
        # OLMoE: model.model.layers[i].mlp is OlmoeMoE
        # The router is model.model.layers[i].mlp.gate
        moe_block = model.model.layers[self.layer_idx].mlp

        def hook_fn(module, input, output):
            # output from OlmoeMoE forward: (hidden_states, router_logits)
            # router_logits shape: (batch*seq, num_experts)
            if isinstance(output, tuple) and len(output) == 2:
                router_logits = output[1]          # (B*S, E)
                routing_weights = torch.softmax(router_logits.float(), dim=-1)
                top_w, top_idx = torch.topk(routing_weights, TOP_K, dim=-1)
                self.expert_indices = top_idx.detach().cpu()   # (B*S, top_k)
                self.expert_weights = top_w.detach().cpu()     # (B*S, top_k)

        self._hook = moe_block.register_forward_hook(hook_fn)

    def remove(self):
        if self._hook:
            self._hook.remove()


class ExpertAblationHook:
    """Zero-gates a single expert for one forward pass."""

    def __init__(self, model, layer_idx: int, expert_id: int):
        self.layer_idx  = layer_idx
        self.expert_id  = expert_id
        self._hooks     = []
        self._register(model)

    def _register(self, model):
        moe_block = model.model.layers[self.layer_idx].mlp

        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                hidden, logits = output
                # Zero the router logit for the target expert across all tokens
                logits_mod = logits.clone()
                logits_mod[:, self.expert_id] = float('-inf')
                return (hidden, logits_mod)
            return output

        # We need to hook deeper: intercept after router but before expert dispatch.
        # Simplest stable approach: hook each individual expert's forward and zero its output.
        expert = moe_block.experts[self.expert_id]

        def expert_hook(module, input, output):
            return torch.zeros_like(output)

        self._hooks.append(expert.register_forward_hook(expert_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()

# ── Loss computation ───────────────────────────────────────────────────────────

def compute_token_loss(model, input_ids, attention_mask, token_pos: int) -> float:
    """Cross-entropy loss at a single token position (0-indexed in sequence)."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
        )
    logits = out.logits[0]              # (seq, vocab)
    # Loss at position token_pos: predict token at token_pos+1
    if token_pos + 1 >= logits.shape[0]:
        return float('nan')
    log_probs = torch.log_softmax(logits[token_pos].float(), dim=-1)
    target_id = input_ids[0, token_pos + 1].item()
    return -log_probs[target_id].item()


# ── Per-layer experiment ───────────────────────────────────────────────────────

def run_layer(model, tokenizer, tokenized_samples, layer_idx: int, rng) -> dict:
    print(f"\n{'='*65}")
    print(f"  LAYER {layer_idx}  (n={NUM_TESTS} tokens)")
    print(f"{'='*65}")

    capture_hook = RouterCaptureHook(model, layer_idx)

    comparisons = []
    attempts    = 0
    max_attempts = NUM_TESTS * 10

    sample_pool = list(range(len(tokenized_samples)))

    while len(comparisons) < NUM_TESTS and attempts < max_attempts:
        attempts += 1
        # Pick a random sample
        si = rng.integers(0, len(sample_pool))
        ids, mask = tokenized_samples[si]
        seq_len = mask[0].sum().item()

        if seq_len < 3:
            continue

        # Randomly pick a valid token position (not padding, not last)
        valid_positions = list(range(1, seq_len - 1))
        if not valid_positions:
            continue
        token_pos = int(rng.choice(valid_positions))

        # Forward pass to capture routing weights at this token
        capture_hook.expert_indices = None
        capture_hook.expert_weights = None

        with torch.no_grad():
            _ = model(
                input_ids=ids.to(DEVICE),
                attention_mask=mask.to(DEVICE),
            )

        if capture_hook.expert_indices is None:
            continue

        # Flatten (batch=1) -> get routing for this token position
        flat_idx  = capture_hook.expert_indices   # (B*S, top_k)
        flat_wts  = capture_hook.expert_weights   # (B*S, top_k)

        # Offset: batch_size=1 so flat index = token_pos
        if token_pos >= flat_idx.shape[0]:
            continue

        tok_experts = flat_idx[token_pos].numpy()    # (top_k,)
        tok_weights = flat_wts[token_pos].numpy()    # (top_k,)

        if len(tok_experts) < 2:
            continue

        # Identify high- and low-weight active expert
        order        = np.argsort(tok_weights)[::-1]   # descending
        high_exp_idx = tok_experts[order[0]]
        low_exp_idx  = tok_experts[order[-1]]
        high_wt      = tok_weights[order[0]]
        low_wt       = tok_weights[order[-1]]
        weight_ratio = high_wt / (low_wt + 1e-9)

        # Baseline loss at this token
        baseline_loss = compute_token_loss(model, ids, mask, token_pos)
        if np.isnan(baseline_loss):
            continue

        # Ablate high-weight expert
        h_hook = ExpertAblationHook(model, layer_idx, int(high_exp_idx))
        high_loss = compute_token_loss(model, ids, mask, token_pos)
        h_hook.remove()

        # Ablate low-weight expert
        l_hook = ExpertAblationHook(model, layer_idx, int(low_exp_idx))
        low_loss = compute_token_loss(model, ids, mask, token_pos)
        l_hook.remove()

        high_delta = high_loss - baseline_loss
        low_delta  = low_loss  - baseline_loss

        comparisons.append({
            "token_pos":     token_pos,
            "high_expert":   int(high_exp_idx),
            "low_expert":    int(low_exp_idx),
            "high_weight":   float(high_wt),
            "low_weight":    float(low_wt),
            "weight_ratio":  float(weight_ratio),
            "baseline_loss": float(baseline_loss),
            "high_loss":     float(high_loss),
            "low_loss":      float(low_loss),
            "high_delta":    float(high_delta),
            "low_delta":     float(low_delta),
        })

        if len(comparisons) % 20 == 0:
            print(f"    {len(comparisons)}/{NUM_TESTS} tokens done", flush=True)

    capture_hook.remove()

    # ── Statistics ──────────────────────────────────────────────────────────
    n = len(comparisons)
    if n < 10:
        print(f"  WARNING: only {n} comparisons collected for layer {layer_idx}")

    high_deltas = np.array([c["high_delta"] for c in comparisons])
    low_deltas  = np.array([c["low_delta"]  for c in comparisons])
    diffs       = high_deltas - low_deltas
    weight_ratios = np.array([c["weight_ratio"] for c in comparisons])

    mean_diff   = float(np.mean(diffs))
    std_diff    = float(np.std(diffs, ddof=1))
    cohens_d    = mean_diff / (std_diff + 1e-9)

    t_stat, t_pval   = stats.ttest_rel(high_deltas, low_deltas)
    w_stat, w_pval   = stats.wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')
    rho, rho_p       = stats.spearmanr(weight_ratios, diffs)

    sig = "***" if t_pval < 0.001 else ("**" if t_pval < 0.01 else ("*" if t_pval < 0.05 else "ns"))

    result = {
        "layer":          layer_idx,
        "n":              n,
        "mean_diff":      mean_diff,
        "std_diff":       std_diff,
        "cohens_d":       float(cohens_d),
        "t_stat":         float(t_stat),
        "t_pval":         float(t_pval),
        "sig":            sig,
        "wilcoxon_stat":  float(w_stat),
        "wilcoxon_pval":  float(w_pval),
        "spearman_rho":   float(rho),
        "spearman_pval":  float(rho_p),
        "comparisons":    comparisons,
    }

    print(f"\n  Layer {layer_idx} results:")
    print(f"    n             = {n}")
    print(f"    Mean diff     = {mean_diff:+.4f}")
    print(f"    Cohen's d     = {cohens_d:+.3f}")
    print(f"    Paired t      : t={t_stat:.3f}, p={t_pval:.4f}  [{sig}]")
    print(f"    Wilcoxon      : W={w_stat:.1f}, p={w_pval:.4f}")
    print(f"    Spearman rho  : rho={rho:+.3f}, p={rho_p:.4f}")

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ROUTING WEIGHT CONTROL -- ALL FIVE TESTED LAYERS")
    print(f"  Layers : {LAYERS}")
    print(f"  n/layer: {NUM_TESTS}")
    print(f"  Device : {DEVICE}")
    print("=" * 65)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    # Load model once
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Load data once
    print(f"\nLoading WikiText-2 ({NUM_SAMPLES} samples)")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts   = [t for t in dataset["text"] if len(t.strip()) > 50][:NUM_SAMPLES]

    tokenized_samples = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        if enc["input_ids"].shape[1] >= 4:
            tokenized_samples.append(
                (enc["input_ids"], enc["attention_mask"])
            )

    print(f"  {len(tokenized_samples)} usable samples")

    # Run all layers
    all_results = {}
    for layer_idx in LAYERS:
        t0 = time.time()
        result = run_layer(model, tokenizer, tokenized_samples, layer_idx, rng)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = elapsed
        all_results[layer_idx] = result

        # Save after each layer (resume-safe)
        out_path = RESULTS_DIR / "routing_weight_control_all_layers.json"
        with open(out_path, "w", encoding="ascii", errors="replace") as f:
            json.dump(
                {str(k): {kk: vv for kk, vv in v.items() if kk != "comparisons"}
                 for k, v in all_results.items()},
                f, indent=2
            )
        print(f"\n  Saved summary -> {out_path}")

    # Final summary table
    print(f"\n{'='*65}")
    print("SUMMARY TABLE  (routing weight as high/low selector)")
    print(f"{'='*65}")
    print(f"{'Layer':>6}  {'n':>4}  {'MeanDiff':>9}  {'Cohen d':>8}  {'p(t)':>8}  {'Sig':>4}  {'Spearman rho':>13}")
    print("-" * 65)
    for layer_idx in LAYERS:
        r = all_results[layer_idx]
        print(
            f"  {r['layer']:>4}  {r['n']:>4}  {r['mean_diff']:>+9.4f}  "
            f"{r['cohens_d']:>+8.3f}  {r['t_pval']:>8.4f}  {r['sig']:>4}  "
            f"{r['spearman_rho']:>+13.3f}"
        )

    # Full JSON with comparisons
    full_path = RESULTS_DIR / "routing_weight_control_all_layers_full.json"
    with open(full_path, "w", encoding="ascii", errors="replace") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nFull results (with per-token comparisons) -> {full_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()
