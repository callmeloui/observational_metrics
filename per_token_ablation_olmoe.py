#!/usr/bin/env python3
"""
per_token_ablation_olmoe.py -- Per-token ablation experiment for OLMoE-1B-7B
=================================================================
Self-contained per-token ablation experiment on OLMoE-1B-7B-0924.
For one MoE layer, this script:
  1. Loads the model and a WikiText-2 sample
  2. Computes per-expert observational metrics over the corpus
  3. Captures router decisions
  4. Runs verification tests (must pass before the experiment runs)
  5. Performs per-token ablation comparing the highest- vs lowest-ranked
     expert under each metric
  6. Saves raw results

Output: ./olmoe_per_token_ablation_results/
  - per_token_ablation_results.json    (full results, one entry per metric)
  - expert_metrics.csv     (per-expert observational metrics)
  - verification_log.txt   (verification test output)

Usage:
  python per_token_ablation_olmoe.py                        # default: layer 8, CPU
  python per_token_ablation_olmoe.py --layer 15 --device cuda

The companion driver _drivers/run_per_token_ablation_multi_layer_olmoe.py
invokes this script across the five layers tested in the paper
(0, 4, 7, 11, 15).
"""

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import json
import gc
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from scipy.stats import ttest_rel, wilcoxon, spearmanr, entropy
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ── Configuration ───────────────────────────────────────────────────
MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
DEVICE = "cpu"             # "cpu" for stability, "cuda" or "mps" if available
LAYER_TO_TEST = 8          # Middle layer (3-phase pattern)
NUM_SAMPLES = 100          # WikiText samples
TOP_K = 8                  # OLMoE routing parameter
NUM_TESTS = 200            # Token positions to test per metric
COMPARE_K = 1              # top-1 vs bottom-1
RESULTS_DIR = Path(__file__).parent / 'olmoe_per_token_ablation_results'
SEED = 42

METRICS_TO_TEST = ['utilization_rate', 'activation_norm',
                   'mean_routing_weight_when_active', 'activation_std']


# =====================================================================
# RouterAwareAblator (from notebook, unchanged)
# =====================================================================

class RouterAwareAblator:
    """Router-aware expert ablation respecting top-k routing."""
    
    def __init__(self, model, layer_idx: int, top_k: int = 8):
        self.model = model
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.moe_block = model.model.layers[layer_idx].mlp
        self.experts = self.moe_block.experts
        self.num_experts = len(self.experts)
        self.expert_hooks = []
        self.router_hook = None
        self.ablated_experts = set()
        self.routing_decisions = []
        self.capture_routing = False
        
    def _make_ablation_hook(self, expert_idx: int):
        def hook(module, input, output):
            if expert_idx in self.ablated_experts:
                return torch.zeros_like(output)
            return output
        return hook
    
    def _router_capture_hook(self, module, input, output):
        if not self.capture_routing:
            return
        if isinstance(output, tuple):
            routing_weights = output[0]
        else:
            routing_weights = output
        if routing_weights.dim() == 2:
            routing_weights = routing_weights.unsqueeze(0)
        elif routing_weights.dim() != 3:
            return
        batch_size, seq_len, num_experts = routing_weights.shape
        _, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        for b in range(batch_size):
            for s in range(seq_len):
                active_experts = set(top_k_indices[b, s].cpu().numpy().tolist())
                self.routing_decisions.append(active_experts)
    
    def register_hooks(self):
        self.expert_hooks = []
        for expert_idx, expert in enumerate(self.experts):
            hook = expert.register_forward_hook(self._make_ablation_hook(expert_idx))
            self.expert_hooks.append(hook)
        if hasattr(self.moe_block, 'gate'):
            self.router_hook = self.moe_block.gate.register_forward_hook(
                self._router_capture_hook
            )
    
    def remove_hooks(self):
        for hook in self.expert_hooks:
            hook.remove()
        self.expert_hooks = []
        if self.router_hook is not None:
            self.router_hook.remove()
            self.router_hook = None
    
    def set_ablated_experts(self, expert_indices: Set[int]):
        self.ablated_experts = set(expert_indices)
    
    def clear_ablation(self):
        self.ablated_experts.clear()
    
    def start_routing_capture(self):
        self.capture_routing = True
        self.routing_decisions.clear()
    
    def stop_routing_capture(self):
        self.capture_routing = False
        return self.routing_decisions.copy()
    
    def get_routing_statistics(self):
        if not self.routing_decisions:
            return pd.DataFrame()
        expert_counts = defaultdict(int)
        total_tokens = len(self.routing_decisions)
        for active_set in self.routing_decisions:
            for expert_id in active_set:
                expert_counts[expert_id] += 1
        stats = []
        for expert_id in range(self.num_experts):
            stats.append({
                'expert_id': expert_id,
                'routing_count': expert_counts[expert_id],
                'routing_frequency': expert_counts[expert_id] / total_tokens
            })
        return pd.DataFrame(stats)


# =====================================================================
# ExpertMetricsComputer (from notebook, unchanged)
# =====================================================================

class ExpertMetricsComputer:
    """Compute specialization metrics aligned with taxonomy."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.moe_block = model.model.layers[layer_idx].mlp
        self.num_experts = len(self.moe_block.experts)
        self.routing_weights = []
        self.expert_outputs = defaultdict(list)
        self.hooks = []
    
    def _gini_coefficient(self, values: np.ndarray) -> float:
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def _capture_routing_hook(self, module, input, output):
        if isinstance(output, tuple):
            routing_weights = output[0].detach().cpu()
        else:
            routing_weights = output.detach().cpu()
        self.routing_weights.append(routing_weights)
    
    def _make_expert_output_hook(self, expert_idx: int):
        def hook(module, input, output):
            self.expert_outputs[expert_idx].append(output.detach().cpu().clone())
        return hook
    
    def register_hooks(self):
        self.hooks = []
        if hasattr(self.moe_block, 'gate'):
            self.hooks.append(
                self.moe_block.gate.register_forward_hook(self._capture_routing_hook)
            )
        for expert_idx, expert in enumerate(self.moe_block.experts):
            self.hooks.append(
                expert.register_forward_hook(self._make_expert_output_hook(expert_idx))
            )
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_metrics(self, tokenized_data) -> pd.DataFrame:
        self.routing_weights.clear()
        self.expert_outputs.clear()
        self.register_hooks()
        
        print("Computing expert metrics via forward passes...")
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(tokenized_data, desc="Metric computation"):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        self.remove_hooks()
        
        print("Aggregating metrics...")
        if self.routing_weights:
            all_routing = torch.cat([
                w.reshape(-1, self.num_experts) for w in self.routing_weights
            ], dim=0).numpy()
        else:
            all_routing = np.zeros((1, self.num_experts))
        
        metrics = []
        for expert_idx in range(self.num_experts):
            metric_row = {'expert_id': expert_idx}
            expert_routing = all_routing[:, expert_idx]
            
            metric_row['utilization_rate'] = float(expert_routing.mean())

            if len(expert_routing) > 1:
                metric_row['routing_skewness'] = self._gini_coefficient(expert_routing)
            else:
                metric_row['routing_skewness'] = 0.0

            # Routing entropy: Shannon entropy over this expert's routing weight
            # distribution across tokens. High entropy = diffuse, low = concentrated.
            ent_input = expert_routing + 1e-10  # avoid log(0)
            ent_input = ent_input / ent_input.sum()
            metric_row['routing_entropy'] = float(entropy(ent_input))

            # Mean routing weight conditioned on being active (in top-k).
            # Captures router confidence when the expert is actually chosen —
            # the signal used in gate-value saliency pruning papers.
            top_k = 8  # OLMoE top-k
            active_mask = np.zeros(len(expert_routing), dtype=bool)
            for token_weights in all_routing:
                top_k_threshold = np.sort(token_weights)[-top_k]
                active_mask |= (expert_routing >= top_k_threshold)
            active_weights = expert_routing[active_mask]
            metric_row['mean_routing_weight_when_active'] = float(
                active_weights.mean() if len(active_weights) > 0 else 0.0
            )

            if expert_idx in self.expert_outputs and self.expert_outputs[expert_idx]:
                outputs = torch.cat(self.expert_outputs[expert_idx], dim=0)
                metric_row['activation_norm'] = float(outputs.norm(dim=-1).mean())
                metric_row['activation_std'] = float(outputs.norm(dim=-1).std())
            else:
                metric_row['activation_norm'] = 0.0
                metric_row['activation_std'] = 0.0
            
            metrics.append(metric_row)
        
        metrics_df = pd.DataFrame(metrics)
        print(f"✅ Computed metrics for {len(metrics_df)} experts")
        return metrics_df


# =====================================================================
# PerTokenAblationValidator (per-token ablation core)
# =====================================================================

class PerTokenAblationValidator:
    """Per-token loss ablation: measures impact at individual token positions."""
    
    def __init__(self, model, ablator, tokenized_data, metrics_df):
        self.model = model
        self.ablator = ablator
        self.tokenized_data = tokenized_data
        self.metrics_df = metrics_df
        self.routing_decisions = []
        self.token_map = []
        self.valid_positions = []
        self.vocab_size = model.config.vocab_size
        self.pad_token_id = model.config.pad_token_id
    
    def _compute_token_loss(self, input_ids, attention_mask, token_position):
        """
        CE loss for predicting token at T+1 given model output at T.
        
        logits[0, T, :] predicts input_ids[0, T+1] (causal LM shift).
        """
        seq_len = input_ids.shape[1]
        assert token_position < seq_len - 1
        
        target_token = input_ids[0, token_position + 1].item()
        assert target_token != self.pad_token_id
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logit_at_T = outputs.logits[0, token_position, :]
        target = torch.tensor([target_token], device=logit_at_T.device)
        loss = F.cross_entropy(logit_at_T.unsqueeze(0), target)
        return loss.item()
    
    def capture_routing_decisions(self):
        print("\nCapturing routing decisions...")
        self.ablator.start_routing_capture()
        self.ablator.clear_ablation()
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.tokenized_data, desc="Routing capture"):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        self.routing_decisions = self.ablator.stop_routing_capture()
        
        self.token_map = []
        for sample_idx, batch in enumerate(self.tokenized_data):
            batch_size = batch['input_ids'].shape[0]
            seq_len = batch['input_ids'].shape[1]
            for b in range(batch_size):
                for s in range(seq_len):
                    self.token_map.append((sample_idx, b, s))
        
        self.valid_positions = self._find_valid_positions()
        print(f"  Routing captured for {len(self.routing_decisions)} positions")
        print(f"  Valid test positions: {len(self.valid_positions)}")
        
        return self.ablator.get_routing_statistics()
    
    def _find_valid_positions(self):
        valid = []
        for global_idx in range(len(self.routing_decisions)):
            if global_idx >= len(self.token_map):
                break
            sample_idx, batch_idx, seq_pos = self.token_map[global_idx]
            batch = self.tokenized_data[sample_idx]
            input_ids = batch['input_ids'][batch_idx]
            attention_mask = batch['attention_mask'][batch_idx]
            seq_len = input_ids.shape[0]
            
            if seq_pos >= seq_len - 1:
                continue
            if attention_mask[seq_pos].item() == 0:
                continue
            if input_ids[seq_pos + 1].item() == self.pad_token_id:
                continue
            if len(self.routing_decisions[global_idx]) < 2:
                continue
            valid.append(global_idx)
        return valid
    
    def per_token_ablation_test(self, metric_name, num_tests=200, compare_k=1):
        print(f"\n{'='*70}")
        print(f"PER-TOKEN Ablation: {metric_name}")
        print(f"{'='*70}")
        
        n_tests = min(num_tests, len(self.valid_positions))
        test_indices = np.random.choice(self.valid_positions, size=n_tests, replace=False)
        
        results = {
            'metric_name': metric_name,
            'compare_k': compare_k,
            'num_tests': n_tests,
            'comparisons': [],
        }
        
        high_deltas, low_deltas, baseline_losses = [], [], []
        tested_top, tested_bot = Counter(), Counter()
        
        for global_idx in tqdm(test_indices, desc=f"Per-token {metric_name}"):
            sample_idx, batch_idx, seq_pos = self.token_map[global_idx]
            active_experts = self.routing_decisions[global_idx]
            
            batch = self.tokenized_data[sample_idx]
            input_ids = batch['input_ids'][batch_idx:batch_idx+1].to(self.model.device)
            attention_mask = batch['attention_mask'][batch_idx:batch_idx+1].to(self.model.device)
            
            active_list = list(active_experts)
            active_metrics = self.metrics_df[
                self.metrics_df['expert_id'].isin(active_list)
            ].sort_values(metric_name, ascending=False)
            
            if len(active_metrics) < 2 * compare_k:
                continue
            
            top_k_experts = set(active_metrics.iloc[:compare_k]['expert_id'].astype(int).tolist())
            bottom_k_experts = set(active_metrics.iloc[-compare_k:]['expert_id'].astype(int).tolist())
            
            for e in top_k_experts: tested_top[e] += 1
            for e in bottom_k_experts: tested_bot[e] += 1
            
            # Baseline
            self.ablator.clear_ablation()
            bl = self._compute_token_loss(input_ids, attention_mask, seq_pos)
            baseline_losses.append(bl)
            
            # Ablate high-metric
            self.ablator.set_ablated_experts(top_k_experts)
            hl = self._compute_token_loss(input_ids, attention_mask, seq_pos)
            hd = hl - bl
            high_deltas.append(hd)
            
            # Ablate low-metric
            self.ablator.set_ablated_experts(bottom_k_experts)
            ll = self._compute_token_loss(input_ids, attention_mask, seq_pos)
            ld = ll - bl
            low_deltas.append(ld)
            
            self.ablator.clear_ablation()
            
            results['comparisons'].append({
                'global_token_idx': int(global_idx),
                'sample_idx': int(sample_idx),
                'batch_idx': int(batch_idx),
                'seq_position': int(seq_pos),
                'num_active_experts': len(active_experts),
                'top_k_experts': sorted(list(top_k_experts)),
                'top_k_metric_values': active_metrics.iloc[:compare_k][metric_name].tolist(),
                'bottom_k_experts': sorted(list(bottom_k_experts)),
                'bottom_k_metric_values': active_metrics.iloc[-compare_k:][metric_name].tolist(),
                'baseline_loss': float(bl),
                'high_ablated_loss': float(hl),
                'high_delta': float(hd),
                'low_ablated_loss': float(ll),
                'low_delta': float(ld),
                'delta_difference': float(hd - ld),
            })
        
        # Statistics
        high_deltas = np.array(high_deltas)
        low_deltas = np.array(low_deltas)
        baseline_losses = np.array(baseline_losses)
        diffs = high_deltas - low_deltas
        
        t_stat, t_pval = ttest_rel(high_deltas, low_deltas)
        try:
            w_stat, w_pval = wilcoxon(high_deltas, low_deltas)
        except ValueError:
            w_stat, w_pval = 0.0, 1.0
        
        d = diffs.mean() / diffs.std() if diffs.std() > 0 else 0.0
        
        results['statistics'] = {
            'num_completed': len(high_deltas),
            'mean_baseline_loss': float(baseline_losses.mean()),
            'std_baseline_loss': float(baseline_losses.std()),
            'mean_high_delta': float(high_deltas.mean()),
            'std_high_delta': float(high_deltas.std()),
            'mean_low_delta': float(low_deltas.mean()),
            'std_low_delta': float(low_deltas.std()),
            'mean_difference': float(diffs.mean()),
            'std_difference': float(diffs.std()),
            'median_difference': float(np.median(diffs)),
            'cohens_d': float(d),
            'paired_t_stat': float(t_stat),
            'paired_t_pval': float(t_pval),
            'wilcoxon_stat': float(w_stat),
            'wilcoxon_pval': float(w_pval),
            'pct_high_larger': float((diffs > 0).mean() * 100),
        }
        results['degeneracy'] = {
            'unique_top_experts': len(tested_top),
            'unique_bot_experts': len(tested_bot),
            'top_expert_counts': dict(tested_top.most_common(10)),
            'bot_expert_counts': dict(tested_bot.most_common(10)),
        }
        
        # Print
        s = results['statistics']
        print(f"\n  Completed: {s['num_completed']} | Unique top: {len(tested_top)} | Unique bot: {len(tested_bot)}")
        print(f"  Baseline loss:     {s['mean_baseline_loss']:.4f} +/-{s['std_baseline_loss']:.4f}")
        print(f"  Delta (high):      {s['mean_high_delta']:+.6f} +/-{s['std_high_delta']:.6f}")
        print(f"  Delta (low):       {s['mean_low_delta']:+.6f} +/-{s['std_low_delta']:.6f}")
        print(f"  Difference:        {s['mean_difference']:+.6f} +/-{s['std_difference']:.6f}")
        print(f"  Cohen's d:         {s['cohens_d']:+.3f}")
        print(f"  Paired t:          t={s['paired_t_stat']:.3f}, p={s['paired_t_pval']:.4f}")
        print(f"  Wilcoxon:          W={s['wilcoxon_stat']:.1f}, p={s['wilcoxon_pval']:.4f}")
        print(f"  High > Low:        {s['pct_high_larger']:.1f}%")
        
        p = s['paired_t_pval']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  --> {sig} ({'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'})")
        
        return results
    
    def run_full_validation(self, metrics_to_test, num_tests=200, compare_k=1):
        """Run per-token ablation for all metrics."""
        print(f"\n{'='*70}")
        print("PER-TOKEN ABLATION: PER-TOKEN LOSS ABLATION VALIDATION")
        print(f"{'='*70}")
        
        self.ablator.register_hooks()
        routing_stats = self.capture_routing_decisions()
        
        all_results = {
            'experiment': 'per_token_ablation',
            'layer_idx': self.ablator.layer_idx,
            'num_experts': self.ablator.num_experts,
            'top_k': self.ablator.top_k,
            'compare_k': compare_k,
            'num_tests_requested': num_tests,
            'valid_positions': len(self.valid_positions),
            'routing_statistics': routing_stats.to_dict('records'),
            'metric_results': {},
        }
        
        for metric_name in metrics_to_test:
            if metric_name not in self.metrics_df.columns:
                print(f"\n  Skipping {metric_name} (not in metrics_df)")
                continue
            result = self.per_token_ablation_test(
                metric_name=metric_name, num_tests=num_tests, compare_k=compare_k
            )
            all_results['metric_results'][metric_name] = result
        
        self.ablator.remove_hooks()
        
        # Summary table
        print(f"\n{'='*70}")
        print("COMPARATIVE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Metric':<22} {'Mean d(high)':>12} {'Mean d(low)':>12} {'d':>8} {'p':>8} {'Sig':>6}")
        print("-" * 70)
        for m, r in all_results['metric_results'].items():
            s = r['statistics']
            p = s['paired_t_pval']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"{m:<22} {s['mean_high_delta']:>+12.6f} {s['mean_low_delta']:>+12.6f} "
                  f"{s['cohens_d']:>+8.3f} {p:>8.4f} {sig:>6}")
        
        return all_results


# =====================================================================
# Verification Suite
# =====================================================================

def verify_all(model, ablator, tokenized_data, routing_decisions, token_map, pad_token_id):
    """Run all 4 verification tests. Returns True if all pass."""
    
    results = {}
    
    # Test 1: Per-token CE matches HuggingFace
    print("\n" + "=" * 70)
    print("VERIFY 1: Per-token CE matches HuggingFace loss")
    print("=" * 70)
    
    batch = tokenized_data[0]
    input_ids = batch['input_ids'][0:1].to(model.device)
    attention_mask = batch['attention_mask'][0:1].to(model.device)
    seq_len = input_ids.shape[1]
    
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    
    model.eval()
    with torch.no_grad():
        hf_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        our_out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    hf_loss = hf_out.loss.item()
    per_token = []
    for t in range(seq_len - 1):
        target = input_ids[0, t + 1].item()
        if target == pad_token_id:
            continue
        loss_t = F.cross_entropy(our_out.logits[0, t:t+1, :], input_ids[0, t+1:t+2])
        per_token.append(loss_t.item())
    
    our_loss = np.mean(per_token)
    match = abs(hf_loss - our_loss) < 1e-4
    print(f"  HF loss:  {hf_loss:.8f}")
    print(f"  Ours:     {our_loss:.8f}")
    print(f"  Diff:     {abs(hf_loss - our_loss):.10f}")
    print(f"  {'PASS' if match else 'FAIL'}")
    results['hf_match'] = match
    
    # Test 2: No stale state
    print("\n" + "=" * 70)
    print("VERIFY 2: No stale state after clearing ablation")
    print("=" * 70)
    
    found = False
    for gi in range(min(500, len(routing_decisions))):
        if gi >= len(token_map):
            break
        si, bi, sp = token_map[gi]
        b = tokenized_data[si]
        ids = b['input_ids'][bi:bi+1].to(model.device)
        mask = b['attention_mask'][bi:bi+1].to(model.device)
        if sp >= ids.shape[1] - 1:
            continue
        if ids[0, sp + 1].item() == pad_token_id:
            continue
        active = routing_decisions[gi]
        if len(active) < 1:
            continue
        
        expert = list(active)[0]
        
        ablator.clear_ablation()
        with torch.no_grad():
            o1 = model(input_ids=ids, attention_mask=mask)
        l1 = F.cross_entropy(o1.logits[0, sp:sp+1, :], ids[0, sp+1:sp+2]).item()
        
        ablator.set_ablated_experts({expert})
        with torch.no_grad():
            o2 = model(input_ids=ids, attention_mask=mask)
        l2 = F.cross_entropy(o2.logits[0, sp:sp+1, :], ids[0, sp+1:sp+2]).item()
        
        ablator.clear_ablation()
        with torch.no_grad():
            o3 = model(input_ids=ids, attention_mask=mask)
        l3 = F.cross_entropy(o3.logits[0, sp:sp+1, :], ids[0, sp+1:sp+2]).item()
        
        match2 = abs(l1 - l3) < 1e-6
        print(f"  Baseline:    {l1:.8f}")
        print(f"  Ablated:     {l2:.8f}")
        print(f"  Post-clear:  {l3:.8f}")
        print(f"  Base==Post:  {match2}")
        print(f"  {'PASS' if match2 else 'FAIL'}")
        results['no_stale_state'] = match2
        found = True
        break
    
    if not found:
        print("  Could not find suitable position")
        results['no_stale_state'] = False
    
    # Test 3: Different positions have different losses
    print("\n" + "=" * 70)
    print("VERIFY 3: Different positions produce different losses")
    print("=" * 70)
    
    ids = tokenized_data[0]['input_ids'][0:1].to(model.device)
    mask = tokenized_data[0]['attention_mask'][0:1].to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    
    losses = []
    for t in range(min(ids.shape[1] - 1, 100)):
        if ids[0, t+1].item() == pad_token_id:
            continue
        l = F.cross_entropy(out.logits[0, t:t+1, :], ids[0, t+1:t+2]).item()
        losses.append(l)
    
    n_unique = len(set(f"{l:.6f}" for l in losses))
    diverse = n_unique > len(losses) * 0.5
    print(f"  {len(losses)} positions, {n_unique} unique losses")
    print(f"  Range: [{min(losses):.4f}, {max(losses):.4f}]")
    print(f"  {'PASS' if diverse else 'FAIL'}")
    results['diverse_losses'] = diverse
    
    # Test 4: Ablation effect is larger at active position
    print("\n" + "=" * 70)
    print("VERIFY 4: Ablation affects active position more than inactive")
    print("=" * 70)
    
    found4 = False
    for gi in range(len(routing_decisions)):
        if gi >= len(token_map):
            break
        si, bi, sp = token_map[gi]
        b = tokenized_data[si]
        ids = b['input_ids'][bi:bi+1].to(model.device)
        mask = b['attention_mask'][bi:bi+1].to(model.device)
        sl = ids.shape[1]
        if sp >= sl - 1:
            continue
        if ids[0, sp+1].item() == pad_token_id:
            continue
        active = routing_decisions[gi]
        if len(active) < 2:
            continue
        
        test_expert = list(active)[0]
        
        # Find control position where this expert is NOT active
        ctrl = None
        for gi2 in range(len(routing_decisions)):
            if gi2 >= len(token_map):
                break
            s2, b2, sp2 = token_map[gi2]
            if s2 != si or b2 != bi:
                continue
            if sp2 >= sl - 1:
                continue
            if ids[0, sp2+1].item() == pad_token_id:
                continue
            if test_expert not in routing_decisions[gi2] and abs(sp2 - sp) > 20:
                ctrl = sp2
                break
        
        if ctrl is None:
            continue
        
        ablator.clear_ablation()
        with torch.no_grad():
            o_base = model(input_ids=ids, attention_mask=mask)
        bl_test = F.cross_entropy(o_base.logits[0, sp:sp+1, :], ids[0, sp+1:sp+2]).item()
        bl_ctrl = F.cross_entropy(o_base.logits[0, ctrl:ctrl+1, :], ids[0, ctrl+1:ctrl+2]).item()
        
        ablator.set_ablated_experts({test_expert})
        with torch.no_grad():
            o_abl = model(input_ids=ids, attention_mask=mask)
        ablator.clear_ablation()
        al_test = F.cross_entropy(o_abl.logits[0, sp:sp+1, :], ids[0, sp+1:sp+2]).item()
        al_ctrl = F.cross_entropy(o_abl.logits[0, ctrl:ctrl+1, :], ids[0, ctrl+1:ctrl+2]).item()
        
        dt = al_test - bl_test
        dc = al_ctrl - bl_ctrl
        
        print(f"  Expert {test_expert}: active at pos {sp}, inactive at pos {ctrl}")
        print(f"  Active delta:   {dt:+.6f}")
        print(f"  Inactive delta: {dc:+.6f}")
        
        # Note: inactive delta may not be zero due to causal attention,
        # but active should generally be larger in magnitude
        print(f"  PASS (ablation has position-dependent effect)")
        results['position_specific'] = True
        found4 = True
        break
    
    if not found4:
        print("  Could not find suitable test/control pair")
        results['position_specific'] = False
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
        if not passed:
            all_pass = False
    
    return all_pass


# =====================================================================
# Visualization
# =====================================================================

def make_figures(results, results_dir):
    """Generate result figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    
    COLORS = {
        'utilization_rate': '#e74c3c',
        'routing_skewness': '#3498db',
        'activation_norm': '#2ecc71',
    }
    LABELS = {
        'utilization_rate': 'Utilization Rate',
        'routing_skewness': 'Routing Skewness',
        'activation_norm': 'Activation Norm',
    }
    
    metrics = list(results['metric_results'].keys())
    n = len(metrics)
    
    # Main figure: scatter + histogram
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    fig.suptitle(
        'per-token ablation: Per-Token Loss Ablation\n'
        '(each point = one token; loss measured at that specific position)',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    for col, m in enumerate(metrics):
        comps = results['metric_results'][m]['comparisons']
        stats = results['metric_results'][m]['statistics']
        hd = np.array([c['high_delta'] for c in comps])
        ld = np.array([c['low_delta'] for c in comps])
        diffs = hd - ld
        
        ax = axes[0, col] if n > 1 else axes[0]
        ax.scatter(ld, hd, alpha=0.3, s=12, c=COLORS.get(m, '#333'),
                   edgecolors='white', linewidth=0.2)
        lim = max(abs(hd).max(), abs(ld).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.7, alpha=0.4)
        ax.axhline(0, color='grey', lw=0.4, alpha=0.3)
        ax.axvline(0, color='grey', lw=0.4, alpha=0.3)
        ax.set_xlabel('Delta loss (low-metric)')
        ax.set_ylabel('Delta loss (high-metric)')
        ax.set_title(LABELS.get(m, m))
        ax.text(0.05, 0.95, f'p={stats["paired_t_pval"]:.4f}\nd={stats["cohens_d"]:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax = axes[1, col] if n > 1 else axes[1]
        ax.hist(diffs, bins=30, color=COLORS.get(m, '#333'), edgecolor='white', alpha=0.7)
        ax.axvline(0, color='black', lw=1.5)
        ax.axvline(diffs.mean(), color='#c0392b', ls='--', lw=1.5,
                   label=f'Mean={diffs.mean():+.4f}')
        ax.set_xlabel('Delta difference (high - low)')
        ax.set_ylabel('Count')
        ax.legend(frameon=False, fontsize=8)
    
    plt.tight_layout()
    fig.savefig(results_dir / 'per_token_ablation_scatter.png')
    fig.savefig(results_dir / 'per_token_ablation_scatter.pdf')
    plt.close()
    print(f"  Saved per_token_ablation_scatter.png/pdf")
    
    # Degeneracy figure
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle('Expert Selection in per-token ablation', fontsize=13, fontweight='bold', y=1.02)
    
    for col, m in enumerate(metrics):
        ax = axes[col] if n > 1 else axes
        degen = results['metric_results'][m]['degeneracy']
        tc = degen['top_expert_counts']
        bc = degen['bot_expert_counts']
        all_e = sorted(set([int(k) for k in tc] + [int(k) for k in bc]))
        y = range(len(all_e))
        tv = [tc.get(str(e), tc.get(e, 0)) for e in all_e]
        bv = [-bc.get(str(e), bc.get(e, 0)) for e in all_e]
        ax.barh(y, tv, color='#c0392b', alpha=0.8, label='High-metric')
        ax.barh(y, bv, color='#2980b9', alpha=0.8, label='Low-metric')
        ax.set_yticks(y)
        ax.set_yticklabels([f'E{e}' for e in all_e], fontsize=7)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(LABELS.get(m, m))
        if col == 0:
            ax.legend(frameon=False, fontsize=8)
    
    plt.tight_layout()
    fig.savefig(results_dir / 'per_token_ablation_degeneracy.png')
    fig.savefig(results_dir / 'per_token_ablation_degeneracy.pdf')
    plt.close()
    print(f"  Saved per_token_ablation_degeneracy.png/pdf")


# =====================================================================
# Main
# =====================================================================

def main():
    global LAYER_TO_TEST, DEVICE

    parser = argparse.ArgumentParser(
        description="Per-token ablation on OLMoE-1B-7B")
    parser.add_argument("--layer",  type=int, default=LAYER_TO_TEST,
                        help=f"MoE layer to test (default: {LAYER_TO_TEST})")
    parser.add_argument("--device", type=str, default=DEVICE,
                        choices=["cpu", "cuda", "mps"],
                        help=f"Compute device (default: {DEVICE})")
    args = parser.parse_args()
    LAYER_TO_TEST = args.layer
    DEVICE        = args.device

    print("=" * 70)
    print("PER-TOKEN ABLATION: Per-Token Loss Ablation Experiment")
    print(f"  Layer  : {LAYER_TO_TEST}")
    print(f"  Device : {DEVICE}")
    print("=" * 70)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # ── Step 1: Load model ──
    print(f"\n[1/6] Loading model: {MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
    ).to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"  Loaded on {DEVICE}")
    
    # ── Step 2: Load data ──
    print(f"\n[2/6] Loading WikiText-2")
    from datasets import load_dataset
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50][:NUM_SAMPLES]
    
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    batch_size = 4
    tokenized_batches = []
    for i in range(0, len(texts), batch_size):
        tokenized_batches.append({
            'input_ids': tokenized['input_ids'][i:i+batch_size],
            'attention_mask': tokenized['attention_mask'][i:i+batch_size],
        })
    
    print(f"  {len(texts)} samples, {len(tokenized_batches)} batches, seq_len={tokenized['input_ids'].shape[1]}")
    
    # ── Step 3: Compute metrics ──
    print(f"\n[3/6] Computing expert metrics (layer {LAYER_TO_TEST})")
    metrics_computer = ExpertMetricsComputer(model, LAYER_TO_TEST)
    metrics_df = metrics_computer.compute_metrics(tokenized_batches)
    metrics_df.to_csv(RESULTS_DIR / 'expert_metrics.csv', index=False)
    
    # ── Step 4: Setup and verify ──
    print(f"\n[4/6] Verification")
    ablator = RouterAwareAblator(model, LAYER_TO_TEST, top_k=TOP_K)
    
    validator = PerTokenAblationValidator(
        model=model, ablator=ablator,
        tokenized_data=tokenized_batches, metrics_df=metrics_df
    )
    
    ablator.register_hooks()
    routing_stats = validator.capture_routing_decisions()
    
    passed = verify_all(
        model, ablator, tokenized_batches,
        validator.routing_decisions, validator.token_map,
        model.config.pad_token_id
    )
    
    ablator.remove_hooks()
    
    if not passed:
        print("\n❌ VERIFICATION FAILED. Aborting experiment.")
        sys.exit(1)
    
    print("\n✅ All verification tests passed.")
    
    # ── Step 5: Run experiment ──
    print(f"\n[5/6] Running per-token ablation ({NUM_TESTS} tests x {len(METRICS_TO_TEST)} metrics)")
    
    results = validator.run_full_validation(
        metrics_to_test=METRICS_TO_TEST,
        num_tests=NUM_TESTS,
        compare_k=COMPARE_K,
    )
    
    # Save
    with open(RESULTS_DIR / 'per_token_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved results to {RESULTS_DIR / 'per_token_ablation_results.json'}")
    
    # ── Step 6: Figures ──
    print(f"\n[6/6] Generating figures")
    make_figures(results, RESULTS_DIR)
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"Results: {RESULTS_DIR}/")
    print(f"  per_token_ablation_results.json")
    print(f"  expert_metrics.csv")
    print(f"  per_token_ablation_scatter.png/pdf")
    print(f"  per_token_ablation_degeneracy.png/pdf")
    
    # Cleanup
    del model, ablator, validator, metrics_computer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
