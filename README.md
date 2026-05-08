# Mechanistic Audit of MoE Pruning Metrics — Code

Code accompanying the workshop paper. Anonymized for double-blind review.

This repository contains the experiment scripts used to produce the empirical
results in the paper. It does not contain figure-generation scripts, model
checkpoints, or precomputed results — see the paper for findings, and run
the scripts here to reproduce the underlying JSON outputs.

## Models

All models are loaded from the Hugging Face hub at runtime; no checkpoints
are bundled here.

| Identifier in code             | Hugging Face id                  |
|--------------------------------|----------------------------------|
| OLMoE (primary model)          | `allenai/OLMoE-1B-7B-0924`       |
| Qwen1.5-MoE (cross-arch)       | `Qwen/Qwen1.5-MoE-A2.7B`         |
| DeepSeek-V2-Lite (cross-arch)  | `deepseek-ai/DeepSeek-V2-Lite`   |

Loading the latter two requires `trust_remote_code=True`.

## Layout

Worker scripts live at the top level. Multi-layer drivers — small wrappers
that invoke a worker once per layer and consolidate outputs — live under
`_drivers/`. The drivers are subprocess shells; the workers are the science.

```
.
├── per_token_ablation_olmoe.py        # per-token causal ablation, single layer
├── per_token_ablation_deepseek.py     # per-token causal ablation, single layer
│
├── routing_weight_control_olmoe.py    # routing-weight control, all 5 layers
├── routing_weight_control_deepseek.py # routing-weight control, all 5 layers
│
├── redistribution_olmoe.py            # functional redistribution, single layer
├── redistribution_qwen.py             # functional redistribution, all 5 layers
│
├── progressive_ablation_olmoe.py      # progressive ablation across layers
│
├── validation_qwen.py                 # per-token ablation + routing-weight control combined
│
└── _drivers/
    ├── run_per_token_ablation_multi_layer_olmoe.py
    ├── run_per_token_ablation_multi_layer_deepseek.py
    └── run_redistribution_all_layers.py
```

`validation_qwen.py` bundles Qwen per-token ablation and the Qwen routing-weight
control into a single script because both share model loading, tokenisation, and
the per-layer loop. The OLMoE and DeepSeek versions of these two experiments are
kept as separate files because their entry points were already separate; the
asymmetry is cosmetic.

## Mapping experiments to scripts

Paper section in parentheses.

| Experiment                                    | OLMoE                                  | Qwen1.5-MoE                            | DeepSeek-V2-Lite                       |
|-----------------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| Per-token causal ablation (§6.1, §7.1, §7.4)  | `per_token_ablation_olmoe.py`          | `validation_qwen.py` (ablation half)   | `per_token_ablation_deepseek.py`       |
| Routing-weight control (§6.2, §7.2, §7.4)     | `routing_weight_control_olmoe.py`      | `validation_qwen.py` (control half)    | `routing_weight_control_deepseek.py`   |
| Redistribution analysis (§6.3, §7.3, §7.4)    | `redistribution_olmoe.py`              | `redistribution_qwen.py`               | not run                                |
| Progressive ablation (§6.4, §7.5)             | `progressive_ablation_olmoe.py`        | not run                                | not run                                |

Layers tested in the paper: OLMoE [0, 4, 7, 11, 15] of 16 layers; Qwen
[0, 6, 12, 18, 23] of 24 layers; DeepSeek [1, 7, 13, 20, 26] of 27 layers
(layer 0 is dense in DeepSeek). n=200 token positions per layer per metric
throughout, except progressive ablation which uses n=500.

Corpus: WikiText-2 raw, test split, 100 samples (200 for progressive ablation),
truncated to 512 tokens, consistent across all experiments and all models.

## Running

Each worker is self-contained:

```bash
# Per-token ablation on OLMoE, single layer (CPU-friendly)
python per_token_ablation_olmoe.py --layer 8 --device cpu

# Per-token ablation on DeepSeek, on a CUDA GPU
python per_token_ablation_deepseek.py --layer 13 --device cuda

# All five per-token ablation layers on OLMoE via the driver
python _drivers/run_per_token_ablation_multi_layer_olmoe.py --device cuda

# Routing-weight control across all five layers (one script call)
python routing_weight_control_olmoe.py
python routing_weight_control_deepseek.py

# Qwen per-token ablation + routing-weight control in one go
python validation_qwen.py

# Redistribution analysis
python redistribution_olmoe.py --layer 8 --device cuda
python _drivers/run_redistribution_all_layers.py --device cuda
python redistribution_qwen.py

# Progressive ablation
python progressive_ablation_olmoe.py
```

Drivers are invoked from the repo root so the relative paths to the workers
resolve correctly.

## Verification

The paper's Appendix B describes a four-test verification procedure that runs
before any data collection (per-token cross-entropy matches Hugging Face
reference loss; no stale state after clearing ablation hooks; position
diversity; position-specific ablation effect). The verification logic is
embedded in each per-token ablation worker (`verify_all` / runtime probes)
and aborts data collection if any test fails. There is no separate
`verify.py`.

## Outputs

Each script writes a results directory next to itself; subsequent invocations
overwrite the contents. Filenames inside follow the pattern documented at the
top of each script. The format throughout is JSON; per-token detail is saved
to a separate `_full.json` file where applicable.

## Reproducibility notes

- All scripts use a fixed random seed (42) and deterministic NumPy/PyTorch
  RNGs.
- We do **not** force CUDA-level determinism
  (`torch.use_deterministic_algorithms(True)`,
  `torch.backends.cudnn.deterministic = True`). Re-running on a different GPU,
  CUDA version, or PyTorch build can produce small numerical differences in
  loss values, though the qualitative conclusions
  (sign of effect, Bonferroni significance, depth gradient) are stable.
- All experiments in the paper were run on a single NVIDIA RTX 3090. OLMoE
  is loaded in fp32, Qwen1.5-MoE in bfloat16 (its native release precision),
  DeepSeek-V2-Lite in fp16. Per-token ablation on OLMoE will also run on CPU
  in fp32 (MacBook M3 Pro: roughly 30-60 minutes per layer); DeepSeek and
  Qwen require a CUDA GPU with at least 24 GB of VRAM.

## Dependencies

See `requirements.txt`. Pinned versions match those used to produce the
reported results; later versions of `transformers` may work but have not been
tested. `trust_remote_code=True` is required for Qwen1.5-MoE and
DeepSeek-V2-Lite.

## Repository scope

This repository contains the experiments cited in the workshop paper. Earlier
exploratory scripts (linguistic routing analysis, pairwise expert interaction
studies) are part of the larger thesis project and are not included here to
keep the review surface minimal.
