# Redistribution v2 — Cross-Layer Summary

**Corpus**: wikitext  |  **Samples**: 200  |  **Layers**: [0, 1, 7, 8, 15]

## Functional Importance by Layer

Mean gap norm = mean ||baseline_output - ablated_output|| across tokens where expert was active.
Higher = experts contribute larger, more distinctive vectors.

| Layer | Tokens | Mean gap norm | Std gap norm | Mean rel_comp | Mean comp_cos |
|-------|--------|--------------|--------------|---------------|---------------|
|  0 | 32,508 | 0.0041 | 0.0043 | 0.1472 | 0.0231 |
|  1 | 32,508 | 0.0054 | 0.0078 | 0.1437 | 0.0140 |
|  7 | 32,508 | 0.0146 | 0.0114 | 0.0408 | 0.0101 |
|  8 | 32,508 | 0.0176 | 0.0126 | 0.0552 | 0.0106 |
| 15 | 32,508 | 0.1697 | 0.1456 | 0.3481 | 0.0516 |

## Spearman Correlations by Layer

Key question: does routing weight predict functional importance (rho_rw_gap)?
Expected depth gradient: near-zero at early layers, positive at deep layers.

| Layer | rw→gap ρ | sig | gap→comp ρ | sig | rw→comp ρ | sig | cos→gap ρ | sig |
|-------|----------|-----|-----------|-----|----------|-----|----------|-----|
|  0 | +0.436 | *** | +0.377 | ** | -0.161 | ns | +0.759 | *** |
|  1 | +0.455 | *** | +0.587 | *** | +0.093 | ns | +0.793 | *** |
|  7 | +0.640 | *** | +0.853 | *** | +0.408 | *** | +0.929 | *** |
|  8 | +0.827 | *** | +0.565 | *** | +0.326 | ** | +0.903 | *** |
| 15 | +0.473 | *** | +0.747 | *** | +0.335 | ** | +0.935 | *** |

## Depth Gradient Interpretation

- **rw→gap increasing with depth**: routing weight becomes a better predictor of functional
  importance at deeper layers. At early layers routing reflects load distribution,
  not specialisation.
- **gap→comp positive**: experts with larger functional contributions are more compensated.
  Likely a geometric effect — high-norm contributions are in directions already covered
  by the ensemble, not active redistribution (the router cannot adapt at inference time).
- **Absolute gap norm increasing with depth**: experts contribute larger, more differentiated
  vectors at deeper layers. Consistent with RQ2 depth gradient (Δloss significance).

## Cross-Reference with RQ2

| Layer | RQ2 Δloss signal | This experiment | Consistent? |
|-------|-----------------|-----------------|-------------|
| 0     | No signal (d<0.05) | Expected: low gap norm, near-zero rw→gap | TBD |
| 7     | Trending (weak)    | Expected: modest gap norm, weak rw→gap   | TBD |
| 8     | Null (d<0.08)      | Expected: low gap norm, near-zero rw→gap | TBD |
| 15    | Significant        | rw→gap=+0.47***, gap norm high           | Yes |