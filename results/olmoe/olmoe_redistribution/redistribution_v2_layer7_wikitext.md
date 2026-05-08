# Router Redistribution v2 — Layer 7 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0426 | 0.0076 | 0.0262 | 0.0674 |
| Gap norm (functional importance) | 0.0146 | 0.0114 | 0.0003 | 0.0540 |
| Compensation cosine | 0.0101 | 0.0072 | 0.0006 | 0.0333 |
| Relative compensation | 0.0408 | 0.0248 | 0.0030 | 0.1176 |
| Token coverage | 0.1250 | 0.1123 | 0.0038 | 0.5186 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.6405 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.8525 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.4084 | 0.0008 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.9295 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 55 | 0.0413 | 0.0540 | 0.0220 | 0.0891 | 0.1279 |
| 2 | 43 | 0.0459 | 0.0518 | 0.0203 | 0.0735 | 0.1552 |
| 3 | 44 | 0.0407 | 0.0363 | 0.0250 | 0.0918 | 0.1979 |
| 4 | 31 | 0.0447 | 0.0341 | 0.0310 | 0.0900 | 0.2101 |
| 5 | 21 | 0.0674 | 0.0329 | 0.0179 | 0.0488 | 0.5186 |
| 6 | 46 | 0.0447 | 0.0313 | 0.0333 | 0.1176 | 0.1428 |
| 7 | 38 | 0.0467 | 0.0295 | 0.0191 | 0.0689 | 0.1746 |
| 8 | 5 | 0.0458 | 0.0283 | 0.0277 | 0.0974 | 0.1302 |
| 9 | 63 | 0.0562 | 0.0272 | 0.0114 | 0.0599 | 0.1689 |
| 10 | 26 | 0.0464 | 0.0253 | 0.0192 | 0.0600 | 0.1093 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 46 | 0.0447 | 0.0313 | 0.0333 | 0.1176 | 0.1428 |
| 2 | 5 | 0.0458 | 0.0283 | 0.0277 | 0.0974 | 0.1302 |
| 3 | 44 | 0.0407 | 0.0363 | 0.0250 | 0.0918 | 0.1979 |
| 4 | 31 | 0.0447 | 0.0341 | 0.0310 | 0.0900 | 0.2101 |
| 5 | 33 | 0.0393 | 0.0123 | 0.0153 | 0.0895 | 0.3812 |
| 6 | 55 | 0.0413 | 0.0540 | 0.0220 | 0.0891 | 0.1279 |
| 7 | 1 | 0.0377 | 0.0238 | 0.0161 | 0.0793 | 0.4743 |
| 8 | 43 | 0.0459 | 0.0518 | 0.0203 | 0.0735 | 0.1552 |
| 9 | 38 | 0.0467 | 0.0295 | 0.0191 | 0.0689 | 0.1746 |
| 10 | 62 | 0.0361 | 0.0193 | 0.0178 | 0.0679 | 0.0741 |