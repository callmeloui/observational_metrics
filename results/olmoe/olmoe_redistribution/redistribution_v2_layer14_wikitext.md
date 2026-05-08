# Router Redistribution v2 — Layer 14 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0572 | 0.0128 | 0.0282 | 0.0988 |
| Gap norm (functional importance) | 0.1142 | 0.0848 | 0.0000 | 0.3188 |
| Compensation cosine | 0.0171 | 0.0122 | 0.0000 | 0.0475 |
| Relative compensation | 0.1253 | 0.1208 | 0.0000 | 0.7775 |
| Token coverage | 0.1250 | 0.0941 | 0.0069 | 0.4850 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.5745 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.5302 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.2430 | 0.0530 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8133 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 3 | 0.0657 | 0.3188 | 0.0245 | 0.1281 | 0.1878 |
| 2 | 2 | 0.0589 | 0.2987 | 0.0428 | 0.1814 | 0.1936 |
| 3 | 52 | 0.0988 | 0.2877 | 0.0194 | 0.1186 | 0.1682 |
| 4 | 37 | 0.0674 | 0.2803 | 0.0331 | 0.3276 | 0.2470 |
| 5 | 49 | 0.0638 | 0.2740 | 0.0475 | 0.2440 | 0.2130 |
| 6 | 7 | 0.0461 | 0.2663 | 0.0301 | 0.0999 | 0.4850 |
| 7 | 38 | 0.0637 | 0.2457 | 0.0364 | 0.1093 | 0.1215 |
| 8 | 5 | 0.0740 | 0.2308 | 0.0364 | 0.2292 | 0.1625 |
| 9 | 21 | 0.0603 | 0.2296 | 0.0182 | 0.1087 | 0.0979 |
| 10 | 14 | 0.0542 | 0.2290 | 0.0236 | 0.1676 | 0.1210 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 22 | 0.0438 | 0.0191 | 0.0027 | 0.7775 | 0.2082 |
| 2 | 42 | 0.0510 | 0.0728 | 0.0268 | 0.4611 | 0.0812 |
| 3 | 11 | 0.0457 | 0.0724 | 0.0245 | 0.4050 | 0.1206 |
| 4 | 58 | 0.0553 | 0.0149 | 0.0048 | 0.3446 | 0.3607 |
| 5 | 37 | 0.0674 | 0.2803 | 0.0331 | 0.3276 | 0.2470 |
| 6 | 49 | 0.0638 | 0.2740 | 0.0475 | 0.2440 | 0.2130 |
| 7 | 5 | 0.0740 | 0.2308 | 0.0364 | 0.2292 | 0.1625 |
| 8 | 63 | 0.0561 | 0.0952 | 0.0369 | 0.2150 | 0.1427 |
| 9 | 13 | 0.0660 | 0.0846 | 0.0180 | 0.2009 | 0.0908 |
| 10 | 12 | 0.0600 | 0.0658 | 0.0095 | 0.1871 | 0.0870 |