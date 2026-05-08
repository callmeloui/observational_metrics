# Router Redistribution v2 — Layer 13 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0573 | 0.0119 | 0.0317 | 0.0942 |
| Gap norm (functional importance) | 0.1008 | 0.1435 | 0.0000 | 1.0559 |
| Compensation cosine | 0.0086 | 0.0067 | 0.0000 | 0.0337 |
| Relative compensation | 0.0454 | 0.0288 | 0.0000 | 0.1603 |
| Token coverage | 0.1250 | 0.0841 | 0.0056 | 0.3748 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.5598 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.7313 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.3687 | 0.0027 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.9086 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 52 | 0.0703 | 1.0559 | 0.0322 | 0.0764 | 0.2405 |
| 2 | 63 | 0.0547 | 0.4273 | 0.0337 | 0.1603 | 0.1546 |
| 3 | 7 | 0.0942 | 0.3416 | 0.0158 | 0.0747 | 0.2344 |
| 4 | 43 | 0.0769 | 0.2993 | 0.0175 | 0.0949 | 0.2408 |
| 5 | 36 | 0.0542 | 0.1953 | 0.0224 | 0.0787 | 0.1416 |
| 6 | 56 | 0.0662 | 0.1832 | 0.0162 | 0.0931 | 0.1624 |
| 7 | 27 | 0.0884 | 0.1829 | 0.0097 | 0.0495 | 0.1726 |
| 8 | 1 | 0.0559 | 0.1575 | 0.0113 | 0.0405 | 0.1078 |
| 9 | 29 | 0.0826 | 0.1538 | 0.0085 | 0.0600 | 0.1449 |
| 10 | 41 | 0.0600 | 0.1516 | 0.0153 | 0.0518 | 0.1386 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 63 | 0.0547 | 0.4273 | 0.0337 | 0.1603 | 0.1546 |
| 2 | 46 | 0.0616 | 0.1510 | 0.0129 | 0.0959 | 0.2079 |
| 3 | 43 | 0.0769 | 0.2993 | 0.0175 | 0.0949 | 0.2408 |
| 4 | 56 | 0.0662 | 0.1832 | 0.0162 | 0.0931 | 0.1624 |
| 5 | 48 | 0.0667 | 0.1107 | 0.0152 | 0.0883 | 0.0953 |
| 6 | 34 | 0.0561 | 0.0661 | 0.0108 | 0.0862 | 0.0848 |
| 7 | 55 | 0.0528 | 0.1163 | 0.0118 | 0.0837 | 0.2481 |
| 8 | 36 | 0.0542 | 0.1953 | 0.0224 | 0.0787 | 0.1416 |
| 9 | 52 | 0.0703 | 1.0559 | 0.0322 | 0.0764 | 0.2405 |
| 10 | 32 | 0.0613 | 0.1311 | 0.0154 | 0.0763 | 0.2254 |