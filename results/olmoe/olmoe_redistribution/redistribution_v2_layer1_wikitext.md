# Router Redistribution v2 — Layer 1 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0356 | 0.0059 | 0.0243 | 0.0510 |
| Gap norm (functional importance) | 0.0054 | 0.0078 | 0.0000 | 0.0506 |
| Compensation cosine | 0.0140 | 0.0104 | 0.0001 | 0.0497 |
| Relative compensation | 0.1437 | 0.2483 | 0.0011 | 1.1796 |
| Token coverage | 0.1250 | 0.0907 | 0.0296 | 0.4617 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.4551 | 0.0002 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.5870 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.0929 | 0.4653 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.7935 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 9 | 0.0401 | 0.0506 | 0.0251 | 0.0554 | 0.1845 |
| 2 | 18 | 0.0364 | 0.0419 | 0.0497 | 1.1796 | 0.2578 |
| 3 | 11 | 0.0367 | 0.0155 | 0.0406 | 0.2212 | 0.3328 |
| 4 | 5 | 0.0330 | 0.0145 | 0.0393 | 0.1377 | 0.3644 |
| 5 | 59 | 0.0340 | 0.0112 | 0.0329 | 0.1063 | 0.2294 |
| 6 | 37 | 0.0367 | 0.0090 | 0.0291 | 0.0956 | 0.1706 |
| 7 | 42 | 0.0310 | 0.0073 | 0.0183 | 0.0903 | 0.2750 |
| 8 | 40 | 0.0414 | 0.0073 | 0.0266 | 0.6705 | 0.1028 |
| 9 | 53 | 0.0420 | 0.0071 | 0.0355 | 0.9938 | 0.1920 |
| 10 | 47 | 0.0328 | 0.0067 | 0.0248 | 0.1152 | 0.4617 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 18 | 0.0364 | 0.0419 | 0.0497 | 1.1796 | 0.2578 |
| 2 | 32 | 0.0401 | 0.0053 | 0.0265 | 1.1496 | 0.1262 |
| 3 | 53 | 0.0420 | 0.0071 | 0.0355 | 0.9938 | 0.1920 |
| 4 | 39 | 0.0435 | 0.0044 | 0.0197 | 0.7793 | 0.0775 |
| 5 | 40 | 0.0414 | 0.0073 | 0.0266 | 0.6705 | 0.1028 |
| 6 | 45 | 0.0316 | 0.0052 | 0.0215 | 0.3589 | 0.1544 |
| 7 | 49 | 0.0339 | 0.0027 | 0.0097 | 0.2499 | 0.0730 |
| 8 | 11 | 0.0367 | 0.0155 | 0.0406 | 0.2212 | 0.3328 |
| 9 | 5 | 0.0330 | 0.0145 | 0.0393 | 0.1377 | 0.3644 |
| 10 | 47 | 0.0328 | 0.0067 | 0.0248 | 0.1152 | 0.4617 |