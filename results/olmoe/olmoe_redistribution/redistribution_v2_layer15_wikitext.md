# Router Redistribution v2 — Layer 15 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0571 | 0.0140 | 0.0318 | 0.1065 |
| Gap norm (functional importance) | 0.1697 | 0.1456 | 0.0014 | 0.6629 |
| Compensation cosine | 0.0516 | 0.0393 | 0.0009 | 0.1584 |
| Relative compensation | 0.3481 | 0.2462 | 0.0072 | 1.2423 |
| Token coverage | 0.1250 | 0.0948 | 0.0092 | 0.4146 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.4730 | 0.0001 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.7467 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.3347 | 0.0069 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.9346 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 7 | 0.0567 | 0.6629 | 0.0983 | 0.2613 | 0.1848 |
| 2 | 26 | 0.0572 | 0.6071 | 0.1352 | 0.8601 | 0.2508 |
| 3 | 48 | 0.0707 | 0.4598 | 0.1495 | 1.1443 | 0.2021 |
| 4 | 3 | 0.0529 | 0.4580 | 0.1584 | 1.2423 | 0.4146 |
| 5 | 49 | 0.0553 | 0.4238 | 0.1361 | 0.9600 | 0.2487 |
| 6 | 56 | 0.0639 | 0.3486 | 0.0930 | 0.3162 | 0.1198 |
| 7 | 41 | 0.0814 | 0.3463 | 0.0755 | 0.3880 | 0.1004 |
| 8 | 45 | 0.0733 | 0.3360 | 0.1072 | 0.7060 | 0.1501 |
| 9 | 2 | 0.0479 | 0.3263 | 0.1186 | 0.4083 | 0.2209 |
| 10 | 21 | 0.0577 | 0.3166 | 0.1107 | 0.6223 | 0.1466 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 3 | 0.0529 | 0.4580 | 0.1584 | 1.2423 | 0.4146 |
| 2 | 48 | 0.0707 | 0.4598 | 0.1495 | 1.1443 | 0.2021 |
| 3 | 49 | 0.0553 | 0.4238 | 0.1361 | 0.9600 | 0.2487 |
| 4 | 26 | 0.0572 | 0.6071 | 0.1352 | 0.8601 | 0.2508 |
| 5 | 17 | 0.0505 | 0.2994 | 0.1169 | 0.7394 | 0.2991 |
| 6 | 45 | 0.0733 | 0.3360 | 0.1072 | 0.7060 | 0.1501 |
| 7 | 21 | 0.0577 | 0.3166 | 0.1107 | 0.6223 | 0.1466 |
| 8 | 40 | 0.0581 | 0.2914 | 0.1014 | 0.6145 | 0.1991 |
| 9 | 23 | 0.0612 | 0.1962 | 0.0775 | 0.5753 | 0.1357 |
| 10 | 14 | 0.0526 | 0.1754 | 0.0643 | 0.5108 | 0.1063 |