# Router Redistribution v2 — Layer 0 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0423 | 0.0095 | 0.0255 | 0.0719 |
| Gap norm (functional importance) | 0.0041 | 0.0043 | 0.0001 | 0.0287 |
| Compensation cosine | 0.0231 | 0.0164 | 0.0008 | 0.0962 |
| Relative compensation | 0.1472 | 0.0900 | 0.0133 | 0.3959 |
| Token coverage | 0.1250 | 0.1069 | 0.0066 | 0.7583 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.4365 | 0.0003 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.3773 | 0.0021 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | -0.1612 | 0.2033 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.7592 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 41 | 0.0618 | 0.0287 | 0.0759 | 0.1134 | 0.2148 |
| 2 | 6 | 0.0392 | 0.0229 | 0.0962 | 0.3739 | 0.7583 |
| 3 | 12 | 0.0374 | 0.0082 | 0.0532 | 0.2732 | 0.1452 |
| 4 | 38 | 0.0719 | 0.0075 | 0.0502 | 0.1032 | 0.0778 |
| 5 | 19 | 0.0427 | 0.0071 | 0.0403 | 0.1907 | 0.1917 |
| 6 | 17 | 0.0420 | 0.0069 | 0.0453 | 0.2046 | 0.3596 |
| 7 | 25 | 0.0560 | 0.0065 | 0.0196 | 0.1092 | 0.1070 |
| 8 | 5 | 0.0456 | 0.0063 | 0.0249 | 0.1064 | 0.1303 |
| 9 | 26 | 0.0403 | 0.0062 | 0.0287 | 0.2044 | 0.1851 |
| 10 | 29 | 0.0530 | 0.0058 | 0.0219 | 0.1393 | 0.1889 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 57 | 0.0465 | 0.0028 | 0.0457 | 0.3959 | 0.4316 |
| 2 | 3 | 0.0386 | 0.0036 | 0.0279 | 0.3955 | 0.1276 |
| 3 | 43 | 0.0275 | 0.0009 | 0.0197 | 0.3834 | 0.0705 |
| 4 | 6 | 0.0392 | 0.0229 | 0.0962 | 0.3739 | 0.7583 |
| 5 | 7 | 0.0302 | 0.0013 | 0.0213 | 0.2920 | 0.0829 |
| 6 | 12 | 0.0374 | 0.0082 | 0.0532 | 0.2732 | 0.1452 |
| 7 | 39 | 0.0290 | 0.0009 | 0.0176 | 0.2649 | 0.0710 |
| 8 | 28 | 0.0349 | 0.0054 | 0.0319 | 0.2585 | 0.1948 |
| 9 | 46 | 0.0498 | 0.0052 | 0.0456 | 0.2396 | 0.1343 |
| 10 | 54 | 0.0532 | 0.0058 | 0.0268 | 0.2371 | 0.1473 |