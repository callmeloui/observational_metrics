# Router Redistribution v2 — Layer 2 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0404 | 0.0117 | 0.0240 | 0.1141 |
| Gap norm (functional importance) | 0.0078 | 0.0159 | 0.0001 | 0.1301 |
| Compensation cosine | 0.0111 | 0.0071 | 0.0002 | 0.0363 |
| Relative compensation | 0.0721 | 0.1097 | 0.0019 | 0.8195 |
| Token coverage | 0.1250 | 0.1145 | 0.0290 | 0.5635 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.4356 | 0.0003 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.6282 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | -0.0125 | 0.9219 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8353 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 30 | 0.1141 | 0.1301 | 0.0053 | 0.0235 | 0.0433 |
| 2 | 36 | 0.0435 | 0.0279 | 0.0363 | 0.1326 | 0.3315 |
| 3 | 59 | 0.0308 | 0.0145 | 0.0249 | 0.2123 | 0.3755 |
| 4 | 58 | 0.0326 | 0.0137 | 0.0246 | 0.0849 | 0.4653 |
| 5 | 9 | 0.0450 | 0.0112 | 0.0267 | 0.1196 | 0.5635 |
| 6 | 7 | 0.0360 | 0.0100 | 0.0197 | 0.0926 | 0.3359 |
| 7 | 57 | 0.0352 | 0.0098 | 0.0220 | 0.0781 | 0.1841 |
| 8 | 34 | 0.0520 | 0.0096 | 0.0176 | 0.8195 | 0.1180 |
| 9 | 14 | 0.0442 | 0.0090 | 0.0154 | 0.0468 | 0.1067 |
| 10 | 26 | 0.0627 | 0.0089 | 0.0219 | 0.0602 | 0.0847 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 34 | 0.0520 | 0.0096 | 0.0176 | 0.8195 | 0.1180 |
| 2 | 60 | 0.0379 | 0.0059 | 0.0169 | 0.4030 | 0.1239 |
| 3 | 59 | 0.0308 | 0.0145 | 0.0249 | 0.2123 | 0.3755 |
| 4 | 45 | 0.0363 | 0.0055 | 0.0146 | 0.1532 | 0.1485 |
| 5 | 36 | 0.0435 | 0.0279 | 0.0363 | 0.1326 | 0.3315 |
| 6 | 9 | 0.0450 | 0.0112 | 0.0267 | 0.1196 | 0.5635 |
| 7 | 10 | 0.0338 | 0.0071 | 0.0138 | 0.1067 | 0.4180 |
| 8 | 54 | 0.0325 | 0.0030 | 0.0088 | 0.0974 | 0.0742 |
| 9 | 22 | 0.0454 | 0.0075 | 0.0225 | 0.0958 | 0.0982 |
| 10 | 7 | 0.0360 | 0.0100 | 0.0197 | 0.0926 | 0.3359 |