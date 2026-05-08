# Router Redistribution v2 — Layer 5 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0383 | 0.0065 | 0.0252 | 0.0586 |
| Gap norm (functional importance) | 0.0107 | 0.0083 | 0.0001 | 0.0356 |
| Compensation cosine | 0.0105 | 0.0066 | 0.0002 | 0.0326 |
| Relative compensation | 0.0434 | 0.0243 | 0.0014 | 0.1143 |
| Token coverage | 0.1250 | 0.0990 | 0.0061 | 0.4981 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.7093 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.8160 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.5123 | 0.0000 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.9016 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 20 | 0.0406 | 0.0356 | 0.0236 | 0.0847 | 0.1005 |
| 2 | 44 | 0.0443 | 0.0298 | 0.0189 | 0.0750 | 0.2370 |
| 3 | 63 | 0.0380 | 0.0289 | 0.0326 | 0.1143 | 0.2748 |
| 4 | 57 | 0.0586 | 0.0282 | 0.0231 | 0.0476 | 0.1223 |
| 5 | 6 | 0.0547 | 0.0278 | 0.0213 | 0.0684 | 0.1461 |
| 6 | 40 | 0.0432 | 0.0255 | 0.0184 | 0.0943 | 0.2147 |
| 7 | 23 | 0.0396 | 0.0247 | 0.0170 | 0.0628 | 0.1855 |
| 8 | 43 | 0.0580 | 0.0232 | 0.0182 | 0.0685 | 0.1226 |
| 9 | 10 | 0.0449 | 0.0229 | 0.0192 | 0.0664 | 0.1620 |
| 10 | 37 | 0.0424 | 0.0219 | 0.0144 | 0.0585 | 0.1626 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 63 | 0.0380 | 0.0289 | 0.0326 | 0.1143 | 0.2748 |
| 2 | 1 | 0.0384 | 0.0061 | 0.0144 | 0.1022 | 0.0600 |
| 3 | 40 | 0.0432 | 0.0255 | 0.0184 | 0.0943 | 0.2147 |
| 4 | 55 | 0.0424 | 0.0090 | 0.0167 | 0.0862 | 0.4981 |
| 5 | 20 | 0.0406 | 0.0356 | 0.0236 | 0.0847 | 0.1005 |
| 6 | 44 | 0.0443 | 0.0298 | 0.0189 | 0.0750 | 0.2370 |
| 7 | 54 | 0.0446 | 0.0165 | 0.0187 | 0.0706 | 0.1081 |
| 8 | 8 | 0.0396 | 0.0146 | 0.0160 | 0.0692 | 0.1001 |
| 9 | 43 | 0.0580 | 0.0232 | 0.0182 | 0.0685 | 0.1226 |
| 10 | 6 | 0.0547 | 0.0278 | 0.0213 | 0.0684 | 0.1461 |