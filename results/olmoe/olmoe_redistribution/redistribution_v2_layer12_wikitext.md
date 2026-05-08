# Router Redistribution v2 — Layer 12 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0573 | 0.0121 | 0.0308 | 0.0928 |
| Gap norm (functional importance) | 0.0664 | 0.0541 | 0.0012 | 0.3507 |
| Compensation cosine | 0.0098 | 0.0061 | 0.0003 | 0.0288 |
| Relative compensation | 0.0500 | 0.0288 | 0.0027 | 0.1430 |
| Token coverage | 0.1250 | 0.0865 | 0.0095 | 0.4365 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.3254 | 0.0087 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.6365 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.1003 | 0.4305 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8291 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 40 | 0.0675 | 0.3507 | 0.0189 | 0.1191 | 0.2437 |
| 2 | 19 | 0.0580 | 0.2231 | 0.0288 | 0.1430 | 0.4365 |
| 3 | 2 | 0.0694 | 0.1808 | 0.0205 | 0.1090 | 0.2414 |
| 4 | 11 | 0.0532 | 0.1314 | 0.0124 | 0.0535 | 0.1083 |
| 5 | 7 | 0.0815 | 0.1197 | 0.0194 | 0.0531 | 0.0949 |
| 6 | 50 | 0.0556 | 0.1176 | 0.0193 | 0.0584 | 0.1397 |
| 7 | 21 | 0.0701 | 0.1158 | 0.0233 | 0.0793 | 0.1114 |
| 8 | 52 | 0.0537 | 0.1146 | 0.0161 | 0.0535 | 0.1368 |
| 9 | 5 | 0.0534 | 0.1082 | 0.0170 | 0.0555 | 0.1137 |
| 10 | 25 | 0.0507 | 0.1067 | 0.0170 | 0.0930 | 0.1002 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 19 | 0.0580 | 0.2231 | 0.0288 | 0.1430 | 0.4365 |
| 2 | 45 | 0.0455 | 0.0847 | 0.0172 | 0.1210 | 0.1277 |
| 3 | 40 | 0.0675 | 0.3507 | 0.0189 | 0.1191 | 0.2437 |
| 4 | 2 | 0.0694 | 0.1808 | 0.0205 | 0.1090 | 0.2414 |
| 5 | 32 | 0.0509 | 0.0618 | 0.0164 | 0.1078 | 0.1257 |
| 6 | 25 | 0.0507 | 0.1067 | 0.0170 | 0.0930 | 0.1002 |
| 7 | 0 | 0.0565 | 0.0581 | 0.0126 | 0.0886 | 0.0875 |
| 8 | 3 | 0.0571 | 0.0713 | 0.0137 | 0.0823 | 0.1260 |
| 9 | 21 | 0.0701 | 0.1158 | 0.0233 | 0.0793 | 0.1114 |
| 10 | 22 | 0.0566 | 0.0871 | 0.0180 | 0.0726 | 0.0963 |