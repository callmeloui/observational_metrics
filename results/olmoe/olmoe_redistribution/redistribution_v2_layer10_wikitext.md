# Router Redistribution v2 — Layer 10 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0535 | 0.0122 | 0.0288 | 0.1019 |
| Gap norm (functional importance) | 0.0356 | 0.0225 | 0.0029 | 0.1170 |
| Compensation cosine | 0.0101 | 0.0060 | 0.0008 | 0.0292 |
| Relative compensation | 0.0486 | 0.0264 | 0.0094 | 0.1327 |
| Token coverage | 0.1250 | 0.0853 | 0.0216 | 0.3927 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.6308 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.6444 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.1993 | 0.1144 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8833 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 40 | 0.0496 | 0.1170 | 0.0225 | 0.1030 | 0.2571 |
| 2 | 31 | 0.0585 | 0.0964 | 0.0292 | 0.0979 | 0.1496 |
| 3 | 5 | 0.0706 | 0.0733 | 0.0111 | 0.0511 | 0.1275 |
| 4 | 62 | 0.0543 | 0.0713 | 0.0218 | 0.0932 | 0.2386 |
| 5 | 17 | 0.0522 | 0.0696 | 0.0231 | 0.0735 | 0.1371 |
| 6 | 13 | 0.0816 | 0.0683 | 0.0136 | 0.0577 | 0.1386 |
| 7 | 37 | 0.0555 | 0.0677 | 0.0184 | 0.0597 | 0.1295 |
| 8 | 54 | 0.0479 | 0.0670 | 0.0152 | 0.0582 | 0.1779 |
| 9 | 8 | 0.0572 | 0.0611 | 0.0186 | 0.0819 | 0.1228 |
| 10 | 0 | 0.0697 | 0.0562 | 0.0134 | 0.0463 | 0.1074 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 43 | 0.0397 | 0.0244 | 0.0087 | 0.1327 | 0.2359 |
| 2 | 44 | 0.0385 | 0.0256 | 0.0131 | 0.1229 | 0.2327 |
| 3 | 40 | 0.0496 | 0.1170 | 0.0225 | 0.1030 | 0.2571 |
| 4 | 31 | 0.0585 | 0.0964 | 0.0292 | 0.0979 | 0.1496 |
| 5 | 62 | 0.0543 | 0.0713 | 0.0218 | 0.0932 | 0.2386 |
| 6 | 46 | 0.0451 | 0.0142 | 0.0033 | 0.0915 | 0.1465 |
| 7 | 8 | 0.0572 | 0.0611 | 0.0186 | 0.0819 | 0.1228 |
| 8 | 9 | 0.0566 | 0.0410 | 0.0141 | 0.0762 | 0.1007 |
| 9 | 1 | 0.0491 | 0.0424 | 0.0147 | 0.0755 | 0.1018 |
| 10 | 17 | 0.0522 | 0.0696 | 0.0231 | 0.0735 | 0.1371 |