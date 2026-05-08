# Router Redistribution v2 — Layer 4 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0383 | 0.0103 | 0.0256 | 0.1007 |
| Gap norm (functional importance) | 0.0097 | 0.0099 | 0.0005 | 0.0615 |
| Compensation cosine | 0.0106 | 0.0077 | 0.0006 | 0.0394 |
| Relative compensation | 0.2413 | 0.6625 | 0.0027 | 4.0066 |
| Token coverage | 0.1250 | 0.1065 | 0.0067 | 0.4536 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.6975 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.6181 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.1716 | 0.1751 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8800 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 14 | 0.1007 | 0.0615 | 0.0189 | 0.0290 | 0.0813 |
| 2 | 26 | 0.0383 | 0.0345 | 0.0394 | 0.1341 | 0.4149 |
| 3 | 34 | 0.0515 | 0.0325 | 0.0260 | 0.0816 | 0.2012 |
| 4 | 62 | 0.0419 | 0.0299 | 0.0288 | 0.0733 | 0.1846 |
| 5 | 37 | 0.0490 | 0.0274 | 0.0288 | 0.0905 | 0.1538 |
| 6 | 35 | 0.0422 | 0.0244 | 0.0206 | 0.0782 | 0.1342 |
| 7 | 47 | 0.0310 | 0.0191 | 0.0270 | 2.4206 | 0.4213 |
| 8 | 10 | 0.0457 | 0.0182 | 0.0207 | 0.0688 | 0.1099 |
| 9 | 49 | 0.0434 | 0.0152 | 0.0132 | 0.0644 | 0.1739 |
| 10 | 23 | 0.0384 | 0.0146 | 0.0235 | 0.0621 | 0.1286 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 27 | 0.0334 | 0.0101 | 0.0101 | 4.0066 | 0.3512 |
| 2 | 47 | 0.0310 | 0.0191 | 0.0270 | 2.4206 | 0.4213 |
| 3 | 25 | 0.0286 | 0.0029 | 0.0057 | 2.2768 | 0.2366 |
| 4 | 7 | 0.0484 | 0.0130 | 0.0143 | 1.4925 | 0.0901 |
| 5 | 63 | 0.0256 | 0.0013 | 0.0032 | 1.0458 | 0.0742 |
| 6 | 6 | 0.0324 | 0.0062 | 0.0081 | 0.9124 | 0.3176 |
| 7 | 56 | 0.0339 | 0.0071 | 0.0084 | 0.5146 | 0.0843 |
| 8 | 46 | 0.0330 | 0.0079 | 0.0070 | 0.4128 | 0.0710 |
| 9 | 26 | 0.0383 | 0.0345 | 0.0394 | 0.1341 | 0.4149 |
| 10 | 37 | 0.0490 | 0.0274 | 0.0288 | 0.0905 | 0.1538 |