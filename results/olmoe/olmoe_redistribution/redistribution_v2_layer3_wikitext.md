# Router Redistribution v2 — Layer 3 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0426 | 0.0104 | 0.0257 | 0.0985 |
| Gap norm (functional importance) | 0.0083 | 0.0075 | 0.0004 | 0.0493 |
| Compensation cosine | 0.0103 | 0.0063 | 0.0007 | 0.0258 |
| Relative compensation | 0.0495 | 0.0272 | 0.0053 | 0.1612 |
| Token coverage | 0.1250 | 0.1035 | 0.0229 | 0.5427 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.4658 | 0.0001 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.4759 | 0.0001 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.0061 | 0.9619 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8094 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 39 | 0.0985 | 0.0493 | 0.0022 | 0.0134 | 0.0266 |
| 2 | 24 | 0.0524 | 0.0240 | 0.0195 | 0.0691 | 0.1538 |
| 3 | 25 | 0.0434 | 0.0240 | 0.0190 | 0.0663 | 0.1423 |
| 4 | 0 | 0.0514 | 0.0200 | 0.0136 | 0.0634 | 0.2129 |
| 5 | 9 | 0.0450 | 0.0196 | 0.0211 | 0.0547 | 0.2416 |
| 6 | 48 | 0.0471 | 0.0167 | 0.0180 | 0.0701 | 0.1365 |
| 7 | 52 | 0.0444 | 0.0164 | 0.0146 | 0.0559 | 0.2415 |
| 8 | 20 | 0.0479 | 0.0159 | 0.0202 | 0.0677 | 0.1492 |
| 9 | 22 | 0.0346 | 0.0147 | 0.0232 | 0.1035 | 0.0853 |
| 10 | 33 | 0.0343 | 0.0128 | 0.0183 | 0.0660 | 0.2224 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 23 | 0.0302 | 0.0107 | 0.0196 | 0.1612 | 0.5427 |
| 2 | 15 | 0.0271 | 0.0006 | 0.0052 | 0.1177 | 0.0409 |
| 3 | 22 | 0.0346 | 0.0147 | 0.0232 | 0.1035 | 0.0853 |
| 4 | 63 | 0.0461 | 0.0029 | 0.0026 | 0.0934 | 0.0439 |
| 5 | 43 | 0.0257 | 0.0007 | 0.0020 | 0.0921 | 0.0415 |
| 6 | 10 | 0.0387 | 0.0090 | 0.0112 | 0.0773 | 0.0819 |
| 7 | 30 | 0.0456 | 0.0110 | 0.0258 | 0.0758 | 0.1339 |
| 8 | 18 | 0.0398 | 0.0064 | 0.0128 | 0.0718 | 0.0788 |
| 9 | 31 | 0.0367 | 0.0106 | 0.0169 | 0.0713 | 0.1459 |
| 10 | 48 | 0.0471 | 0.0167 | 0.0180 | 0.0701 | 0.1365 |