# Router Redistribution v2 — Layer 11 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0520 | 0.0114 | 0.0297 | 0.0872 |
| Gap norm (functional importance) | 0.0487 | 0.0415 | 0.0009 | 0.1772 |
| Compensation cosine | 0.0084 | 0.0060 | 0.0002 | 0.0241 |
| Relative compensation | 0.0468 | 0.0305 | 0.0014 | 0.2012 |
| Token coverage | 0.1250 | 0.0891 | 0.0136 | 0.4158 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.6696 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.5900 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.4155 | 0.0006 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8968 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 57 | 0.0596 | 0.1772 | 0.0155 | 0.0708 | 0.3125 |
| 2 | 6 | 0.0530 | 0.1746 | 0.0173 | 0.0739 | 0.2635 |
| 3 | 55 | 0.0475 | 0.1640 | 0.0229 | 0.1006 | 0.2951 |
| 4 | 44 | 0.0574 | 0.1590 | 0.0089 | 0.0534 | 0.2239 |
| 5 | 43 | 0.0648 | 0.1169 | 0.0176 | 0.1057 | 0.2279 |
| 6 | 17 | 0.0572 | 0.1141 | 0.0241 | 0.0793 | 0.1350 |
| 7 | 58 | 0.0517 | 0.1018 | 0.0186 | 0.0508 | 0.1427 |
| 8 | 0 | 0.0680 | 0.1014 | 0.0209 | 0.0497 | 0.1100 |
| 9 | 45 | 0.0703 | 0.0890 | 0.0187 | 0.0830 | 0.1148 |
| 10 | 15 | 0.0471 | 0.0760 | 0.0127 | 0.0512 | 0.2233 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 38 | 0.0350 | 0.0089 | 0.0027 | 0.2012 | 0.0363 |
| 2 | 43 | 0.0648 | 0.1169 | 0.0176 | 0.1057 | 0.2279 |
| 3 | 55 | 0.0475 | 0.1640 | 0.0229 | 0.1006 | 0.2951 |
| 4 | 33 | 0.0641 | 0.0268 | 0.0063 | 0.0889 | 0.0704 |
| 5 | 45 | 0.0703 | 0.0890 | 0.0187 | 0.0830 | 0.1148 |
| 6 | 17 | 0.0572 | 0.1141 | 0.0241 | 0.0793 | 0.1350 |
| 7 | 7 | 0.0549 | 0.0390 | 0.0090 | 0.0788 | 0.1065 |
| 8 | 59 | 0.0455 | 0.0393 | 0.0077 | 0.0776 | 0.1598 |
| 9 | 10 | 0.0465 | 0.0096 | 0.0044 | 0.0760 | 0.0336 |
| 10 | 6 | 0.0530 | 0.1746 | 0.0173 | 0.0739 | 0.2635 |