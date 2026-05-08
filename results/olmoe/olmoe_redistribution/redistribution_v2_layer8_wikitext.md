# Router Redistribution v2 — Layer 8 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0436 | 0.0103 | 0.0293 | 0.1071 |
| Gap norm (functional importance) | 0.0176 | 0.0126 | 0.0002 | 0.0592 |
| Compensation cosine | 0.0106 | 0.0066 | 0.0001 | 0.0270 |
| Relative compensation | 0.0552 | 0.0724 | 0.0030 | 0.5928 |
| Token coverage | 0.1250 | 0.1160 | 0.0030 | 0.4678 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.8271 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.5650 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.3263 | 0.0085 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.9034 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 56 | 0.0501 | 0.0592 | 0.0172 | 0.0606 | 0.1318 |
| 2 | 37 | 0.0476 | 0.0477 | 0.0221 | 0.0743 | 0.3564 |
| 3 | 12 | 0.0510 | 0.0393 | 0.0220 | 0.0767 | 0.1332 |
| 4 | 45 | 0.0471 | 0.0390 | 0.0169 | 0.0657 | 0.4188 |
| 5 | 18 | 0.0444 | 0.0386 | 0.0270 | 0.1155 | 0.3964 |
| 6 | 26 | 0.0506 | 0.0364 | 0.0203 | 0.0665 | 0.1933 |
| 7 | 15 | 0.0475 | 0.0360 | 0.0220 | 0.0817 | 0.1892 |
| 8 | 55 | 0.0529 | 0.0327 | 0.0155 | 0.0552 | 0.1185 |
| 9 | 63 | 0.0484 | 0.0316 | 0.0242 | 0.0822 | 0.1187 |
| 10 | 34 | 0.0459 | 0.0315 | 0.0186 | 0.0644 | 0.1111 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 33 | 0.0383 | 0.0130 | 0.0116 | 0.5928 | 0.1992 |
| 2 | 18 | 0.0444 | 0.0386 | 0.0270 | 0.1155 | 0.3964 |
| 3 | 60 | 0.0317 | 0.0019 | 0.0016 | 0.1057 | 0.0271 |
| 4 | 7 | 0.0305 | 0.0011 | 0.0021 | 0.1025 | 0.0132 |
| 5 | 5 | 0.0391 | 0.0289 | 0.0203 | 0.1005 | 0.2014 |
| 6 | 59 | 0.0338 | 0.0017 | 0.0021 | 0.0884 | 0.0310 |
| 7 | 52 | 0.0488 | 0.0253 | 0.0180 | 0.0844 | 0.1146 |
| 8 | 63 | 0.0484 | 0.0316 | 0.0242 | 0.0822 | 0.1187 |
| 9 | 15 | 0.0475 | 0.0360 | 0.0220 | 0.0817 | 0.1892 |
| 10 | 12 | 0.0510 | 0.0393 | 0.0220 | 0.0767 | 0.1332 |