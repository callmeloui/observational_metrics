# Router Redistribution v2 — Layer 6 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0393 | 0.0090 | 0.0227 | 0.0751 |
| Gap norm (functional importance) | 0.0137 | 0.0135 | 0.0002 | 0.0781 |
| Compensation cosine | 0.0095 | 0.0061 | 0.0001 | 0.0265 |
| Relative compensation | 0.0463 | 0.0304 | 0.0002 | 0.1225 |
| Token coverage | 0.1250 | 0.1107 | 0.0029 | 0.4706 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.7125 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.7018 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.4154 | 0.0006 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8612 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 18 | 0.0513 | 0.0781 | 0.0265 | 0.0682 | 0.3105 |
| 2 | 21 | 0.0414 | 0.0618 | 0.0231 | 0.0815 | 0.1246 |
| 3 | 19 | 0.0515 | 0.0394 | 0.0131 | 0.0618 | 0.2505 |
| 4 | 0 | 0.0358 | 0.0310 | 0.0159 | 0.0702 | 0.1376 |
| 5 | 9 | 0.0517 | 0.0299 | 0.0151 | 0.1123 | 0.1253 |
| 6 | 15 | 0.0370 | 0.0295 | 0.0135 | 0.0517 | 0.3725 |
| 7 | 20 | 0.0605 | 0.0281 | 0.0152 | 0.0613 | 0.1371 |
| 8 | 28 | 0.0381 | 0.0259 | 0.0142 | 0.0662 | 0.1497 |
| 9 | 39 | 0.0442 | 0.0249 | 0.0115 | 0.0532 | 0.1175 |
| 10 | 61 | 0.0451 | 0.0220 | 0.0137 | 0.0563 | 0.1632 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 27 | 0.0284 | 0.0196 | 0.0234 | 0.1225 | 0.0499 |
| 2 | 49 | 0.0335 | 0.0032 | 0.0064 | 0.1217 | 0.3726 |
| 3 | 23 | 0.0330 | 0.0043 | 0.0055 | 0.1178 | 0.1048 |
| 4 | 9 | 0.0517 | 0.0299 | 0.0151 | 0.1123 | 0.1253 |
| 5 | 1 | 0.0586 | 0.0181 | 0.0123 | 0.1080 | 0.1146 |
| 6 | 40 | 0.0427 | 0.0104 | 0.0136 | 0.0984 | 0.4706 |
| 7 | 45 | 0.0323 | 0.0158 | 0.0183 | 0.0914 | 0.2392 |
| 8 | 59 | 0.0399 | 0.0091 | 0.0157 | 0.0902 | 0.4317 |
| 9 | 21 | 0.0414 | 0.0618 | 0.0231 | 0.0815 | 0.1246 |
| 10 | 0 | 0.0358 | 0.0310 | 0.0159 | 0.0702 | 0.1376 |