# Router Redistribution v2 — Layer 9 / wikitext

**Tokens**: 32508  |  **Hidden dim**: 2048  |  **Samples**: 200

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Routing weight | 0.0481 | 0.0087 | 0.0292 | 0.0800 |
| Gap norm (functional importance) | 0.0261 | 0.0157 | 0.0000 | 0.0711 |
| Compensation cosine | 0.0122 | 0.0069 | 0.0000 | 0.0295 |
| Relative compensation | 0.0759 | 0.1367 | 0.0000 | 1.0946 |
| Token coverage | 0.1250 | 0.1017 | 0.0026 | 0.4283 |

## Spearman Correlations

| Comparison | ρ | p-value | Interpretation |
|------------|---|---------|----------------|
| routing_weight → gap_norm | 0.6707 | 0.0000 | Does routing predict functional importance? |
| gap_norm → rel_compensation | 0.8001 | 0.0000 | Do important experts have less redundancy? |
| routing_weight → rel_compensation | 0.5166 | 0.0000 | Does routing predict redundancy? |
| comp_cosine → gap_norm | 0.8570 | 0.0000 | Do larger gaps get more directional compensation? |

## Most Functionally Important Experts (highest gap norm)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 17 | 0.0431 | 0.0711 | 0.0226 | 0.1074 | 0.1577 |
| 2 | 7 | 0.0505 | 0.0695 | 0.0253 | 0.0784 | 0.4139 |
| 3 | 50 | 0.0608 | 0.0602 | 0.0230 | 0.1049 | 0.2375 |
| 4 | 63 | 0.0556 | 0.0556 | 0.0295 | 0.1045 | 0.1525 |
| 5 | 6 | 0.0578 | 0.0548 | 0.0195 | 1.0946 | 0.4236 |
| 6 | 18 | 0.0522 | 0.0504 | 0.0155 | 0.0749 | 0.3482 |
| 7 | 32 | 0.0513 | 0.0447 | 0.0149 | 0.0507 | 0.1019 |
| 8 | 8 | 0.0586 | 0.0445 | 0.0059 | 0.0415 | 0.1103 |
| 9 | 48 | 0.0535 | 0.0428 | 0.0287 | 0.1209 | 0.1308 |
| 10 | 20 | 0.0455 | 0.0423 | 0.0156 | 0.3315 | 0.3433 |

## Most Redundant Experts (highest relative compensation)

| Rank | Expert | Routing weight | Gap norm | Comp cosine | Rel compensation | Token coverage |
|------|--------|---------------|----------|-------------|-----------------|----------------|
| 1 | 6 | 0.0578 | 0.0548 | 0.0195 | 1.0946 | 0.4236 |
| 2 | 20 | 0.0455 | 0.0423 | 0.0156 | 0.3315 | 0.3433 |
| 3 | 42 | 0.0464 | 0.0220 | 0.0142 | 0.1940 | 0.0862 |
| 4 | 48 | 0.0535 | 0.0428 | 0.0287 | 0.1209 | 0.1308 |
| 5 | 9 | 0.0496 | 0.0276 | 0.0180 | 0.1101 | 0.1156 |
| 6 | 17 | 0.0431 | 0.0711 | 0.0226 | 0.1074 | 0.1577 |
| 7 | 37 | 0.0483 | 0.0394 | 0.0278 | 0.1051 | 0.1250 |
| 8 | 50 | 0.0608 | 0.0602 | 0.0230 | 0.1049 | 0.2375 |
| 9 | 63 | 0.0556 | 0.0556 | 0.0295 | 0.1045 | 0.1525 |
| 10 | 36 | 0.0505 | 0.0304 | 0.0200 | 0.1011 | 0.1256 |