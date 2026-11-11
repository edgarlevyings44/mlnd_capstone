# Data audit

- Rows: 7892  |  Columns: 13  |  Duplicate rows: 0
- Class counts (0 healthy, 1 infected): {0: 5045, 1: 2847}  |  Infected: 36.07%  |  Imbalance ratio: 1.77 : 1
- Validation: schema, label values, counties and [0, 1] ranges all passed.

## Records and infection rate by county
| County | 0 | 1 | total | pct_infected |
|---|---|---|---|---|
| Bungoma | 859 | 446 | 1305 | 34.2 |
| Kakamega | 854 | 437 | 1291 | 33.8 |
| Nakuru | 823 | 498 | 1321 | 37.7 |
| Narok | 834 | 534 | 1368 | 39.0 |
| Trans Nzoia | 817 | 463 | 1280 | 36.2 |
| Uasin Gishu | 858 | 469 | 1327 | 35.3 |

## Feature summary
| feature | mean | std | min | max |
|---|---|---|---|---|
| Chlorosis_Intensity | 0.581 | 0.249 | 0.0 | 1.0 |
| Green_Yellow_Ratio | 0.573 | 0.248 | 0.0 | 1.0 |
| Necrotic_Area | 0.578 | 0.25 | 0.0 | 1.0 |
| Texture_Roughness | 0.577 | 0.248 | 0.0 | 1.0 |
| Patchiness | 0.575 | 0.248 | 0.0 | 1.0 |
| Lesion_Distribution | 0.58 | 0.249 | 0.0 | 1.0 |
| Elongated_Streaks | 0.579 | 0.252 | 0.0 | 1.0 |
| Irregular_Lesions | 0.577 | 0.251 | 0.0 | 1.0 |
| Infected_Distribution | 0.576 | 0.251 | 0.0 | 1.0 |
| Boundary_Sharpness | 0.577 | 0.249 | 0.0 | 1.0 |
| Transition_Smoothness | 0.575 | 0.248 | 0.0 | 1.0 |

## Clipping check (values exactly on 0 or 1)
`ratio` = count exactly on the bound / count in the neighbouring strip.

| feature | at_0 | near_0 | ratio_0 | at_1 | near_1 | ratio_1 | clipped |
|---|---|---|---|---|---|---|---|
| Chlorosis_Intensity | 82 | 21 | 3.9 | 290 | 41 | 7.1 | True |
| Green_Yellow_Ratio | 91 | 15 | 6.1 | 259 | 37 | 7.0 | True |
| Necrotic_Area | 98 | 12 | 8.2 | 297 | 36 | 8.2 | True |
| Texture_Roughness | 79 | 13 | 6.1 | 308 | 31 | 9.9 | True |
| Patchiness | 85 | 14 | 6.1 | 265 | 30 | 8.8 | True |
| Lesion_Distribution | 81 | 13 | 6.2 | 279 | 36 | 7.8 | True |
| Elongated_Streaks | 93 | 9 | 10.3 | 316 | 36 | 8.8 | True |
| Irregular_Lesions | 113 | 10 | 11.3 | 291 | 34 | 8.6 | True |
| Infected_Distribution | 95 | 12 | 7.9 | 326 | 31 | 10.5 | True |
| Boundary_Sharpness | 100 | 10 | 10.0 | 276 | 45 | 6.1 | True |
| Transition_Smoothness | 81 | 10 | 8.1 | 260 | 23 | 11.3 | True |

**Verdict:** 11 of 11 features show a pile-up of values exactly on 0 and/or 1 that is far larger than the density just inside the bound. That pattern is consistent with clipping of out-of-range readings, not with genuine continuous readings.

Share of values sitting exactly on 0 or 1, by class: {0: 0.033, 1: 0.0745}
