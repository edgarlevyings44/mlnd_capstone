# Global SHAP importance (1000 training rows, final model)

Mean absolute contribution in log-odds. County columns are merged into one driver. Overlapping features (e.g. the severity score and the symptoms it is built from) share credit, and SHAP describes the model, not field causes. The risk score is not a calibrated probability.

| feature | label | mean_abs_shap | share |
|---|---|---|---|
| Severity_Score | Overall severity score | 0.324 | 0.1117 |
| Green_Yellow_Ratio | Green-to-yellow balance | 0.3134 | 0.1081 |
| Lesion_Distribution | Spread of lesions | 0.2791 | 0.0962 |
| Boundary_Sharpness | Sharpness of lesion edges | 0.2758 | 0.0951 |
| Irregular_Lesions | Irregularly shaped lesions | 0.2587 | 0.0892 |
| Transition_Smoothness | Smoothness of the healthy-to-infected change | 0.2518 | 0.0868 |
| Patchiness | Patchy colour pattern | 0.2267 | 0.0782 |
| Infected_Distribution | Spread of the infected area | 0.2152 | 0.0742 |
| Elongated_Streaks | Streak-shaped marks | 0.2138 | 0.0737 |
| Texture_Roughness | Leaf surface roughness | 0.1668 | 0.0575 |
| Necrotic_Texture_Interaction | Dead tissue x roughness | 0.1242 | 0.0428 |
| County | County | 0.0615 | 0.0212 |
| Chlorosis_Intensity | Leaf yellowing (chlorosis intensity) | 0.0571 | 0.0197 |
| Chlorotic_Necrotic_Index | Yellowing x dead-tissue index | 0.044 | 0.0152 |
| Lesion_Severity_Ratio | Streaks and irregular lesions relative to green balance | 0.0364 | 0.0126 |
| Necrotic_Area | Dead tissue (necrotic area) | 0.0316 | 0.0109 |
| Boundary_Transition_Diff | Edge sharpness minus transition smoothness | 0.0195 | 0.0067 |
