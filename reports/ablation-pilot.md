# Executed ablation pilot

Validation only: one NAIP scene / 16 tiles; 100 steps per model. Mean ± sample SD across three seeds describes initialization variability, not geographic uncertainty. No independent-scene confidence interval can be estimated. All settings were fixed before these runs.

| Variant | Parameters | PSNR mean ± seed SD (dB) | SAM mean (rad) | Training seconds total |
|---|---:|---:|---:|---:|
| cnn | 7552 | 36.4807 ± 1.4000 | 0.074913 | 2.48 |
| cnn_attention | 7799 | 42.5424 ± 0.7805 | 0.030556 | 4.72 |
| cnn_transformer | 10560 | 38.7524 ± 1.1036 | 0.052275 | 6.24 |
| hybrid | 10807 | 43.6869 ± 0.3711 | 0.024107 | 6.77 |
| hybrid_add | 10279 | 42.5440 ± 0.2277 | 0.029666 | 6.57 |
| hybrid_gate | 11467 | 45.2122 ± 0.0953 | 0.016890 | 7.52 |
| hybrid_constant_gate | 10807 | 45.0625 ± 0.1539 | 0.017652 | 7.83 |
| hybrid_consistency | 10807 | 43.6734 ± 0.3571 | 0.024334 | 15.97 |

Bicubic: 45.8104 dB. Compare within this fixed numerical scaling only; physical reflectance calibration is unverified.

Gate inputs contain LR only; the constant 0.5 control separates attenuation from learned conditioning. Consistency weight 0.1 applies the synthetic-naip-v1 forward operator, not a cross-sensor physics model. CNN+Transformer omits attention; additive fusion averages branch features. Parameter counts differ and are disclosed. The test split remains unused. See ablation-pilot-runs.json for measured run IDs, hashes and configurations.