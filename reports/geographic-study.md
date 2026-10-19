# Geographic development diagnostic

Six completed CPU runs; zero-initialized residual, 1000 steps, four train and four validation scenes, 16 patches per scene. Selection uses validation L1 every 100 steps. All rows are validation-only; the original test remains unused.
| Model | Seed | Mean scene PSNR gain over bicubic (dB) | Seconds |
|---|---:|---:|---:|
| cnn | 17 | 1.8529 | 5.43 |
| cnn | 23 | 1.2894 | 5.42 |
| cnn | 41 | 1.5704 | 7.10 |
| hybrid | 17 | 1.6490 | 15.12 |
| hybrid | 23 | 1.1575 | 15.82 |
| hybrid | 41 | 1.1946 | 14.12 |

cnn: mean across seeds 1.5709 dB; sample standard deviation 0.2818 dB. These are training-seed variations, not geographic confidence intervals.

hybrid: mean across seeds 1.3337 dB; sample standard deviation 0.2737 dB. These are training-seed variations, not geographic confidence intervals.

The expanded training data and zero initialization together produce validation gains over bicubic. Because both training distribution and initialization differ from the original pilot, this comparison does not isolate a causal effect of either change. CNN exceeds hybrid on this small development protocol; a novelty claim is unsupported. Four validation scenes are still a small, selected geographic sample. These results do not establish cross-sensor satellite performance, physical reflectance accuracy or held-out generalization.

Reproduce or verify completed hashes: `python scripts/geographic_study.py`. Configuration is frozen in `configs/geographic-study.json`; source scenes/checksums/splits are in `configs/naip-development-manifest.json`. Individual results and checkpoint hashes are in `reports/geographic-study.json`. Completed runs are skipped only after evidence checks; a mid-run interruption may leave an incomplete database run and requires explicit review.

