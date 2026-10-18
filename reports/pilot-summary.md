# Measured pilot validation results

One validation scene. Synthetic degradation on real NAIP aerial imagery; no final test or geographic confidence interval.

| Model | Method | PSNR dB | Parameters | Training seconds |
|---|---|---:|---:|---:|
| cnn | bicubic | 45.8104 | 7552 | 0.89 |
| cnn | bilinear | 44.4413 | 7552 | 0.89 |
| cnn | model | 37.6707 | 7552 | 0.89 |
| cnn | nearest | 44.7848 | 7552 | 0.89 |
| attention | bicubic | 45.8104 | 7799 | 0.99 |
| attention | bilinear | 44.4413 | 7799 | 0.99 |
| attention | model | 42.8686 | 7799 | 0.99 |
| attention | nearest | 44.7848 | 7799 | 0.99 |
| transformer | bicubic | 45.8104 | 5392 | 1.17 |
| transformer | bilinear | 44.4413 | 5392 | 1.17 |
| transformer | model | 30.5688 | 5392 | 1.17 |
| transformer | nearest | 44.7848 | 5392 | 1.17 |
| hybrid | bicubic | 45.8104 | 10807 | 1.78 |
| hybrid | bilinear | 44.4413 | 10807 | 1.78 |
| hybrid | model | 43.2617 | 10807 | 1.78 |
| hybrid | nearest | 44.7848 | 10807 | 1.78 |
| srcnn | bicubic | 45.8104 | 12532 | 1.00 |
| srcnn | bilinear | 44.4413 | 12532 | 1.00 |
| srcnn | model | 25.7763 | 12532 | 1.00 |
| srcnn | nearest | 44.7848 | 12532 | 1.00 |
