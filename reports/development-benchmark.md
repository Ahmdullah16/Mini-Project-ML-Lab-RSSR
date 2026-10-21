# Development benchmark

Four development scenes; bootstrap intervals are descriptive and unstable at n=4. Seed selection is optimistic. SAM/per-band RMSE use normalized DN, not reflectance. Synthetic blur/noise is not real cross-sensor validation. RSS includes interpreter and cached allocations; sampled peak is not allocator peak.

| Bands | Variant | Mean PSNR delta dB | Seed SD | Scene bootstrap 95% |
|---|---|---:|---:|---|
| 4 | attention | 1.7947 | 0.1873 | [1.4641441163799154, 2.145661745701582] |
| 4 | cnn | 1.5709 | 0.2818 | [1.2560282308804083, 1.9086789182619182] |
| 4 | hybrid | 1.3337 | 0.2737 | [1.0798165452525232, 1.5875749783463733] |
| 4 | hybrid_add | 1.5524 | 0.1171 | [1.221863041547698, 1.9429756180569866] |
| 4 | hybrid_consistency | 1.3774 | 0.3486 | [1.117166305001554, 1.637554400640017] |
| 4 | hybrid_gate | 1.6657 | 0.1624 | [1.3552614329988846, 1.997627988173611] |
| 4 | hybrid_half | 1.6226 | 0.2078 | [1.3291290333344474, 1.9161196214883582] |
| 4 | hybrid_no_attention | 0.9904 | 0.3846 | [0.8328528719430146, 1.147994370361604] |
| 4 | srcnn_educational | -4.9360 | 0.2811 | [-6.463211203333075, -3.7520283292114938] |
| 4 | transformer | 0.6180 | 0.2463 | [0.5129462056881025, 0.7230601465232951] |
| 3 | cnn | 1.6697 | 0.0608 | [1.3738521023494326, 1.9655329269455826] |
| 3 | hybrid | 1.6699 | 0.0902 | [1.3947503915539237, 1.945092922523307] |
