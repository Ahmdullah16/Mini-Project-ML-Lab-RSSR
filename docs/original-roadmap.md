# Mini Project ML Lab

**Attention- and Transformer-Based Deep Learning for Super-Resolution of Remote Sensing Images**

A beginner-friendly laboratory that builds from image arrays to reproducible research through 40 focused mini-projects.

**Curriculum status: proposed for review.** Project titles and objective topics are preserved from the live repository syllabus. Detailed mathematics, experiments, and completion criteria are proposed expansions for review. All 40 projects are **Not started**. This change contains documentation only: there are no implementation notebooks, trained models, measured results, or completed experiments.

## Start here

Basic Python syntax and familiarity with files are enough to begin. Work in numerical order; each project lists its prerequisites and a small experiment before more costly training. CPU-only work is sufficient for the foundations. Later model experiments may benefit from a GPU; use tiny fixtures and short smoke runs first.

1. Create a Python 3.11 or 3.12 environment.
2. Install the starter dependencies below.
3. Open MP01 from the roadmap and read its completion criteria.
4. Record actual work in the [progress tracker](progress.md). Read the [contributor and agent guidance](../AGENTS.md) before changing the lab.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](../requirements.txt) is an unpinned starter stack, not a validated environment lock. For accelerator-specific PyTorch builds, follow the [official installer](https://pytorch.org/get-started/locally/). Rasterio is needed from stage 08; no data or model downloads happen automatically. Record resolved versions and hardware with future experiments.

## Complete 40-project roadmap

Each title links to a brief covering learning objectives, prerequisites, mathematics, planned experiments, deliverables, references, and completion criteria.

### Stage 01 — Python and NumPy (MP01–MP04)

| Project | Title | Objective |
| --- | --- | --- |
| MP01 | [Digital Image Explorer](../stage_01_python_numpy/MP01_digital_image_explorer/README.md) | Pixels, RGB channels, grayscale images, image dimensions, bit depth, histograms. |
| MP02 | [NumPy Image Laboratory](../stage_01_python_numpy/MP02_numpy_image_laboratory/README.md) | Indexing, slicing, reshaping, broadcasting, normalization, vectorized operations. |
| MP03 | [Image Resizing Laboratory](../stage_01_python_numpy/MP03_image_resizing_laboratory/README.md) | Nearest-neighbor, bilinear and bicubic interpolation. |
| MP04 | [Image Comparison Tool](../stage_01_python_numpy/MP04_image_comparison_tool/README.md) | Image differences, absolute error, MSE and visualization. |

### Stage 02 — Image processing (MP05–MP08)

| Project | Title | Objective |
| --- | --- | --- |
| MP05 | [Gaussian Blur Simulator](../stage_02_image_processing/MP05_gaussian_blur_simulator/README.md) | Convolution, Gaussian kernels, kernel size and blur strength. |
| MP06 | [Low-Resolution Image Generator](../stage_02_image_processing/MP06_low_resolution_image_generator/README.md) | Downsampling, scale factors, sampling and aliasing. |
| MP07 | [Synthetic Degradation Laboratory](../stage_02_image_processing/MP07_synthetic_degradation_laboratory/README.md) | Blur, noise, downsampling and degradation combinations. |
| MP08 | [Frequency-Domain Image Explorer](../stage_02_image_processing/MP08_frequency_domain_image_explorer/README.md) | Fourier transforms, spatial frequencies, edge and texture information. |

### Stage 03 — Image metrics (MP09–MP11)

| Project | Title | Objective |
| --- | --- | --- |
| MP09 | [MSE and PSNR from Scratch](../stage_03_metrics/MP09_mse_and_psnr_from_scratch/README.md) | Reconstruction error, dynamic range and PSNR. |
| MP10 | [SSIM Evaluation Tool](../stage_03_metrics/MP10_ssim_evaluation_tool/README.md) | Luminance, contrast and structural similarity. |
| MP11 | [Super-Resolution Evaluation Dashboard](../stage_03_metrics/MP11_super_resolution_evaluation_dashboard/README.md) | Metrics, error maps, comparison figures and result tables. |

### Stage 04 — Mathematics and PyTorch (MP12–MP16)

| Project | Title | Objective |
| --- | --- | --- |
| MP12 | [Linear Algebra Visualizer](../stage_04_math_pytorch/MP12_linear_algebra_visualizer/README.md) | Vectors, matrices, tensor shapes, transformations and norms. |
| MP13 | [Gradient Descent from Scratch](../stage_04_math_pytorch/MP13_gradient_descent_from_scratch/README.md) | Derivatives, gradients, learning rates and optimization. |
| MP14 | [Neural Network from Scratch](../stage_04_math_pytorch/MP14_neural_network_from_scratch/README.md) | Forward propagation, activation functions, loss and backpropagation. |
| MP15 | [First PyTorch Neural Network](../stage_04_math_pytorch/MP15_first_pytorch_neural_network/README.md) | Tensors, autograd, optimizers and model training. |
| MP16 | [Reusable Training Pipeline](../stage_04_math_pytorch/MP16_reusable_training_pipeline/README.md) | Dataset, DataLoader, validation, checkpoints and reproducibility. |

### Stage 05 — CNN super-resolution (MP17–MP22)

| Project | Title | Objective |
| --- | --- | --- |
| MP17 | [Convolution Playground](../stage_05_cnn_sr/MP17_convolution_playground/README.md) | Kernels, padding, stride, feature maps and receptive fields. |
| MP18 | [CNN Edge Detection Laboratory](../stage_05_cnn_sr/MP18_cnn_edge_detection_laboratory/README.md) | Sobel filters and learned convolutional features. |
| MP19 | [CNN Image Reconstruction](../stage_05_cnn_sr/MP19_cnn_image_reconstruction/README.md) | Supervised reconstruction using convolutional neural networks. |
| MP20 | [SRCNN Implementation](../stage_05_cnn_sr/MP20_srcnn_implementation/README.md) | Feature extraction, nonlinear mapping and SR reconstruction. |
| MP21 | [Residual CNN Super-Resolution](../stage_05_cnn_sr/MP21_residual_cnn_super_resolution/README.md) | Residual blocks, skip connections and deep feature extraction. |
| MP22 | [Upsampling Architecture Comparison](../stage_05_cnn_sr/MP22_upsampling_architecture_comparison/README.md) | Interpolation, transposed convolution and PixelShuffle. |

### Stage 06 — Attention (MP23–MP26)

| Project | Title | Objective |
| --- | --- | --- |
| MP23 | [Channel Attention from Scratch](../stage_06_attention/MP23_channel_attention_from_scratch/README.md) | Global pooling, channel weights and feature recalibration. |
| MP24 | [Spatial Attention from Scratch](../stage_06_attention/MP24_spatial_attention_from_scratch/README.md) | Spatial attention maps and feature weighting. |
| MP25 | [Combined Attention Module](../stage_06_attention/MP25_combined_attention_module/README.md) | Channel attention, spatial attention and CBAM-style integration. |
| MP26 | [Attention-Based SR Network](../stage_06_attention/MP26_attention_based_sr_network/README.md) | Residual attention blocks and SR performance comparisons. |

### Stage 07 — Transformers (MP27–MP31)

| Project | Title | Objective |
| --- | --- | --- |
| MP27 | [Self-Attention from Scratch](../stage_07_transformers/MP27_self_attention_from_scratch/README.md) | Queries, keys, values, scaled dot-product attention and softmax. |
| MP28 | [Multi-Head Self-Attention](../stage_07_transformers/MP28_multi_head_self_attention/README.md) | Attention heads, feature projections and concatenation. |
| MP29 | [Transformer Encoder Block](../stage_07_transformers/MP29_transformer_encoder_block/README.md) | Multi-head attention, normalization, residual connections and feed-forward networks. |
| MP30 | [Image Patches and Window Attention](../stage_07_transformers/MP30_image_patches_and_window_attention/README.md) | Patch embeddings, tokens, local windows and attention complexity. |
| MP31 | [Transformer-Based SR Baseline](../stage_07_transformers/MP31_transformer_based_sr_baseline/README.md) | SwinIR-style image restoration and Transformer SR evaluation. |

### Stage 08 — Remote sensing (MP32–MP36)

| Project | Title | Objective |
| --- | --- | --- |
| MP32 | [Satellite Image Explorer](../stage_08_remote_sensing/MP32_satellite_image_explorer/README.md) | Satellite bands, spatial resolution, GeoTIFF and raster metadata. |
| MP33 | [Satellite Preprocessing Pipeline](../stage_08_remote_sensing/MP33_satellite_preprocessing_pipeline/README.md) | Band selection, normalization, invalid pixels and patch extraction. |
| MP34 | [Satellite HR–LR Pair Builder](../stage_08_remote_sensing/MP34_satellite_hr_lr_pair_builder/README.md) | Paired data, synthetic degradation and alignment. |
| MP35 | [Geographic Dataset Splitting](../stage_08_remote_sensing/MP35_geographic_dataset_splitting/README.md) | Scene-independent train, validation and test splits. |
| MP36 | [Cross-Sensor Dataset Investigation](../stage_08_remote_sensing/MP36_cross_sensor_dataset_investigation/README.md) | WorldStrat, Sen2Venµs, geometric and radiometric differences. |

### Stage 09 — Hybrid research (MP37–MP40)

| Project | Title | Objective |
| --- | --- | --- |
| MP37 | [Feature Fusion Laboratory](../stage_09_hybrid_research/MP37_feature_fusion_laboratory/README.md) | CNN features, Transformer features, addition and concatenation. |
| MP38 | [Hybrid CNN–Attention–Transformer SR Model](../stage_09_hybrid_research/MP38_hybrid_cnn_attention_transformer_sr_model/README.md) | Integrated architecture and reconstruction. |
| MP39 | [Controlled Ablation Study](../stage_09_hybrid_research/MP39_controlled_ablation_study/README.md) | CNN, CNN + Attention, CNN + Transformer, full hybrid. |
| MP40 | [Robustness and Efficiency Benchmark](../stage_09_hybrid_research/MP40_robustness_and_efficiency_benchmark/README.md) | Blur, noise, cross-sensor shift, parameters, FLOPs, inference time and GPU memory. |

## Research conventions

Image experiments must state shape, channel order, dtype, color space, and value range. Super-resolution comparisons must use the same pairs, degradation, scale factor, alignment, border crop, metric range, and aggregation rule. Report PSNR/SSIM alongside representative failures; stronger pixel metrics alone do not establish scientific or spectral fidelity.

Split scenes or geographic regions before producing overlapping patches. Fit normalization on training data only. Keep validation for model selection and test data locked until final evaluation. Label synthetic degradations clearly and avoid claims of real-sensor recovery without supporting data.

Store local data under `data/`, outputs under `outputs/`, and weights under `checkpoints/`; these paths are ignored by [.gitignore](../.gitignore). Document source URLs, licenses, checksums, split manifests, seeds, package versions, hardware, compute budget, and evaluation commands. Do not commit credentials or redistribute restricted data. No dataset or code license is granted by this curriculum; choose a repository license explicitly before redistribution.

## Definition of done

Complete a project only after its checklist is met, its commands and evidence are documented, and its limitations are stated. Use the [progress tracker](progress.md) as the status record. Repository scaffolding, plans, and unchecked criteria are not evidence of a completed project.
