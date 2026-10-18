# Prerequisite and contribution map

Original objectives and original prerequisite prose are preserved in syllabus.json. The map below adds implementation connections; all learner mastery remains unassessed.

| Project | Prerequisites | Contribution | Intro notebook |
|---|---|---|---|
| MP01 Digital Image Explorer | Python variables and functions | Pixels, RGB channels, grayscale images, image dimensions, bit depth, histograms. | [Open](../stage_01_python_numpy/MP01_digital_image_explorer/digital_image_explorer.ipynb) |
| MP02 NumPy Image Laboratory | MP01 and prior foundations | Indexing, slicing, reshaping, broadcasting, normalization, vectorized operations. | [Open](../notebooks/MP02.ipynb) |
| MP03 Image Resizing Laboratory | MP02 and prior foundations | Nearest-neighbor, bilinear and bicubic interpolation. | [Open](../notebooks/MP03.ipynb) |
| MP04 Image Comparison Tool | MP03 and prior foundations | Image differences, absolute error, MSE and visualization. | [Open](../notebooks/MP04.ipynb) |
| MP05 Gaussian Blur Simulator | MP04 and prior foundations | Convolution, Gaussian kernels, kernel size and blur strength. | [Open](../notebooks/MP05.ipynb) |
| MP06 Low-Resolution Image Generator | MP05 and prior foundations | Downsampling, scale factors, sampling and aliasing. | [Open](../notebooks/MP06.ipynb) |
| MP07 Synthetic Degradation Laboratory | MP06 and prior foundations | Blur, noise, downsampling and degradation combinations. | [Open](../notebooks/MP07.ipynb) |
| MP08 Frequency-Domain Image Explorer | MP07 and prior foundations | Fourier transforms, spatial frequencies, edge and texture information. | [Open](../notebooks/MP08.ipynb) |
| MP09 MSE and PSNR from Scratch | MP08 and prior foundations | Reconstruction error, dynamic range and PSNR. | [Open](../notebooks/MP09.ipynb) |
| MP10 SSIM Evaluation Tool | MP09 and prior foundations | Luminance, contrast and structural similarity. | [Open](../notebooks/MP10.ipynb) |
| MP11 Super-Resolution Evaluation Dashboard | MP10 and prior foundations | Metrics, error maps, comparison figures and result tables. | [Open](../notebooks/MP11.ipynb) |
| MP12 Linear Algebra Visualizer | MP11 and prior foundations | Vectors, matrices, tensor shapes, transformations and norms. | [Open](../notebooks/MP12.ipynb) |
| MP13 Gradient Descent from Scratch | MP12 and prior foundations | Derivatives, gradients, learning rates and optimization. | [Open](../notebooks/MP13.ipynb) |
| MP14 Neural Network from Scratch | MP13 and prior foundations | Forward propagation, activation functions, loss and backpropagation. | [Open](../notebooks/MP14.ipynb) |
| MP15 First PyTorch Neural Network | MP14 and prior foundations | Tensors, autograd, optimizers and model training. | [Open](../notebooks/MP15.ipynb) |
| MP16 Reusable Training Pipeline | MP15 and prior foundations | Dataset, DataLoader, validation, checkpoints and reproducibility. | [Open](../notebooks/MP16.ipynb) |
| MP17 Convolution Playground | MP16 and prior foundations | Kernels, padding, stride, feature maps and receptive fields. | [Open](../notebooks/MP17.ipynb) |
| MP18 CNN Edge Detection Laboratory | MP17 and prior foundations | Sobel filters and learned convolutional features. | [Open](../notebooks/MP18.ipynb) |
| MP19 CNN Image Reconstruction | MP18 and prior foundations | Supervised reconstruction using convolutional neural networks. | [Open](../notebooks/MP19.ipynb) |
| MP20 SRCNN Implementation | MP19 and prior foundations | Feature extraction, nonlinear mapping and SR reconstruction. | [Open](../notebooks/MP20.ipynb) |
| MP21 Residual CNN Super-Resolution | MP20 and prior foundations | Residual blocks, skip connections and deep feature extraction. | [Open](../notebooks/MP21.ipynb) |
| MP22 Upsampling Architecture Comparison | MP21 and prior foundations | Interpolation, transposed convolution and PixelShuffle. | [Open](../notebooks/MP22.ipynb) |
| MP23 Channel Attention from Scratch | MP22 and prior foundations | Global pooling, channel weights and feature recalibration. | [Open](../notebooks/MP23.ipynb) |
| MP24 Spatial Attention from Scratch | MP23 and prior foundations | Spatial attention maps and feature weighting. | [Open](../notebooks/MP24.ipynb) |
| MP25 Combined Attention Module | MP24 and prior foundations | Channel attention, spatial attention and CBAM-style integration. | [Open](../notebooks/MP25.ipynb) |
| MP26 Attention-Based SR Network | MP25 and prior foundations | Residual attention blocks and SR performance comparisons. | [Open](../notebooks/MP26.ipynb) |
| MP27 Self-Attention from Scratch | MP26 and prior foundations | Queries, keys, values, scaled dot-product attention and softmax. | [Open](../notebooks/MP27.ipynb) |
| MP28 Multi-Head Self-Attention | MP27 and prior foundations | Attention heads, feature projections and concatenation. | [Open](../notebooks/MP28.ipynb) |
| MP29 Transformer Encoder Block | MP28 and prior foundations | Multi-head attention, normalization, residual connections and feed-forward networks. | [Open](../notebooks/MP29.ipynb) |
| MP30 Image Patches and Window Attention | MP29 and prior foundations | Patch embeddings, tokens, local windows and attention complexity. | [Open](../notebooks/MP30.ipynb) |
| MP31 Transformer-Based SR Baseline | MP30 and prior foundations | SwinIR-style image restoration and Transformer SR evaluation. | [Open](../notebooks/MP31.ipynb) |
| MP32 Satellite Image Explorer | MP31 and prior foundations | Satellite bands, spatial resolution, GeoTIFF and raster metadata. | [Open](../notebooks/MP32.ipynb) |
| MP33 Satellite Preprocessing Pipeline | MP32 and prior foundations | Band selection, normalization, invalid pixels and patch extraction. | [Open](../notebooks/MP33.ipynb) |
| MP34 Satellite HR–LR Pair Builder | MP33 and prior foundations | Paired data, synthetic degradation and alignment. | [Open](../notebooks/MP34.ipynb) |
| MP35 Geographic Dataset Splitting | MP34 and prior foundations | Scene-independent train, validation and test splits. | [Open](../notebooks/MP35.ipynb) |
| MP36 Cross-Sensor Dataset Investigation | MP35 and prior foundations | WorldStrat, Sen2Venµs, geometric and radiometric differences. | [Open](../notebooks/MP36.ipynb) |
| MP37 Feature Fusion Laboratory | MP36 and prior foundations | CNN features, Transformer features, addition and concatenation. | [Open](../notebooks/MP37.ipynb) |
| MP38 Hybrid CNN–Attention–Transformer SR Model | MP37 and prior foundations | Integrated architecture and reconstruction. | [Open](../notebooks/MP38.ipynb) |
| MP39 Controlled Ablation Study | MP38 and prior foundations | CNN, CNN + Attention, CNN + Transformer, full hybrid. | [Open](../notebooks/MP39.ipynb) |
| MP40 Robustness and Efficiency Benchmark | MP39 and prior foundations | Blur, noise, cross-sensor shift, parameters, FLOPs, inference time and GPU memory. | [Open](../notebooks/MP40.ipynb) |