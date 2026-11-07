# RSSR project basics — presentation guide

**Course:** EE5302L — Introduction to Machine Learning Lab  
**Roll number:** 264102307

## One-minute explanation

This project studies whether a computer can make a low-resolution remote-sensing image larger while keeping its structures faithful to the input. A model receives an image, extracts patterns such as edges and repeated shapes, and predicts a correction. The displayed output has more pixels; that alone does not prove that every added detail existed in the landscape.

## The basic image flow

For the compact learned-residual models, the main idea is:

```text
low-resolution image
        ├── bicubic resize (simple reference image) ───────────┐
        └── feature embedding → model blocks → reconstruction ─┤ add → output
                                                              ┘
```

In the code, an image tensor has shape `N × C × H × W`: number of images, channels, height and width. The shared feature embedding converts the input channels into a set of learned feature maps. The selected model processes these maps, and the reconstruction layer predicts a residual correction. The output is the bicubic image plus that correction:

\[
\hat{x}=B_s(x)+R_\theta(x),
\]

where `x` is the input, `B_s` is bicubic resizing at scale `s`, `Rθ` is the learned correction, and `x̂` is the prediction. At ×2, an input of `H × W` becomes `2H × 2W`; the pixel count is four times larger. More pixels do not guarantee recovery of true missing detail.

## What the main building blocks do

| Block | Beginner explanation | Basic use in this project |
|---|---|---|
| Convolution (CNN) | A small learned filter slides over nearby pixels and responds to patterns such as edges. | Builds local features and textures. |
| Attention | The model assigns different weights to feature channels or positions. | Lets useful feature responses contribute more strongly. |
| Transformer window | A small group of feature positions compares information using attention. | Adds relationships among positions within each window. |
| Fusion | Two feature paths are joined and projected back to the working width. | Combines local CNN and window-Transformer features in hybrid models. |
| Residual output | A predicted correction is added to a simple resized baseline. | Learns a change to bicubic output instead of reconstructing everything from zero. |
| PixelShuffle | Rearranges feature channels into neighboring output pixels. | Produces the requested integer enlargement in the reconstruction head. |
| L1 loss | Measures the average absolute difference from a training target. | Supplies a training signal; lower training loss alone does not prove better unseen-scene results. |

These descriptions summarize operations in `src/rssr/models.py`; the exact selected path depends on the model configuration. All seven registered families share embedding and reconstruction operations, so “Transformer” here does not mean convolution-free. The SRCNN teaching path is a separate baseline and returns its learned reconstruction directly.

## What a Python `forward()` function means

`forward()` is the model's main prediction recipe. In ordinary words, it checks the input shape, creates the bicubic baseline, converts the image into features, applies the chosen feature blocks, predicts a residual, and returns the final image. A simplified teaching sketch is:

```python
def forward(image):
    baseline = bicubic_resize(image, scale)
    features = embedding(image)
    features = selected_model_blocks(features)
    residual = reconstruction_and_pixel_shuffle(features)
    return baseline + residual
```

This is pseudocode for explaining the flow, not a replacement implementation. The real `SRModel.forward()` validates NCHW shape and band count; CNN, attention, Transformer and hybrid configurations take different branches. Some configurations also use an explicit residual gate.

## Training versus using the model

During training, the model predicts from an input, compares that prediction with a paired target using a loss, and an optimizer adjusts learned parameters. During inference, the trained parameters are fixed: the model only predicts. The deployed teaching demo does not train on a visitor's image. Evaluation metrics require a valid aligned reference; without one, this project presents images but does not invent accuracy scores.

## Short presentation script

“First, bicubic interpolation gives us a simple enlarged reference. The selected neural network learns image features and predicts a correction to that reference. CNN blocks focus on nearby patterns, attention reweights features, and Transformer windows relate positions. We compare the output only under stated data and evaluation conditions. An image can look sharper or contain more pixels without proving that real ground detail was recovered.”
