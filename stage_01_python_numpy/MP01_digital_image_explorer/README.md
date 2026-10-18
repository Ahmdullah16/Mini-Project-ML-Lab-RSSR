# MP01 — Digital Image Explorer

**Learning status: Not started.** Implementation is prepared for review; automated checks passed as recorded in [VALIDATION.md](VALIDATION.md). Learner exercises and explanations have not been assessed, so MP01 is not marked complete. The shared progress tracker is unchanged.

[Laboratory home](../../README.md) · [Stage overview](../README.md) · [Progress tracker](../../docs/progress.md)

## Learning objectives

Understand pixels, image dimensions, RGB channels, grayscale, bit depth, NumPy arrays, and histograms from zero. Predict numerical results, inspect plots, and explain why normalization, overflow, and histogram conventions matter.

## Prerequisites

No machine learning knowledge or dataset is required. The notebook introduces indexing, variables, arrays, and functions as they appear. Use Python 3.11 or 3.12, a terminal, and NumPy, Matplotlib, and Pillow. JupyterLab is needed only for interactive notebook use. The headless runner needs only the three image libraries plus the Python standard library. CPU execution is sufficient.

## Exact execution instructions

From the repository root, create an environment and install only this project's dependencies (no PyTorch needed):

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy matplotlib Pillow jupyterlab ipykernel
python -m ipykernel install --sys-prefix --name mp01 --display-name "Python (MP01)"
cd stage_01_python_numpy/MP01_digital_image_explorer
python -m jupyterlab digital_image_explorer.ipynb
```

Select **Python (MP01)** as the kernel. Run cells from the top with Shift+Enter, or use **Restart Kernel and Run All Cells**. Open the notebook from this project directory so `image_tools.py` can be imported. Dependencies require installation once; the lesson itself performs no downloads.

For automated checks, from that same MP01 directory:

```bash
python -m unittest discover -s . -p "test_*.py" -v
python run_notebook.py
```

`run_notebook.py` launches no Jupyter kernel. It executes this notebook's ordinary Python cells in sequence in a shared namespace, checks assertions, and captures Matplotlib figures using the Agg backend. It stops on an exception with a nonzero exit status. It does not support notebook magics, widgets, or general IPython display hooks. Use a fresh process as shown above for each run. This is the execution method validated in this environment; interactive Jupyter use was not available.

For a minimal headless environment, install `numpy matplotlib Pillow` and run the two automated commands above; Jupyter is unnecessary for that path.

## Mathematics

- RGB image: `I[r,c,k]`, shape `(H,W,3)`; `H×W` pixels and `H×W×3` samples. Pillow's size is `(W,H)`.
- Unsigned b-bit channel: `2**b` levels, from 0 to `2**b - 1`. uint8 RGB stores 24 bits per pixel.
- Normalization: `f = x/255`; inverse: `round(255*f)`.
- Display gray: `Y' = 0.299 R' + 0.587 G' + 0.114 B'`, applied to encoded RGB. This is not calibrated physical luminance.
- Histogram: count pixels at each integer gray value; counts sum to `H×W`, normalized counts sum to 1.

The notebook includes worked examples and explains every symbol. Helpers require explicit uint8 RGB conventions and reject ambiguous input rather than silently rescaling it.

## Experiments

The [notebook](digital_image_explorer.ipynb) contains 13 small experiments:

1. Construct a six-pixel image and inspect dimensions.
2. Generate and save a deterministic 64×96 RGB image.
3. Inspect coordinates and a twelve-pixel crop.
4. Separate RGB channels.
5. Remove blue from a copy.
6. Compare eight-, two-, and one-bit quantization.
7. Normalize and recover an image exactly.
8. Compare uint8 overflow against safe clipped brightness.
9. Compare equal-average and weighted grayscale.
10. Count and plot a grayscale histogram.
11. Compare RGB channel histograms.
12. Mirror an image while preserving its histogram.
13. Verify a PNG round trip and NumPy view/copy behavior.

Five practice exercises follow, with hints and selected self-check values. The optional user-image cell is separate from these experiments.

## Optional satellite display image

Keep `USER_IMAGE_PATH = None` for a fully self-contained run. To inspect your own image, place a small display crop in this project's ignored `data/` directory and change the optional cell to, for example:

```python
USER_IMAGE_PATH = Path("data/my_satellite_display.png")
```

Supported input is an opaque 8-bit RGB, grayscale, or palette image readable by Pillow; the first frame is used and EXIF orientation is applied. Transparent and high-bit-depth images are rejected. File/format errors are surfaced rather than silently replacing your image with synthetic data.

This path inspects display pixels only. It does not preserve CRS, affine transforms, nodata masks, scientific precision, band meanings, or reflectance units. Do not convert scientific multispectral data through this helper. Export an intentional display visualization separately and retain your original raster; geospatial processing is outside MP01. Large files may exceed memory or Pillow safety limits. Document source and license; no user file is included or uploaded by this lesson.

## Deliverables and expected outputs

- [digital_image_explorer.ipynb](digital_image_explorer.ipynb): teaching notebook with explanations, equations, 13 experiments, five exercises, and a conclusion.
- [image_tools.py](image_tools.py): deterministic fixture, validation, normalization, grayscale, histogram, brightness, and optional loader helpers.
- [test_image_tools.py](test_image_tools.py): 10 standard-library unittest tests for numerical results, shape/dtype checks, overflow, and file loading.
- [run_notebook.py](run_notebook.py): dependency-light execution and figure capture.
- [VALIDATION.md](VALIDATION.md): observed execution and limitations.

A successful run reports RGB shape `(64,96,3)`, 6144 pixels, 18432 raw bytes, grayscale shape `(64,96)`, histogram count total 6144, and probability total approximately 1. The hand-worked grayscale primary values are approximately 76.245, 149.685, and 29.070. Overflow demonstration prints 14 for wrapped `250+20`, and the safe version saturates to 255.

Expect 12 figures covering colors, crops, channels, quantization, normalization, brightness, grayscale, and histograms. The default optional branch prints that it was skipped. The runner creates `outputs/synthetic_rgb.png`, `outputs/synthetic_gray.png`, and `outputs/digital_image_explorer.executed.ipynb`. Generated files stay local under the existing ignore rules; the committed notebook has empty outputs for clean review. Test success should report `Ran 10 tests` and `OK`.

## References

- [NumPy beginners guide](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Matplotlib imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
- [Pillow modes and concepts](https://pillow.readthedocs.io/en/stable/handbook/concepts.html)
- [ITU-R BT.601](https://www.itu.int/rec/R-REC-BT.601)

The synthetic fixture is created locally from explicit numerical gradients and color blocks. It has no external dataset source. No new repository license is assigned by this change.

## Completion criteria

- [ ] Explain every concept and work through the numerical examples yourself.
- [ ] Run the notebook in your learning environment and interpret all 13 experiments.
- [ ] Complete the five practice exercises with your own explanations.
- [ ] Review automated validation and any environment-specific failures.
- [ ] Document your observations and only then update progress with evidence.

Automated implementation validation alone does not establish that a learner has completed the project.
