# MP01 validation record

Implementation prepared for review. MP01 remains **Not started** in the learner progress record; no learner exercises have been assessed.

## Observed environment

Python 3.12.14; NumPy 2.3.5; Matplotlib 3.10.8; Pillow 12.3.0; CPU execution.

## Commands and outcomes

From the MP01 directory:

```bash
python -m unittest discover -s . -p "test_*.py" -v
python run_notebook.py
```

- All **10 tests passed**. They cover known synthetic pixels, shape/dtype, normalization endpoints and midpoint, independently computed primary-color gray values, histogram counts, brightness saturation and input immutability, invalid inputs, lossless PNG/grayscale loading, palette loading, EXIF orientation, rejected high-bit-depth/transparent data, and a missing path.
- The source notebook's **15 Python code cells ran sequentially** in a fresh Python process through the supplied runner. All assertions passed; **12 figures** were captured in a local executed notebook. The two generated PNG files were written successfully.
- The optional user-image cell printed its skip message. No user-supplied satellite image was available, so that branch was not executed on real satellite imagery. The loader was exercised with local generated fixtures in tests.

## Failures and limits

Jupyter's `nbformat`, `nbclient`, and `ipykernel` packages were unavailable. Attempting to install them into a scratch dependency directory failed with a package-proxy connection error, followed by pip's inability to find distributions. No interactive Jupyter kernel or nbclient execution is claimed. The fallback executes ordinary Python cells and captures Matplotlib figures; it is not a complete Jupyter implementation.

Matplotlib initially reported a read-only home cache; the lesson/runner explicitly use a temporary writable Matplotlib cache. Fontconfig emitted `No writable cache directories` during the first headless plot run. Plot rendering still completed and no cell or test failed. The six-pixel figure was visually inspected for correct colored blocks, axes, and labels.

Generated images and the executed notebook are ignored under `outputs/`, not committed. The source notebook retains empty output cells. No external datasets, training, downloads from notebook cells, scientific raster processing, or learner mastery validation occurred.

## Structural and scope checks

Validated all **309 relative Markdown links** in the local laboratory, including the new MP01 links. Checked notebook JSON structure, unique cell IDs, 13 experiment headings, 15 code cells, and empty source execution counts/outputs. This structural check is not nbformat schema validation. Checked changed source files for trailing whitespace. SHA-256 comparison against the pre-change snapshot confirmed every existing file outside MP01 remained byte-for-byte unchanged, including MP02–MP40 and the shared progress tracker. A repeat run with the exact README test and runner commands passed; the first-run Fontconfig warning did not recur.
