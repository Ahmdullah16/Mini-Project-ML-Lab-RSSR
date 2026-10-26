# Exact input and scale contract

Input width and height must each be 1..1024, independently. Odd and rectangular inputs are preserved. All scales 2, 4, 8, 16, 32, 64 are visible. Output is exactly (sW,sH), with s² times the pixels. 256×128 at ×8 gives 2048×1024. 1024×1024 at ×64 gives 65536×65536 (4,294,967,296 pixels).

## Execution and resource policy

All current portable models are genuinely trained native ×2 RGB checkpoints. Larger factors repeat that same model; these are explicitly labeled cascades, not independently trained ×64 models. RGB intermediates clip normalized values and quantize to uint8 after each stage. Bicubic is part of the existing model residual architecture. Preview resampling and the separately labeled Pillow bicubic comparison are interpolation. No scale reduction or input resizing occurs.

Local execution writes tiled, uncompressed BigTIFF stages using 128-pixel cores with 32-pixel context halos. Only tile tensors enter RAM; stage files remain on disk. Preview is separately area-resampled to at most512 pixels per side. A new output directory is required, and free disk is checked before execution. Interrupted RGB jobs contain failure.json and must not be presented as complete. Retained stages consume more storage than the final image; delete local job directories explicitly when no longer needed. No durability is promised on Streamlit.

Free-host policy: at most1,048,576 output pixels per model. Models execute sequentially under one process-wide lock. Configured working-memory allowance is512MiB; this is not a universal measured peak. The largest ×64 RGB request needs12GiB raw final storage and approximately16.01GiB for all retained stages plus overhead. It has not been executed. Every scale remains selectable; oversized host requests give a same-pixels/checkpoint/settings job ZIP. Download, extract and run it in a checkout containing this pipeline:

```bash
python -m pip install -r app/requirements.txt
# Set PYTHONPATH to the checkout's src directory, or install the package.
python run.py
python run.py --execute
```

The first command in the bundle prints preflight only. Colab can use the same CPU commands after extracting the ZIP and setting its checkout source path. Neither runtime capacity nor enough free Colab disk is assumed. No paid service is activated.

Direct RGB CLI from repository root:

```bash
python -m rssr.cli infer-rgb input.png configs/hosted-model.json output-job --scale 8
python -m rssr.cli infer-rgb input.png configs/hosted-model.json output-job --scale 8 --execute
```

Scientific rasters remain distinct from display images:

```bash
python -m rssr.cli infer checkpoint.pt input.tif output.tif --divisor 10000 --scale 4
# Add --execute only after reviewing the plan.
```

Float32 GeoTIFF cascades preserve CRS, bounds, band descriptions and propagated invalid masks. They do not clip or quantize intermediate radiometry. Do not treat RGB display output as reflectance or report generative output as genuine HR. Calibration, sensor shift and large-scale fidelity remain unvalidated.

## Validation evidence and limits

[Actual model checks](../reports/scale-validation.json): all3 pinned models executed all6 scales on3×5 synthetic inputs; a1024×1024 input produced2048×2048 from the pilot. On129×131 synthetic inputs, disk output equaled the original tiled pipeline for all3 models. CNN outputs matched whole-frame display output; hybrid maximum difference was1 uint8 level, seam-strip MAE0.004845. Global pooling makes tiled inference approximate; no seam-free claim is made. Sampled process RSS was280,973,312 bytes, including imports/reference computations, not an allocator peak or huge-job guarantee.

[UI comparison checks](../reports/scale-comparison-validation.json) cover3 actual models at×4, identity cache, immutable labels, resource rejection and partial-failure retry. Full65536-square export and public deployment are not verified. This is implementation validation, not a new SR-quality gain or frozen test result.


Satellite Explorer uses two explicit phases: acquire the source, then preflight/run the selected scale. Its fallback ZIP includes those exact GeoTIFF bytes, source acquisition metadata and pinned model specification; it does not query for replacement imagery. [Offline satellite validation](../reports/scale-satellite-validation.json) verifies actual ×4 dimensions/CRS and host rejection. [Portable job validation](../reports/scale-portable-validation.json) verifies the exported RGB launcher and Gradio callback. [Restart validation](../reports/scale-streamlit-restart.json) records two independent empty-cache processes with identical sample outputs; sampled RSS was approximately394–396MB. This does not verify public hosting.
