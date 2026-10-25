# Inference implementation diagnostic — 2026-10-25

Question: do RGB preprocessing, display conversion or tile context change the declared output? Protocol was committed as `3d41dd7` before execution. [Protocol](../configs/inference-diagnostic-v1.json) · [Measured records](inference-diagnostic-v1.json).

All three existing published RGB checkpoints executed on a synthetic 129×131 color/edge fixture and one checksum-pinned NAIP training crop. No download, optimizer step, validation selection or frozen-test access occurred. Raw DN/10000, Gaussian sigma0.8 and area x2 generated the real crop's synthetic LR; the hosted RGB path then quantizes it to uint8 and converts back using /255. The mean input quantization difference was 0.000984913 and maximum 0.001960754 normalized units, consistent with half an 8-bit level. This is a representation change, not an accuracy result.

The shared display function exactly matched the explicit tiled float output followed by fixed [0,1] clipping and rounding for all six runs. Baseline interpolation also matched. RGB channel controls passed. No hidden sharpening, per-channel stretch or channel swap was found in this tested path.

The hybrid's full/tiled float maximum difference was 0.00171679 on synthetic edges; 0.093694% of output channel values differed after uint8 conversion. On the training crop the maximum difference was 0.0000031814 and no displayed values differed. CNN differences were numerical-scale; CNN v2 had one rounded channel-value difference on the crop. Global channel pooling uses different tile context, so exact hybrid equality is not promised. These two small inputs do not bound seams for every image or size.

No raw output values clipped on the training crop. Synthetic edges deliberately contain extreme colors; out-of-range fractions were 34.2145% (pilot), 9.9365% (CNN v2) and 20.6856% (hybrid). Fractions count scalar channel values, not spatial pixels; this is not a reconstruction error rate. Clipping can remove overshoot, but the diagnostic cannot say whether those predictions were physically correct.

The app now records input ranges/channel order/dtype, raw output range and clipped scalar fraction, and rejects nonfinite predictions. Pixels and checkpoint selection remain unchanged. Six affected numerical/identity tests and actual three-model comparison/cache/failure-recovery checks passed. A metadata cache-version change prevents stale diagnostic metadata being reused. Timing is single-forward diagnostic scope, not a warmed efficiency benchmark.

**Conclusion:** no preprocessing mismatch was demonstrated. Display quantization and approximate global-context tiling are measurable. A quality gain is not established. Further fidelity experiments need a versioned development protocol and suitable independent references; the frozen test stays untouched. Synthetic NAIP training does not establish real satellite or arbitrary-upload performance.
