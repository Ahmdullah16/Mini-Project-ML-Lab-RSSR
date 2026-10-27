# Versioned C/A/T architectures

The authoritative expanded request is [the seven-model completion brief](seven-model-completion-brief.md).
`rssr.architecture_registry.build(id, bands=3, scale=2, width=16, window=4)` returns a trainable **untrained** module. It does not fetch, borrow or rename a checkpoint. Existing `SRModel` names and state dictionaries remain unchanged.

C is an explicit two-layer 3×3 convolutional residual feature branch, A is additional channel/spatial feature gating, and T is a window Transformer with its inherent multi-head self-attention and feed-forward block. Removing A does **not** remove self-attention from T. Every configuration still uses shared 3×3 convolutional embedding and reconstruction, PixelShuffle and a bicubic residual baseline. None is claimed convolution-free.

| ID | C | A | T | Corresponding historical structure (not provenance binding) |
|---|---|---|---|---|
| cat-v1/M1 | yes | no | no | cnn |
| cat-v1/M2 | no | yes | no | new |
| cat-v1/M3 | no | no | yes | transformer |
| cat-v1/M4 | yes | yes | no | attention |
| cat-v1/M5 | yes | no | yes | hybrid, use_attention=False |
| cat-v1/M6 | no | yes | yes | new |
| cat-v1/M7 | yes | yes | yes | hybrid, use_attention=True |

## Computation and diagrams

Let E be convolutional embedding, R reconstruction followed by PixelShuffle, B bicubic upsampling, L(f)=f+C(f), and F concatenate two branches then apply a learned 1×1 projection. Each row defines the entire feature path before R:

| ID | Module path |
|---|---|
| M1 | input → E → L → R + B(input) |
| M2 | input → E → A → R + B(input) |
| M3 | input → E → T → R + B(input) |
| M4 | input → E → L → A → R + B(input) |
| M5 | input → E → parallel L,T → F → R + B(input) |
| M6 | input → E → T → A → R + B(input) |
| M7 | input → E → parallel L,T → F → A → R + B(input) |

Thus y=B(x)+R(g(E(x))), where g is the row's feature path. For a 5×3 input and scale 8, the exact output is 40×24: 64 times as many pixels. Repeating a native ×2 block three times is a cascade, not native ×8 training or evidence of recovered detail. Shared output behavior includes bicubic interpolation; RGB disk cascades additionally clip and quantize each stage.

## Observed evidence and honest status

[Machine-readable registry and 42-row export smoke matrix](../reports/architecture-smoke-matrix.json) records explicit configurations, parameter counts, source hashes, environment, seed, dimensions, timings, output pixel hashes and process RSS snapshots. Checkpoint and training provenance are null: these particular width-4 configurations were randomly initialized. All seven are implemented; this milestone does not claim trained, scientifically evaluated or deployed status. A historical structural mapping alone cannot establish identical training provenance.

Twenty focused tests verify seven distinct parameter signatures, actual C/A/T module presence/absence, finite forward/backward behavior, odd/square/single-pixel inputs, four-band native ×4 reconstruction, and exact outputs after strict historical state-dictionary loading for five structures. The export matrix executes a synthetic 5×3 RGB fixture at every requested scale through the existing disk-backed cascade. Temporary full-resolution TIFFs are inspected then removed. RSS before/after is **not peak memory**. These are implementation smoke tests, never model-quality measurements.

Reproduce from the repository root with `PYTHONPATH=src`:

```text
python -m pytest tests/test_architecture_registry.py -q
python scripts/check_architecture_matrix.py
```

A uses global channel pooling and therefore depends on tile context. T uses aligned windows and replicated boundary padding. Existing overlap logic is reused, but new-variant seam tolerances, representative-image comparisons, maximum-size exports, trained checkpoint binding, CLI/app selection, matched budgets and final trained model/scale evidence remain unfinished. Native ×2/×4 code support does not imply trained weights at either scale. Positive band counts are code support, not trained sensor compatibility.
