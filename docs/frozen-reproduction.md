# Reproduce the frozen data and inference

The frozen selection is in `configs/final-evaluation.json`. Do not change it or select another model based on the held-out results. The local protocol was committed at `9109766` before evaluation; published commit `f9e4880` has the identical tree, but its publication occurred later. `reports/final-heldout.json` records the original result, not an experiment to optimize further.

## Fresh data directory

After installing the project as documented in the root setup guide:

```bash
python scripts/rebuild_frozen_data.py --data-root data/frozen-rebuild
```

This downloads four checksum-pinned original-byte manifests and nine pinned public NAIP scenes (16,883,208 raw bytes on the measured empty-cache run), validates geographic splits, regenerates 144 tiles for each of RGB and RGBNIR, compares the entire prepared-manifest structure and every tile checksum, and checks database integrity/foreign keys. Only after semantic and tile equality does it restore the published manifest bytes. This avoids OS-dependent JSON newline conversion changing a frozen hash. The operation never trains or evaluates a model. Run it again to verify an existing directory; valid raw files are reused, corruption fails closed.

Windows empty-directory rebuilding passed; Linux reconstruction is an explicit CI gate, not inferred from Windows success. The destination is a new research data directory, not the public app cache. Source images, generated arrays and SQLite stay outside Git. The public manifests use immutable artifact commit `b21b797a2c43f3a45879cd6778be4cd81600f2e7`; the raw source URLs also pin a dataset revision. There is no dependency on a previous computer's path.

## Genuine inference checkpoints

```bash
python scripts/check_model_catalog.py
python -m streamlit run app/streamlit_app.py
```

The catalog test requires `app/requirements.txt` and downloads all three published RGB checkpoints into an empty temporary cache, verifies SHA-256, then executes real sample and maximum-size inference. CNN/hybrid inference weights come from the frozen selected training runs; the original pilot remains available. Stripping optimizer state changes the serialized file checksum, so the manifest retains the original training-checkpoint checksum as separate provenance.

Four-band selected checkpoints and a fully portable fresh-checkout research evaluation command remain unfinished. Do not claim the RGB inference files reproduce optimizer resume, all research weights, or durable hosted storage. Rebuilt datasets are verified without repeating the frozen held-out evaluation.
