# Examiner demonstration guide

This demonstrates the current engineering system and measured research scope. It is not a claim that the full laboratory or final submission package is complete. Read [readiness](submission_readiness.md) before presenting.

1. Open https://mini-project-ml-lab-rssr.streamlit.app/. Community Cloud can sleep; use its wake button. A loading or wake screen is not an inference result. If unavailable, use the local command below and explicitly identify it as local.
2. Expand Application version when available. Record the git commit and entry-point hash. Choose an accurately named model; do not describe the original CNN pilot as a hybrid. The new selector must be verified on the actual public deployment before claiming hosted hybrid inference.
3. Select the licensed example and estimate SR. Show input, bicubic and SR, then record dimensions, runtime, model/checkpoint provenance and output downloads. This example comes from training imagery and cannot prove generalization. Contrast enhancement affects previews only.
4. Upload a controlled opaque RGB PNG or JPEG, at most512×512 and8MB. The output should have twice the height/width. Demonstrate the clear invalid-input error using a513-pixel image. Downloaded PNG is a display product, not calibrated reflectance.
5. Optionally run the bounded Sentinel acquisition form. Show source acquisition time separately from ingestion/inference time. Download GeoTIFF and provenance, retaining CRS/transform/bands. A query may return no suitable imagery; report that rather than substitute a cached image and call it fresh. With no paired HR reference, do not invent PSNR.
6. Show [frozen held-out results](../reports/final-heldout.json) alongside [development comparisons](../reports/development-benchmark.md). Explain split-before-tiling, fixed synthetic degradation and frozen selection. RGB CNN and hybrid are close; the experiment does not establish hybrid superiority. One held-out NAIP aerial scene is not cross-sensor satellite validation.
7. Open MP23–MP26 notebooks for gate mathematics, controlled alternatives and three-seed SR evidence. Distinguish same-grid reconstruction inMP23–25 from super-resolution inMP26. An attention map is not a semantic explanation or segmentation result.
8. Demonstrate [frozen rebuilding](frozen-reproduction.md) into a new directory: pinned downloads,288 verified tiles, exact manifests, SQLite checks and cached rerun. Keep source data and generated files outside Git. This does not imply complete portable four-band model evaluation.
9. Explain stateless hosting. Inputs/results/cache are temporary. The local database and offline external-adapter tests do not demonstrate live durable cloud storage. Never expose credentials during the demonstration.
10. End with the actual remaining requirements: MP27–MP40 expansion, outstanding earlier criteria, durable external integration, genuine paired cross-sensor study, final report/slides/package and final candidate-wide checks. Do not mark learner mastery or submission acceptance automatically.

## Local fallback

From a clean Python3.12 environment at the repository root:

```bash
python -m pip install -r app/requirements.txt
python scripts/check_model_catalog.py
python -m streamlit run app/streamlit_app.py
```

For the research/data command, install the project and development dependencies using the root setup guide, then:

```bash
python scripts/rebuild_frozen_data.py --data-root data/examiner-rebuild
python -m pytest -q
python scripts/check_structure.py
```

Allow approximately17MB of pinned raw imagery plus manifest transfers for an empty data directory. Model assets and environment dependencies are separate. No paid service is activated by these commands. Check the current evidence rather than copying old timing/CI values into a final report.
