# Website validation update — 2026-10-29

The new metric laboratory was independently executed in both Live SR and Satellite Explorer for blur, noise, brightness, contrast, shift and sharpening. The reference is a deterministic64×64 synthetic RGB pattern, not an uploaded or acquired image. Tests cover exact zero-strength identity, direct NumPy recomputation of MAE/MSE/PSNR, invalid parameters and deterministic distorted pixels. The first test run found tiny zero-contrast roundoff; the algebra was corrected to preserve identity. The rerun passed17tests. Both-section AppTest executed all controls, verified no network and cleared stale output after parameter changes.

The shared seven-family availability view matches hosted artifacts through recorded training-checkpoint hashes. M1/M7 have pinned hosted RGB checkpoints; M2/M6 have local RGB pilots awaiting remote distribution; M3/M4/M5 verified historical checkpoints are four-band. This view does not complete seven-model execution. Scientific SAM and LPIPS are not reported for the synthetic encoded-RGB panel. Upload/satellite reference accuracy remains unavailable without genuine aligned HR.

The remaining mandatory requirements include full individual native lesson acceptance, compatible checkpoints for every supported mode, satellite multi-model comparison, real paired metrics, full scientific export validation and deployed desktop/mobile/restart verification. No new frozen research result, geographic generalization or public deployment is claimed.

## Satellite comparison execution follow-up

Two genuine cached hosted CNN checkpoints executed sequentially on one synthetic georeferenced5x3fixture, each exporting10x6 pixels with CRS32632 and unchanged extent. Source hashes match. An injected second-model error preserved the first output; targeted retry succeeded without changing it. Scale changes removed stale cached outputs. The UI exposes both Single model and Compare models workflows, per-model GeoTIFF/provenance downloads and measured runtime table/chart. This is display-RGB engineering evidence, not provider acquisition or scientific reflectance acceptance. Synchronized zoom, baseline/detail panels, disagreement maps, all-seven execution and scientific mask verification remain pending. See reports/satellite-comparison-validation.json.

PR40 was externally merged despite a failing Streamlit CI check: an old positional button selector clicked the new teaching control. The upload test now selects its stable key; the seven-section check passed locally. The follow-up still requires its own CI.
