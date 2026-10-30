# Website validation update — 2026-10-29

The new metric laboratory was independently executed in both Live SR and Satellite Explorer for blur, noise, brightness, contrast, shift and sharpening. The reference is a deterministic64×64 synthetic RGB pattern, not an uploaded or acquired image. Tests cover exact zero-strength identity, direct NumPy recomputation of MAE/MSE/PSNR, invalid parameters and deterministic distorted pixels. The first test run found tiny zero-contrast roundoff; the algebra was corrected to preserve identity. The rerun passed17tests. Both-section AppTest executed all controls, verified no network and cleared stale output after parameter changes.

The shared seven-family availability view matches hosted artifacts through recorded training-checkpoint hashes. M1/M7 have pinned hosted RGB checkpoints; M2/M6 have local RGB pilots awaiting remote distribution; M3/M4/M5 verified historical checkpoints are four-band. This view does not complete seven-model execution. Scientific SAM and LPIPS are not reported for the synthetic encoded-RGB panel. Upload/satellite reference accuracy remains unavailable without genuine aligned HR.

The remaining mandatory requirements include full individual native lesson acceptance, compatible checkpoints for every supported mode, satellite multi-model comparison, real paired metrics, full scientific export validation and deployed desktop/mobile/restart verification. No new frozen research result, geographic generalization or public deployment is claimed.

## Satellite comparison execution follow-up

Two genuine cached hosted CNN checkpoints executed sequentially on one synthetic georeferenced5x3fixture, each exporting10x6 pixels with CRS32632 and unchanged extent. Source hashes match. An injected second-model error preserved the first output; targeted retry succeeded without changing it. Scale changes removed stale cached outputs. The UI exposes both Single model and Compare models workflows, per-model GeoTIFF/provenance downloads and measured runtime table/chart. This is display-RGB engineering evidence, not provider acquisition or scientific reflectance acceptance. Synchronized zoom, baseline/detail panels, disagreement maps, all-seven execution and scientific mask verification remain pending. See reports/satellite-comparison-validation.json.

PR40 was externally merged despite a failing Streamlit CI check: an old positional button selector clicked the new teaching control. The upload test now selects its stable key; the seven-section check passed locally. The follow-up still requires its own CI.

## Common satellite detail-view follow-up

The bounded preview now offers shared crop centers, zoom1/2/4, a bicubic/model divider and absolute disagreement from bicubic. Bicubic is computed on the exact output grid, subject to the host pixel cap, then area-reduced to a preview no larger than256pixels per side. Both model and baseline use fixed0–1 display contrast. Disagreement is calculated before display clipping and is explicitly not error against HR. Invalid displayed pixels are black. The export bytes remain unchanged when moving the view controls. Six numerical tests passed, including rejecting mismatched dimensions/CRS/extent and oversized allocation; actual two-model AppTest passed crop/divider controls without changing export hashes. A mixed-dtype divider rounding discrepancy was fixed by keeping the display pipeline float32. Latest browser rendering and scientific mask/cloud/alignment acceptance remain unverified.

PR41 exacthead d257805c1b17d7d4b07698228ee469971df9db54 passed CI40820234069 and was merged externallye85f477. The detail-view follow-up needs its own CI.

## Seven-family deployment integration — 2026-10-30

Nine100step CPU RGB pilots for M3/M4/M5 ran across seeds17/23/41 under a committed plan, using training/validation only. Existing M2/M6 pilots were reused. The seed17 checkpoint for each of M2–M6 was exported without optimizer state; strict reload produced bit-exact synthetic output. Five small model assets plus checksum manifest were published as GitHub prerelease v0.3.0-rgb-pilots, outside Git source history. New hosted catalog entries pin SHA256 and byte budgets. Existing CNN and hybrid checkpoints remain available.

All eight catalog checkpoints across seven families passed clean-download verification and local Live SR inference, including512×512→1024×1024 CPU inference. Local multi-model Live SR and same-source satellite checks executed the full catalog and passed failure/retry/cache checks. Forty-two native/cascaded small RGB exports passed in reports/trained-rgb-capability-v2.json; this does not establish maximum-size execution or research quality.

The reported missing visitor_prose export was reproduced with stale cached Python helpers and corrected by refreshing project imports once per source/root fingerprint. An unchanged rerun preserves imports. The public failure was observed before deployment; current user evidence says the app is accessible and CNN16×16→32×32 worked. Full local release suite:496tests plus17subtests passed54.77seconds. Public rollout of this integration and full40lesson instructional acceptance are separate pending gates.
