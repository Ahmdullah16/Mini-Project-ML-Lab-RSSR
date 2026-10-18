# Next session

Resume from local feature branch feat/reproducible-rssr-lab. Do not restart.

Local commits: e870de1 (preserved syllabus/shared pipeline), 7c21f4e (validation
extensions), dadf319 (corrected explicit uint16 scaling after inspecting real data).
GitHub currently has minimal main only; shell push failed in Windows credential
helper, so use connected GitHub create_tree/create_commit/create_branch and a draft
PR. Do not merge. Local untracked reports/ contains real run summaries to preserve.

Last verified commands from repository root with work/venv Python:

* scripts/pilot.py: five 100-step models completed; validation only.
* scripts/report.py: generated measured table; every model below bicubic.
* scripts/demo_pilot.py: RGB checkpoint data/runs/b5f690d95fe4422b88f6f728ce6d6c2c/best.pt.
* MP01 unittest: 10 passed after workspace TEMP/TMP configuration.
* exact resume check: four steps equals two + resume bit-for-bit.
* scripts/check_structure.py: 40 IDs, 317 links at that revision.
* refresh configs/stac.json: 10 added, then 0 duplicates; partial page budget.

Next action: finish full pytest, install missing Jupyter/Gradio dependencies from
work/wheels, execute all notebooks in a real kernel, exercise local HTTP demo
inference, record failures/results, then publish reviewable branch via connector.
The prior transfer process may no longer exist after interruption; inspect wheel
files and installed packages before redownloading. work/fetch_wheels.py records
checksums and bounded transfers when completed. Existing venv shares bundled
read-only scientific packages; no system installs are necessary.

Resource limits: 2 GB new download ceiling and 2 GPU-hours. Three data assets total
5,627,736 bytes. CPU-only execution used 0 GPU-hours. Dependency transfer accounting
must distinguish unique cached payloads and duplicate/interrupted network attempts.

Remaining work is listed honestly in progress.md. Public hosting credential and
free-compute entitlement are absent; do not claim a hosted URL. Larger research
resource estimates must follow actual pilot measurements, not invented throughput.
