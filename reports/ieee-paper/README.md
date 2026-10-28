# IEEE reconstruction workspace

This directory contains the new evidence audit and reconstruction checkpoint. The editable manuscript remains [paper/main.tex](../../paper/main.tex); the previous validated teaching/report milestone is preserved. No second divergent manuscript or falsely named final PDF is created.

The [master brief](../../docs/ieee-reconstruction-brief.md) explicitly requires reading the existing manuscript, conference template and HOWTO before reconstruction. The current accessible `Downloads/document.pdf` matches the earlier four-page export from this session; it is not evidence that a separately supplied manuscript was transferred. The two reference PDFs have not been found. Their comparison remains **unavailable**, not passed. Independent repository evidence audit continues.

- [Component comparison matrix](paper_audit.md)
- [Scientific claim register](scientific_claims.md)
- [Experiment and architecture contracts](experiment_protocol.md)
- [Readiness](publication_readiness.md)
- [Missing evidence](missing_evidence.md)
- [Resume](RESUME.md)
- [Machine-readable source audit](evidence/source-audit.json)

From repository root, with the existing development environment and `PYTHONPATH=src`, run `python scripts/audit_paper_sources.py` after an intentional source change, then `python scripts/audit_paper_sources.py --check`. Tests: `python -m pytest tests/test_paper_source_audit.py tests/test_paper_evidence.py -q`. These commands only read source/record files; they do not access imagery, checkpoints, provider APIs, training or inference.

The audit verifies exact source hashes, JSON row values, referenced TeX inputs, citation-key resolution and protocol identity. It does not establish reference accuracy, scientific truth, model quality or final visual acceptance. Preserve the frozen source records.
