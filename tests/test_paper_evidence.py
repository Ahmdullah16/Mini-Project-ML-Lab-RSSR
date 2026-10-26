"""Guard evidence provenance and prevent pooling incompatible experiments."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_builder", ROOT / "scripts/build_paper_evidence.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_committed_tables_and_row_provenance():
    generated = builder.build()
    for path, text in generated.items():
        assert (ROOT / path).read_text(encoding="utf-8") == text
    manifest = json.loads(generated["reports/paper-evidence.json"])
    for row in manifest["rows"]:
        source = json.loads((ROOT / row["source"]).read_text(encoding="utf-8"))
        for pointer in row["pointers"]:
            value = source
            for part in pointer.strip("/").split("/"):
                value = value[int(part)] if isinstance(value, list) else value[part]
            assert value is not None
            if row["table"].startswith("frozen_"):
                assert row["table"] == f"frozen_{value['bands']}_band"
                assert row["psnr_db"] == value["psnr"]


@pytest.mark.parametrize("mutation", ["duplicate_seed", "wrong_scene_mean"])
def test_rejects_invalid_aggregation(tmp_path, mutation):
    (tmp_path / "reports").mkdir()
    for name in ("ablation-pilot-runs.json", "geographic-study.json", "final-heldout.json"):
        shutil.copyfile(ROOT / "reports" / name, tmp_path / "reports" / name)
    shutil.copyfile(ROOT / "references.bib", tmp_path / "references.bib")
    name = "ablation-pilot-runs.json" if mutation == "duplicate_seed" else "geographic-study.json"
    path = tmp_path / "reports" / name
    rows = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "duplicate_seed":
        rows[0]["seed"] = 999
    else:
        rows[0]["scene_mean_delta_db"] += 1
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(AssertionError):
        builder.build(tmp_path)
