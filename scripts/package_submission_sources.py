"""Package existing submission sources without inference or scientific regeneration."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def package(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sources = ["paper/main.tex", "paper/literature-context.tex", "paper/references.bib", "paper/supplement.md",
               "docs/model-instructor.md", "docs/student-project-guide.md",
               "docs/examiner-demo.md", "docs/frozen-reproduction.md",
               "reports/paper-evidence.json", "reports/final-heldout.json"]
    sources += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "paper/generated").glob("*.tex"))]
    entries = []
    for source in sources:
        data = (ROOT / source).read_bytes()
        entries.append({"path": source, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"status": "source package; not a final submission",
                "files": entries,
                "missing_deliverables": ["Visually reviewed final report PDF",
                    "Visually reviewed final IEEE PDF", "Final presentation",
                    "Complete seven-model comparison under a matched protocol"],
                "scope": "Recorded evidence copied unchanged; no training or evaluation performed"}
    archive = destination / "RSSR_Submission_Sources.zip"
    if archive.exists():
        raise FileExistsError("Use a new output directory to preserve previous packages")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for entry in entries:
            bundle.write(ROOT / entry["path"], entry["path"])
        bundle.writestr("submission-manifest.json", json.dumps(manifest, indent=2))
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for entry in entries:
            assert hashlib.sha256(bundle.read(entry["path"])).hexdigest() == entry["sha256"]
    print(f"Verified {len(entries)} source files: {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    package(parser.parse_args().output)
