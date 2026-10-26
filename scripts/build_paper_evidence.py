"""Generate paper tables from recorded evidence, without inference or training.

Hashes canonicalize UTF-8 text line endings, matching the claim register.
--check verifies committed generated files without changing them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def latex(text):
    return str(text).replace("_", r"\_")


def table(caption, header, rows, columns):
    return "\n".join([
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{" + caption + "}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{" + columns + "}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}}", r"\end{table}", "",
    ])


def build(root=ROOT):
    sources = {}

    def read(path):
        text = (root / path).read_text(encoding="utf-8")
        sources[path] = {"sha256_utf8_lf": canonical_hash(text)}
        return json.loads(text)

    pilot_path = "reports/ablation-pilot-runs.json"
    geo_path = "reports/geographic-study.json"
    final_path = "reports/final-heldout.json"
    pilot, geo, final = read(pilot_path), read(geo_path), read(final_path)
    rows, tables = [], []
    pilot_stats = {}
    for variant in sorted({r["variant"] for r in pilot}):
        indexed = [(i, r) for i, r in enumerate(pilot) if r["variant"] == variant]
        assert sorted(r["seed"] for _, r in indexed) == [17, 23, 41]
        params = {r["parameters"] for _, r in indexed}
        assert len(params) == 1
        values = [r["psnr"] for _, r in indexed]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        pilot_stats[variant] = mean
        rows.append([latex(variant), str(next(iter(params))), f"{mean:.4f}", f"{sd:.4f}"])
        tables.append({"table": "pilot", "row": variant, "source": pilot_path,
                       "pointers": [f"/{i}" for i, _ in indexed],
                       "mean_psnr_db": mean, "sample_seed_sd_db": sd})
    baseline = {r["bicubic_psnr"] for r in pilot}
    assert len(baseline) == 1, "Pilot baseline changed between runs"
    bicubic = baseline.pop()
    rows.append(["bicubic", "--", f"{bicubic:.4f}", "--"])
    tables.append({"table": "pilot", "row": "bicubic", "source": pilot_path,
                   "pointers": [f"/{i}/bicubic_psnr" for i in range(len(pilot))],
                   "psnr_db": bicubic})
    output = table("Early pilot: one validation scene, three seeds; SD is seed variability, not geographic uncertainty.",
                   ["Variant", "Parameters", "PSNR (dB)", "SD (dB)"], rows, "lrrr")
    rows = []
    for model in sorted({r["model"] for r in geo}):
        indexed = [(i, r) for i, r in enumerate(geo) if r["model"] == model]
        assert sorted(r["seed"] for _, r in indexed) == [17, 23, 41]
        scene_sets = {tuple(sorted(s["scene_id"] for s in r["validation_scenes"])) for _, r in indexed}
        assert len(scene_sets) == 1 and len(next(iter(scene_sets))) == 4
        values = []
        for _, record in indexed:
            value = statistics.mean(s["model"] - s["bicubic"] for s in record["validation_scenes"])
            assert abs(value - record["scene_mean_delta_db"]) < 1e-10
            values.append(value)
        mean, sd = statistics.mean(values), statistics.stdev(values)
        rows.append([latex(model), f"{mean:.4f}", f"{sd:.4f}"])
        tables.append({"table": "geographic_validation", "row": model, "source": geo_path,
                       "pointers": [f"/{i}" for i, _ in indexed],
                       "mean_scene_delta_db": mean, "sample_seed_sd_db": sd})
    output += table("Later development: mean paired PSNR gain over bicubic across four validation scenes, then three seeds. SD across seeds is not a scene confidence interval.",
                    ["Model", r"Mean $\Delta$PSNR (dB)", "Seed SD (dB)"], rows, "lrr")
    assert final["post_test_tuning"] is False
    for bands in (3, 4):
        rows = []
        for i, r in enumerate(final["summary"]):
            if r["bands"] != bands:
                continue
            assert r["scenes"] == 1 and r["tiles"] == 16
            rows.append([latex(r["model"]), f'{r["psnr"]:.4f}', f'{r["ssim"]:.4f}'])
            tables.append({"table": f"frozen_{bands}_band", "row": r["model"], "source": final_path,
                           "pointers": [f"/summary/{i}"], "psnr_db": r["psnr"], "ssim": r["ssim"]})
        output += table(f"Frozen test, {bands} bands: one scene / sixteen tiles. Compare only within this band protocol; official SwinIR has unmatched pretraining.",
                        ["Model", "PSNR (dB)", "SSIM"], rows, "lrr")
    interpretation = (
        f"The pilot learned gate averages {pilot_stats['hybrid_gate']:.4f} dB; the constant gate averages "
        f"{pilot_stats['hybrid_constant_gate']:.4f} dB, a difference of "
        f"{pilot_stats['hybrid_gate'] - pilot_stats['hybrid_constant_gate']:.4f} dB. "
        f"Both remain below the {bicubic:.4f} dB bicubic baseline. "
        "These are descriptive seed means on one validation scene, not evidence of geographic superiority.\n"
    )
    bib = (root / "references.bib").read_text(encoding="utf-8")
    sources["references.bib"] = {"sha256_utf8_lf": canonical_hash(bib)}
    manifest = {"schema_version": 1, "kind": "derived_tables_not_new_experiments",
                "sources": sources, "rows": tables,
                "frozen_protocol_hash": final["protocol_hash"], "freeze_commit": final["freeze_commit"],
                "limitations": ["No new inference or training", "No cross-sensor evidence",
                                "Seed SD is not geographic uncertainty", "Separate band protocols",
                                "Source dirty-worktree flags and checkpoint hashes remain in source records"]}
    return {"paper/generated/results.tex": output,
            "paper/generated/interpretation.tex": interpretation,
            "paper/references.bib": bib,
            "reports/paper-evidence.json": json.dumps(manifest, indent=2, allow_nan=False) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in build().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Stale generated file: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Paper evidence files verified" if args.check else "Paper evidence files generated")


if __name__ == "__main__":
    main()
