"""Package reviewed report artifacts; preserve generated files outside Git."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[1]

def package(output, ieee_pdf):
    output=Path(output).resolve();ieee_pdf=Path(ieee_pdf).resolve()
    output.mkdir(parents=True,exist_ok=True)
    required=['RSSR_Final_Project_Report.pdf','RSSR_Detailed_Report_Editable.docx','RSSR_IEEE_Paper_Editable.docx','RSSR_Parameters.csv','report-source-manifest.json']
    for name in required:
        if not (output/name).is_file(): raise ValueError('Required artifact absent: '+name)
    with ZipFile(output/'RSSR_Detailed_Report_Editable.docx') as doc:
        images=[n for n in doc.namelist() if n.startswith('word/media/')]
        expected=14 if (output/'comparison-evidence.json').exists() else 12
        if len(images)!=expected: raise ValueError(f'Expected {expected} embedded report images')
    shutil.copy2(ieee_pdf,output/'RSSR_IEEE_Research_Paper.pdf')
    source=output/'source'
    for path in list((ROOT/'paper').rglob('*.tex'))+list((ROOT/'paper').rglob('*.bib'))+list((ROOT/'paper').rglob('*.cls')):
        target=source/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for rel in ['references.bib','scripts/build_project_report.py','scripts/build_seven_model_panels.py','scripts/export_editable_reports.py','scripts/generate_technical_architectures.py','scripts/package_report_delivery.py','requirements/report.txt','src/rssr/models.py','src/rssr/architecture_registry.py','docs/report-build.md','reports/paper-evidence.json','reports/reference-audit-20261108.json','reports/reference-review-20261108.md','scripts/check_reference_audit.py']:
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    (output/'README.txt').write_text('RSSR report delivery — 8 November 2026\n\nDetailed report: RSSR_Final_Project_Report.pdf.\nEditable report: RSSR_Detailed_Report_Editable.docx (text/equations/tables editable; diagram and comparison images embedded).\nIEEE publication PDF: RSSR_IEEE_Research_Paper.pdf, reused from the previously reviewed build; source retained under source/paper/.\nIEEE editing copy: RSSR_IEEE_Paper_Editable.docx (single-column editing layout).\nFull-size diagrams: figures/RSSR_Technical_Block_Diagrams.pdf; PNG and editable SVG alongside.\nParameters CSV describes untrained RGB registry configurations, not loaded checkpoint counts or performance.\n\nEvidence scope: recorded experimental tables preserved. No new training or frozen-test evaluation was performed. If comparison-evidence.json is present, it records the bounded training-example display inference executed for the added comparison figures. Twenty-three citation records have primary-source identity/concept observations in the audit; full per-paper experimental extraction and reproduction remain unfinished.\n\nOutstanding: slides, source-bound visual model comparisons, full per-paper experimental extraction, geographic/cross-sensor scientific gates and renewed live deployment/restart checks. This is a reviewed document delivery, not a claim that all project acceptance gates pass.\n',encoding='utf-8')
    files=[p for p in output.rglob('*') if p.is_file() and '-review-' not in p.name and p.suffix not in ('.log','.blg','.zip') and p.name!='delivery-checksums.json']
    manifest={str(p.relative_to(output)).replace('\\','/'):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
    (output/'delivery-checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=output.with_name(output.name+'.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in files+[output/'delivery-checksums.json']:z.write(p,str(p.relative_to(output)))
    with ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive integrity failed')
    print(f'{archive}: {len(manifest)} hashed files; {len(images)} DOCX images; ZIP integrity passed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--ieee-pdf',required=True);package(**vars(p.parse_args()))
