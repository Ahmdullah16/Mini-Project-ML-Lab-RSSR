"""Package reviewed defense outputs and their source snapshots outside Git."""
import argparse,hashlib,json,shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]

def package(output):
 output=Path(output).resolve()
 names=['RSSR_Defense_Presentation.pptx','RSSR_Defense_Presentation.pdf','RSSR_Examiner_Guide.docx','RSSR_Examiner_Guide.pdf','README.txt']
 for name in names:
  if not (output/name).is_file():raise ValueError('Required delivery file absent: '+name)
 paths=['scripts/build_defense_deck.mjs','scripts/finalize_defense_deck.mjs','scripts/build_defense_delivery.ps1','scripts/refresh_slide_chart_caches.py','scripts/package_defense_delivery.py','docs/examiner-defense-guide.md','docs/defense-materials.md','reports/final-heldout.json','reports/paper-evidence.json','reports/reference-audit-20261108.json','reports/reference-review-20261108.md','reports/defense-materials-20261108.json']
 for rel in paths:
  target=output/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
 files=[output/name for name in names]+[p for p in (output/'source').rglob('*') if p.is_file()]
 checks={p.relative_to(output).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
 (output/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
 archive=output.with_suffix('.zip')
 with ZipFile(archive,'w',ZIP_DEFLATED) as z:
  for p in files+[output/'checksums.json']:z.write(p,p.relative_to(output))
 with ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('Archive integrity failed')
 print(f'{archive}: {len(checks)} hashed files; ZIP integrity passed')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);package(**vars(p.parse_args()))
