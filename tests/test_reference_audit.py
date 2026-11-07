import importlib.util,json,shutil
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference_audit',ROOT/'scripts/check_reference_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def fixture(tmp_path):
 (tmp_path/'paper').mkdir();(tmp_path/'reports').mkdir()
 for rel in ['references.bib','paper/references.bib','reports/reference-audit-20261108.json']:shutil.copy2(ROOT/rel,tmp_path/rel)
 return tmp_path

def test_primary_metadata_and_all_records_are_covered():
 assert audit.check(ROOT)['records']==23

@pytest.mark.parametrize('fault',['year','author','coverage','duplicate'])
def test_corrupt_or_incomplete_reference_evidence_is_rejected(tmp_path,fault):
 root=fixture(tmp_path)
 if fault in ('year','author','coverage'):
  p=root/'reports/reference-audit-20261108.json';d=json.loads(p.read_text())
  if fault=='coverage':d['records'].pop()
  else:d['records'][0]['primary_metadata'][fault]='1900' if fault=='year' else 'Someone, Else'
  p.write_text(json.dumps(d))
 else:
  for rel in ['references.bib','paper/references.bib']:
   p=root/rel;p.write_text(p.read_text()+'\n@article{dong2014srcnn,title={Duplicate}}')
 with pytest.raises(ValueError):audit.check(root)

def test_nested_bibtex_and_author_tokenization():
 parsed=audit.entries('@article{x,title={A {nested} title},author={Mu Lee, Kyoung},year={2017}}')
 assert parsed['x']['title']=='A {nested} title'
 assert audit.names('Mu Lee, Kyoung')==audit.names('Lee, Kyoung Mu')
