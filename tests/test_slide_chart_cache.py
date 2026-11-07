import importlib.util,io
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('chart_cache',ROOT/'scripts/refresh_slide_chart_caches.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path,cached):
 b=io.BytesIO()
 with ZipFile(b,'w') as z:
  z.writestr('xl/workbook.xml',f'<workbook xmlns="{m.S}"><sheets><sheet name="Chart Data"/></sheets></workbook>')
  z.writestr('xl/worksheets/sheet1.xml',f'<worksheet xmlns="{m.S}"><sheetData><row><c r="B2"><v>50.2614</v></c></row></sheetData></worksheet>')
 p=tmp_path/'source.pptx'
 with ZipFile(p,'w') as z:
  z.writestr('ppt/charts/chart1.xml',f'<c:chartSpace xmlns:c="{m.C}"><c:numRef><c:f>\'Chart Data\'!$B$2:$B$2</c:f><c:numCache><c:pt idx="0"><c:v>{cached}</c:v></c:pt></c:numCache></c:numRef></c:chartSpace>')
  z.writestr('ppt/charts/_rels/chart1.xml.rels',f'<Relationships xmlns="{m.R}"><Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package" Target="../embeddings/data.xlsx"/></Relationships>')
  z.writestr('ppt/embeddings/data.xlsx',b.getvalue());z.writestr('unchanged.txt','source')
 return p

def test_serialization_repair_preserves_workbook_and_source(tmp_path):
 p=fixture(tmp_path,'50.261400000000002');before=p.read_bytes();out=tmp_path/'fixed.pptx';m.refresh(p,out)
 assert p.read_bytes()==before
 with ZipFile(p) as a,ZipFile(out) as b:
  assert a.read('ppt/embeddings/data.xlsx')==b.read('ppt/embeddings/data.xlsx')
  assert a.read('unchanged.txt')==b.read('unchanged.txt')
  assert ET.fromstring(b.read('ppt/charts/chart1.xml')).findtext(f'.//{{{m.C}}}v')=='50.2614'

def test_substantive_data_drift_is_rejected_before_export(tmp_path):
 p=fixture(tmp_path,'51.0');out=tmp_path/'fixed.pptx'
 with pytest.raises(ValueError,match='Substantive'):m.refresh(p,out)
 assert not out.exists()

def test_existing_output_is_preserved(tmp_path):
 p=fixture(tmp_path,'50.2614');out=tmp_path/'fixed.pptx';out.write_bytes(b'user work')
 with pytest.raises(ValueError,match='exists'):m.refresh(p,out)
 assert out.read_bytes()==b'user work'
