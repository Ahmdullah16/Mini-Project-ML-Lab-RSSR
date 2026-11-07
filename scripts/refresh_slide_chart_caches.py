"""Refresh native chart number caches from embedded worksheet cells.

Only tiny serialization differences (<=1e-12) are accepted. Workbooks and all
non-chart parts are preserved; substantive changes require a new evidence review.
"""
import argparse,io,json,posixpath,re
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as ET
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
S='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R='http://schemas.openxmlformats.org/package/2006/relationships'

def refresh(source,target):
    source,target=Path(source),Path(target)
    if target.exists():raise ValueError('Output exists')
    changed={};points=0
    with ZipFile(source) as z:
        for name in z.namelist():
            if not re.fullmatch(r'ppt/charts/chart\d+.xml',name):continue
            relpath=posixpath.join(posixpath.dirname(name),'_rels',posixpath.basename(name)+'.rels')
            rels=ET.fromstring(z.read(relpath));packages=[r for r in rels if r.get('Type','').endswith('/package')]
            if len(packages)!=1:raise ValueError('One embedded workbook required')
            bookpath=posixpath.normpath(posixpath.join(posixpath.dirname(name),packages[0].get('Target')))
            with ZipFile(io.BytesIO(z.read(bookpath))) as book:
                wb=ET.fromstring(book.read('xl/workbook.xml'));sheets=wb.findall(f'{{{S}}}sheets/{{{S}}}sheet')
                if len(sheets)!=1:raise ValueError('Expected single literal data worksheet')
                ws=ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
                cells={c.get('r'):c.findtext(f'{{{S}}}v') for c in ws.iter(f'{{{S}}}c')}
            chart=ET.fromstring(z.read(name))
            for ref in chart.iter(f'{{{C}}}numRef'):
                formula=ref.findtext(f'{{{C}}}f');match=re.fullmatch(r"'?([^']+)'?!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)",formula or '')
                if not match or match[1]!=sheets[0].get('name') or match[2]!=match[4]:raise ValueError('Unsupported data range')
                cache=ref.find(f'{{{C}}}numCache');pts=cache.findall(f'{{{C}}}pt')
                if len(pts)!=int(match[5])-int(match[3])+1:raise ValueError('Range length mismatch')
                for pt in pts:
                    cell=match[2]+str(int(match[3])+int(pt.get('idx')));actual=cells.get(cell);v=pt.find(f'{{{C}}}v')
                    if actual is None or abs(Decimal(actual)-Decimal(v.text))>Decimal('1e-12'):raise ValueError('Substantive data mismatch: '+cell)
                    v.text=actual;points+=1
            changed[name]=ET.tostring(chart,encoding='utf-8',xml_declaration=True)
        with ZipFile(target,'w',ZIP_DEFLATED) as dest:
            for item in z.infolist():dest.writestr(item,changed.get(item.filename,z.read(item.filename)))
    print(json.dumps({'charts':len(changed),'points':points,'workbooks_unchanged':True,'max_accepted_serialization_difference':'1e-12'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('target');a=p.parse_args();refresh(a.source,a.target)
