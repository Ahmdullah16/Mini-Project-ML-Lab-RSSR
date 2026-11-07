"""Offline bibliography/primary-observation consistency check, not replication."""
import argparse,hashlib,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def entries(text):
    result={}
    for match in re.finditer(r'@\w+\s*\{\s*([^,]+),',text):
        key=match[1].strip()
        if key in result:raise ValueError('Duplicate bibliography key: '+key)
        i=match.end();fields={}
        while i<len(text):
            while i<len(text) and (text[i].isspace() or text[i]==','):i+=1
            if i>=len(text) or text[i]=='}':break
            field=re.match(r'(\w+)\s*=\s*',text[i:])
            if not field:raise ValueError('Malformed field in '+key)
            name=field[1].lower();i+=field.end()
            if text[i]=='{':
                start=i+1;depth=1;i+=1
                while i<len(text) and depth:
                    if text[i]=='{' and text[i-1]!='\\':depth+=1
                    if text[i]=='}' and text[i-1]!='\\':depth-=1
                    i+=1
                if depth:raise ValueError('Unclosed BibTeX value')
                value=text[start:i-1]
            else:
                end=i
                while end<len(text) and text[end] not in ',\n}':end+=1
                value=text[i:end].strip('" ');i=end
            fields[name]=value
        result[key]=fields
    return result

def norm(text):
    text=text.replace(r'\textmu','u').replace('µ','u').replace('μ','u').replace('ø','o').replace('Ł','L')
    text=unicodedata.normalize('NFKD',text)
    return re.sub(r'[^a-z0-9]','',text.lower())

def names(text):return [sorted(norm(w) for w in re.split(r'[\s,]+',name) if norm(w)) for name in text.split(' and ')]

def check(root=ROOT):
    bibliography=(root/'references.bib').read_text(encoding='utf-8')
    if bibliography!=(root/'paper/references.bib').read_text(encoding='utf-8'):raise ValueError('Bibliography copies differ')
    records=entries(bibliography);audit=json.loads((root/'reports/reference-audit-20261108.json').read_text(encoding='utf-8'))
    observations={r['key']:r for r in audit['records']}
    if len(observations)!=len(audit['records']) or set(records)!=set(observations):raise ValueError('Audit coverage differs from bibliography')
    for key,record in records.items():
        row=observations[key];observed=row['primary_metadata']
        if norm(record['title'])!=norm(observed['title']):raise ValueError('Primary title mismatch: '+key)
        if names(record['author'])!=names(observed['author']):raise ValueError('Primary author mismatch: '+key)
        if record['year']!=observed['year']:raise ValueError('Primary year mismatch: '+key)
        if not row['source_url'].startswith('https://') or not row['review_scope'] or not row['applicability']:raise ValueError('Missing source or scope: '+key)
    print(f'{len(records)} bibliography records match primary observations; metadata/concept scope only, not replication')
    return {'records':len(records),'bibliography_sha256':hashlib.sha256(bibliography.encode()).hexdigest()}

if __name__=='__main__':check()
