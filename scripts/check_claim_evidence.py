"""Verify claim links and canonical text hashes without generating research results."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
record=json.loads((ROOT/'docs/claim_evidence.json').read_text(encoding='utf-8'))
ids=set()
for claim in record['claims']:
    assert claim['id'] not in ids, 'Duplicate claim ID'
    ids.add(claim['id'])
    assert claim['status'] in ('measured','locally-tested','planned','unavailable')
    source=(ROOT/claim['source']).resolve()
    assert source.is_relative_to(ROOT) and source.is_file(),claim['source']
    digest=hashlib.sha256(source.read_text(encoding='utf-8').replace('\r\n','\n').encode('utf-8')).hexdigest()
    assert digest==claim['sha256_utf8_lf'], 'Evidence changed: '+claim['id']
    assert claim['limitation'].strip()
print(len(ids),'claim sources/hash/status/limitation records verified; no scientific claims inferred')
