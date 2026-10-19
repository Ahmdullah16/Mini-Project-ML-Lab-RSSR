"""Offline verification that durable fresh results survive moving their data root."""
import json
import shutil
from types import SimpleNamespace
import pytest
from rssr import fresh
from rssr.core import config_hash, digest
from rssr.db import connect


def test_moved_fresh_cache_and_integer_float_dedup(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.pt'; checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(fresh,'load_model',lambda p:(SimpleNamespace(bands=3),{'run_id':'fixture'}))
    item={'id':'item','properties':{'datetime':'2026-01-01T00:00:00Z'},
          'assets':{'visual':{'href':'https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif',
                              'eo:bands':[{'common_name':v} for v in ['red','green','blue']],
                              'raster:bands':[{'data_type':'uint8'}]*3}},
          'links':[{'rel':'license','href':'https://example.org/license'}]}
    root=tmp_path/'original'; root.mkdir()
    bbox=[0.,0.,1.,1.]
    identity=config_hash({'item':'item','asset':item['assets']['visual']['href'],'checkpoint':digest(checkpoint),'bbox':bbox,'size':256})
    folder=root/'fresh'/identity; folder.mkdir(parents=True)
    output=folder/'sr.tif'; output.write_bytes(b'checked result')
    relative=output.relative_to(root).as_posix()
    (folder/'provenance.json').write_text(json.dumps({'status':'completed','output_path':relative}))
    db=connect(root/'catalog.sqlite')
    with db:
        db.execute('INSERT INTO inference_jobs(id,provider_item_id,ingestion_time,status,output_path,output_sha256,checkpoint_sha256,provenance) VALUES(?,?,?,?,?,?,?,?)',
                   (identity,'item','2026-01-01T00:00:00Z','completed',relative,digest(output),digest(checkpoint),'{}'))
    db.close()
    moved=tmp_path/'moved'; shutil.move(root,moved)
    result=fresh.acquire_infer({'bbox':[0,0,1,1]},checkpoint,moved,fetch=lambda *a:{'features':[item]})
    assert result['cache_hit'] and fresh.stored_path(moved,result['output_path']).read_bytes()==b'checked result'
    fresh.stored_path(moved,result['output_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        fresh.acquire_infer({'bbox':bbox},checkpoint,moved,fetch=lambda *a:{'features':[item]})


def test_invalid_query_rejected_before_provider(tmp_path,monkeypatch):
    monkeypatch.setattr(fresh,'load_model',lambda p:(SimpleNamespace(bands=3),{}))
    def no_fetch(*a): raise AssertionError('Provider must not be contacted')
    for bbox in [[0,0,2,1],[0,0,0,1],[0,0,float('nan'),1]]:
        with pytest.raises(ValueError): fresh.acquire_infer({'bbox':bbox},'unused',tmp_path,fetch=no_fetch)
