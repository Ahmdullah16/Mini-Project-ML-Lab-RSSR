import json
import urllib.error
import pytest
from rssr.db import connect
from rssr.scheduler import tick

CONFIG={'start_time':'2024-01-01T00:00:00Z','overlap_hours':24,
        'query':{'endpoint':'https://example.invalid','limit':1,'max_pages':1}}
NOW=1704326400.0 # 2024-01-04 UTC

def item(number):
    return {'id':str(number),'collection':'s2','properties':{'datetime':'2024-01-02T00:00:00Z'},'assets':{}}

def state(root):
    db=connect(root/'catalog.sqlite')
    row=db.execute('SELECT watermark,pending_query,active_job,lease_token,lease_until FROM ingestion_streams').fetchone()
    count=db.execute('SELECT count(*) FROM scenes').fetchone()[0];db.close()
    return row,count

def test_partial_completion_overlap_and_deduplication(tmp_path):
    def first(url,body):
        assert body['datetime'].startswith('2024-01-01')
        return {'features':[item(1)],'links':[{'rel':'next','href':'https://example.invalid/page2'}]}
    assert tick(CONFIG,tmp_path,first,lambda:NOW)['status']=='partial'
    assert state(tmp_path)[0][0] is None
    def second(url,body):
        assert url.endswith('page2') and body is None
        return {'features':[item(1),item(2)],'links':[]}
    assert tick(CONFIG,tmp_path,second,lambda:NOW+60)['items_added']==1
    assert state(tmp_path)[0][0].startswith('2024-01-04')
    def later(url,body):
        assert body['datetime'].startswith('2024-01-03')
        return {'features':[item(2)],'links':[]}
    assert tick(CONFIG,tmp_path,later,lambda:NOW+86400)['items_added']==0
    assert state(tmp_path)[1]==2

def test_crash_lease_expiry_and_completed_reconciliation(tmp_path):
    def crash(url,body): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): tick(CONFIG,tmp_path,crash,lambda:NOW)
    assert tick(CONFIG,tmp_path,crash,lambda:NOW+1)['status']=='leased'
    def finish(url,body):return {'features':[item(1)],'links':[]}
    result=tick(CONFIG,tmp_path,finish,lambda:NOW+7201)
    assert result['status']=='completed'
    # Simulate process loss after refresh commit, before stream acknowledgement.
    db=connect(tmp_path/'catalog.sqlite')
    with db:
        db.execute('UPDATE ingestion_streams SET watermark=NULL,pending_query=?,active_job=?,lease_until=0',
                   (json.dumps(dict(CONFIG['query'],datetime='2024-01-01T00:00:00Z/2024-01-04T00:00:00Z')),result['job_id']))
    db.close()
    assert tick(CONFIG,tmp_path,crash,lambda:NOW+8000)['status']=='recovered_completed'
    assert state(tmp_path)[1]==1

def test_expired_cursor_restarts_frozen_window(tmp_path):
    def first(url,body): return {'features':[item(1)],'links':[{'rel':'next','href':'https://example.invalid/expired'}]}
    tick(CONFIG,tmp_path,first,lambda:NOW)
    calls=[]
    def expire(url,body):
        calls.append(url)
        if url.endswith('expired'):raise urllib.error.HTTPError(url,410,'Expired',{},None)
        return {'features':[item(1),item(2)],'links':[]}
    result=tick(CONFIG,tmp_path,expire,lambda:NOW+1)
    assert result['status']=='completed' and result['items_added']==1
    assert len(calls)==2 and state(tmp_path)[1]==2

def test_failure_never_advances_and_configuration_rejected(tmp_path):
    def fail(url,body):raise OSError('offline')
    with pytest.raises(OSError):tick(CONFIG,tmp_path,fail,lambda:NOW)
    row,_=state(tmp_path);assert row[0] is None and row[1] and row[3] is None
    with pytest.raises(ValueError):tick(dict(CONFIG,start_time='2024-01-01'),tmp_path,fail,lambda:NOW)
