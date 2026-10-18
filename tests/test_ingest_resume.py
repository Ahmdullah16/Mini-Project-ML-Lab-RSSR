import json
import pytest
from rssr.ingest import refresh
from rssr.db import connect


def item(number):
    return {'id':str(number),'collection':'s2','properties':{'datetime':'2024-01-01T00:00:00Z'},'bbox':[1,2,3,4],'assets':{}}


def test_partial_resume_and_post_merge(tmp_path):
    config={'endpoint':'https://example.invalid','limit':1,'max_pages':1}
    def first(url,body):
        assert body=={'limit':1}
        return {'features':[item(1)],'links':[{'rel':'next','href':url,'method':'POST','merge':True,'body':{'token':'page2'}}]}
    a=refresh(config,tmp_path,first)
    assert a['status']=='partial'
    def second(url,body):
        assert body=={'limit':1,'token':'page2'}
        return {'features':[item(1),item(2)],'links':[]}
    b=refresh(config,tmp_path,second,resume_job=a['job_id'])
    assert b['status']=='completed' and b['items_added']==1 and b['items_found']==2
    with pytest.raises(ValueError): refresh(config,tmp_path,second,resume_job=b['job_id'])
    with pytest.raises(ValueError): refresh({**config,'limit':2},tmp_path,second,resume_job=a['job_id'])


def test_failure_cursor_replays_page_without_duplicate(tmp_path):
    config={'endpoint':'https://example.invalid','limit':1,'max_pages':2}
    def fail_second(url,body):
        if url.endswith('/page2'): raise OSError('temporary outage')
        return {'features':[item(1)],'links':[{'rel':'next','href':'https://example.invalid/page2'}]}
    with pytest.raises(OSError): refresh(config,tmp_path,fail_second)
    db=connect(tmp_path/'catalog.sqlite')
    row=db.execute("SELECT id,cursor_or_watermark FROM ingestion_jobs WHERE status='failed'").fetchone()
    assert json.loads(row[1])['href'].endswith('/page2')
    db.close()
    def finish(url,body):
        assert url.endswith('/page2') and body is None
        return {'features':[item(1),item(2)],'links':[]}
    assert refresh(config,tmp_path,finish,resume_job=row[0])['items_added']==1
    db=connect(tmp_path/'catalog.sqlite')
    assert db.execute('SELECT count(*) FROM scenes').fetchone()[0]==2
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    db.close()
