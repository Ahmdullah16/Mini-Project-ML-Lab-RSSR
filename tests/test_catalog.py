from datetime import datetime,timezone
import pytest
from rssr.db import connect
from rssr.catalog import snapshot


def test_snapshot_distinguishes_partial_from_complete(tmp_path):
    path=tmp_path/'catalog with spaces.sqlite'; db=connect(path)
    with db:
        db.execute("INSERT INTO ingestion_jobs(id,provider,query_config,start_time,end_time,status) VALUES('full','p','{}','2026-10-18T10:00:00+00:00','2026-10-18T11:00:00+00:00','completed')")
        db.execute("INSERT INTO ingestion_jobs(id,provider,query_config,start_time,end_time,status) VALUES('partial','p','{}','2026-10-18T12:00:00+00:00','2026-10-18T12:01:00+00:00','partial')")
    db.close(); before=path.read_bytes()
    result=snapshot(path,now=datetime(2026,10,18,13,tzinfo=timezone.utc))
    assert result['latest_job']['status']=='partial'
    assert result['seconds_since_complete_refresh']==7200
    assert result['last_complete_refresh']=='2026-10-18T11:00:00+00:00'
    assert path.read_bytes()==before


def test_snapshot_missing_and_empty(tmp_path):
    p=tmp_path/'absent.sqlite'
    with pytest.raises(FileNotFoundError): snapshot(p)
    assert not p.exists()
    db=connect(p); db.close()
    assert snapshot(p)['last_complete_refresh'] is None
    with pytest.raises(ValueError): snapshot(p,limit=101)
