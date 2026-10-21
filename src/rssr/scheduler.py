"""One bounded incremental tick. External schedulers invoke it; no daemon on import.

Acquisition-time overlap cannot discover arbitrarily late arrivals. Use periodic
explicit backfills for those; no universal provider availability filter is assumed.
"""
import json
import time
import uuid
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .core import config_hash
from .db import connect
from .ingest import refresh, request_json


def timestamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None: raise ValueError('Timezone-aware timestamps required')
    return result.astimezone(timezone.utc)


def tick(config,root,fetch=request_json,clock=time.time):
    """Acquire a lease, replay unfinished pages, advance only completed windows.

    Lease is two hours, above the bounded request retry/page budget. A crash can
    delay the next worker until expiry; a worker with an expired lease cannot
    commit the watermark. Query and reserved job ID are durable before network IO.
    """
    start=timestamp(config['start_time'])
    overlap=config.get('overlap_hours',48)
    if not isinstance(overlap,int) or not 0<=overlap<=24*31:
        raise ValueError('overlap_hours must be an integer in 0..744')
    query=dict(config['query'])
    if 'datetime' in query: raise ValueError('Scheduler owns the datetime interval')
    if query.get('max_pages',2) not in range(1,11) or query.get('limit',10) not in range(1,101):
        raise ValueError('Use bounded STAC page and item limits')
    current=clock(); end=datetime.fromtimestamp(current,timezone.utc)
    if start>end: raise ValueError('start_time is in the future')
    key=config_hash(config); token=uuid.uuid4().hex
    db=connect(Path(root)/'catalog.sqlite')
    db.execute('BEGIN IMMEDIATE')
    db.execute('INSERT OR IGNORE INTO ingestion_streams(id,configuration,updated_at) VALUES(?,?,?)',
               (key,json.dumps(config,sort_keys=True),end.isoformat()))
    row=db.execute('SELECT watermark,pending_query,active_job,lease_until FROM ingestion_streams WHERE id=?',(key,)).fetchone()
    if row[3] is not None and row[3]>current:
        db.rollback();db.close();return {'status':'leased','stream_id':key,'asset_downloads':0}
    watermark,pending,previous,_=row
    if pending: query=json.loads(pending)
    else:
        lower=max(start,timestamp(watermark)-timedelta(hours=overlap)) if watermark else start
        query['datetime']=lower.isoformat()+'/'+end.isoformat()
    # Reconcile a crash after refresh completed but before stream commit.
    old=db.execute('SELECT status,cursor_or_watermark FROM ingestion_jobs WHERE id=?',(previous,)).fetchone() if previous else None
    if old and old[0]=='completed':
        watermark=query['datetime'].split('/')[1]
        db.execute('UPDATE ingestion_streams SET watermark=?,pending_query=NULL,active_job=NULL,lease_token=NULL,lease_until=NULL,updated_at=? WHERE id=?',
                   (watermark,end.isoformat(),key));db.commit();db.close()
        return {'status':'recovered_completed','stream_id':key,'watermark':watermark,'asset_downloads':0}
    resume=previous if old and old[1] and old[0] in ('partial','failed','running') else None
    job=uuid.uuid4().hex
    db.execute('UPDATE ingestion_streams SET pending_query=?,active_job=?,lease_token=?,lease_until=?,updated_at=? WHERE id=?',
               (json.dumps(query),job,token,current+7200,end.isoformat(),key));db.commit();db.close()
    try:
        try: result=refresh(query,root,fetch,resume_job=resume,job_id=job)
        except urllib.error.HTTPError as error:
            if resume is None or error.code not in (400,404,410): raise
            # Expired cursor: replay the same frozen window, with scene deduplication.
            job=uuid.uuid4().hex
            db=connect(Path(root)/'catalog.sqlite')
            with db:
                changed=db.execute('UPDATE ingestion_streams SET active_job=? WHERE id=? AND lease_token=?',(job,key,token)).rowcount
            db.close()
            if changed!=1: raise RuntimeError('Scheduler lease lost')
            result=refresh(query,root,fetch,job_id=job)
        db=connect(Path(root)/'catalog.sqlite')
        with db:
            complete=result['status']=='completed'
            changed=db.execute('''UPDATE ingestion_streams SET watermark=?,pending_query=?,active_job=?,
                lease_token=NULL,lease_until=NULL,updated_at=? WHERE id=? AND lease_token=? AND lease_until>=?''',
                (query['datetime'].split('/')[1] if complete else watermark,None if complete else json.dumps(query),
                 None if complete else job,end.isoformat(),key,token,clock())).rowcount
        db.close()
        if changed!=1: raise RuntimeError('Scheduler lease expired before commit; replay required')
        return dict(result,stream_id=key,watermark=query['datetime'].split('/')[1] if complete else watermark)
    except Exception:
        db=connect(Path(root)/'catalog.sqlite')
        with db: db.execute('UPDATE ingestion_streams SET lease_token=NULL,lease_until=NULL WHERE id=? AND lease_token=?',(key,token))
        db.close();raise
