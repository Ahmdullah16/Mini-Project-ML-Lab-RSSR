"""Read-only catalog snapshots for offline demo freshness reporting."""
from datetime import datetime,timezone
from pathlib import Path
import sqlite3


def snapshot(path,limit=20,now=None):
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=100:
        raise ValueError('Snapshot limit must be 1..100')
    path=Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError('Catalog does not exist; run an explicit refresh first')
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None: raise ValueError('Snapshot time must be timezone-aware')
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        if [r[0] for r in db.execute('SELECT version FROM schema_versions')] not in ([1],[2]):
            raise ValueError('Unsupported catalog schema')
        latest=db.execute('SELECT id,start_time,end_time,status,items_found,items_added FROM ingestion_jobs ORDER BY start_time DESC,id DESC LIMIT 1').fetchone()
        complete=db.execute("SELECT end_time FROM ingestion_jobs WHERE status='completed' AND end_time IS NOT NULL ORDER BY end_time DESC LIMIT 1").fetchone()
        observations=[dict(r) for r in db.execute("SELECT provider_item_id,sensor,acquisition_time,availability_time,ingestion_time,quality_flags FROM scenes WHERE dataset_id LIKE 'stac:%' ORDER BY ingestion_time DESC,id DESC LIMIT ?",(limit,))]
        age=None
        if complete:
            age=max(0.,(now-datetime.fromisoformat(complete[0].replace('Z','+00:00'))).total_seconds())
        return {'mode':'cached metadata; no live sensing or automatic asset download','snapshot_time':now.isoformat(),'last_complete_refresh':complete[0] if complete else None,'seconds_since_complete_refresh':age,'latest_job':dict(latest) if latest else None,'observations':observations,'observation_limit':limit,'reference_metrics':'unavailable without compatible HR reference'}
    finally:
        db.close()
