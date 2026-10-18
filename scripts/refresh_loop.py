"""Explicit opt-in bounded scheduler; no service is installed automatically."""
import argparse,json,time
from pathlib import Path
from rssr.ingest import refresh
p=argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('--root',default='data'); p.add_argument('--cycles',type=int,default=1)
a=p.parse_args(); c=json.loads(Path(a.config).read_text())
if not 1<=a.cycles<=30: p.error('cycles must be 1..30')
interval=c.get('refresh_interval_hours',24)
if interval<1: p.error('refresh interval must be at least one hour')
for i in range(a.cycles):
    try: print(refresh(c,a.root),flush=True)
    except Exception as e: print(type(e).__name__,str(e),flush=True)
    if i+1<a.cycles: time.sleep(interval*3600)
