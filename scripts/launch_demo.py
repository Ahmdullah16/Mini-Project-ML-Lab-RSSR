"""Launch from a portable quickstart bundle without retraining or downloading."""
import argparse
import json
import os
import runpy
from pathlib import Path
import sys
from rssr.core import digest

p=argparse.ArgumentParser()
p.add_argument('--data-root',default='data/submission-pilot')
p.add_argument('--host',default=os.environ.get('RSSR_HOST','127.0.0.1'))
p.add_argument('--port',type=int,default=int(os.environ.get('RSSR_PORT','7860')))
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
data=Path(a.data_root).resolve()
record=json.loads((data/'demo.json').read_text())
checkpoint=data/record['checkpoint']
if digest(checkpoint)!=record['checkpoint_sha256']:
    raise ValueError('Checkpoint checksum mismatch')
os.environ.update(RSSR_CHECKPOINT=str(checkpoint),RSSR_DATA_ROOT=str(data),
                  RSSR_CATALOG=str(data/'catalog.sqlite'),RSSR_HOST=a.host,RSSR_PORT=str(a.port))
sys.path.insert(0,str(root/'app'))
runpy.run_path(str(root/'app/demo.py'),run_name='__main__')
