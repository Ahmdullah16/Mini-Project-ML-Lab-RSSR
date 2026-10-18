import argparse,json
from pathlib import Path
from rssr.fresh import acquire_infer
p=argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('checkpoint'); p.add_argument('--root',default='data'); a=p.parse_args()
print(json.dumps(acquire_infer(json.loads(Path(a.config).read_text()),a.checkpoint,a.root),indent=2))
