"""Portable small real-data pipeline. All generated assets stay under --data-root."""
import argparse,json,os
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rssr.data import download,prepare
from rssr.experiment import train,evaluate
from rssr.inference import geotiff
from rssr.core import digest
from rssr.db import connect
p=argparse.ArgumentParser(); p.add_argument('--data-root',default='data/submission-pilot'); p.add_argument('--steps',type=int,default=100); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; os.chdir(root)
data=Path(a.data_root).resolve(); data.mkdir(parents=True,exist_ok=True)
source=json.loads((root/'configs/naip-pilot-manifest.json').read_text())
transferred=0
for scene in source['scenes']: transferred+=download(scene['url'],data/scene['path'],scene['sha256'])
source.update(name=source['name']+' RGB display pilot',bands=source['bands'][:3],band_indices=[0,1,2])
manifest=data/'rgb-source-manifest.json'; manifest.write_text(json.dumps(source,indent=2))
prepared=prepare(manifest,data,limit=16)
config=json.loads((root/'configs/pilot.json').read_text()); config.update(steps=a.steps,validate_every=min(20,a.steps)); config['model'].update(kind='cnn',bands=3)
run=train(config,prepared,data); evaluation=evaluate(run/'best.pt',prepared,data)
scene=next(s for s in source['scenes'] if s['split']=='train')
with rasterio.open(data/scene['path']) as ds:
    window=Window(0,0,64,64); profile=ds.profile.copy(); profile.update(width=64,height=64,count=3,transform=ds.window_transform(window))
    with rasterio.open(data/'demo-input.tif','w',**profile) as out: out.write(ds.read([1,2,3],window=window))
sr=geotiff(run/'best.pt',data/'demo-input.tif',data/'demo-output.tif',source['divisor'])
with rasterio.open(sr) as out: assert out.shape==(128,128) and out.count==3 and out.crs
record={'dataset_bytes_transferred':transferred,'steps':a.steps,'checkpoint':str((run/'best.pt').relative_to(data)),'checkpoint_sha256':digest(run/'best.pt'),'prepared_manifest':str(prepared.relative_to(data)),'evaluation':str(evaluation.relative_to(data)),'output':'demo-output.tif','output_sha256':digest(sr),'status':'executed engineering pilot; no final research claim'}
(data/'demo.json').write_text(json.dumps(record,indent=2))
db=connect(data/'catalog.sqlite'); assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'; db.close()
print(json.dumps(record,indent=2))
