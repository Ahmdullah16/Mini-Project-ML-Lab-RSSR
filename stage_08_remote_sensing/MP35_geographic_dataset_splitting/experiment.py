"""Export existing split declarations after header-only geometry inspection."""
import hashlib,json,platform
from datetime import datetime,timezone
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds
from mp35_tools import audit_geography
ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'configs/naip-development-manifest.json'
    manifest=json.loads(source.read_text(encoding='utf-8'));scenes=[]
    for item in manifest['scenes']:
        path=ROOT/'data'/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Source checksum mismatch: '+item['id'])
        with rasterio.open(path) as raster:
            scene=dict(item,wgs84_bounds=list(transform_bounds(raster.crs,'EPSG:4326',*raster.bounds)),
                       source_crs=str(raster.crs),source_transform=list(raster.transform)[:6],
                       shape=[raster.count,raster.height,raster.width])
        scenes.append(scene)
    audit=audit_geography(scenes)
    destination=ROOT/'reports/mp35-splits';destination.mkdir(exist_ok=True)
    for split in ('train','validation','test'):
        payload={'original_assignments_preserved':True,'license':manifest['license'],
                 'scenes':[s for s in scenes if s['split']==split]}
        (destination/f'{split}.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    # A small vector map is a teaching artifact, not raster imagery or a basemap.
    colors={'train':'#1768ac','validation':'#ac5000','test':'#7824a5'}
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600" viewBox="0 0 900 600">',
         '<rect width="900" height="600" fill="white"/>',
         '<g font-family="sans-serif" font-size="14"><text x="30" y="30">MP35: existing scene centers (WGS84 longitude / latitude degrees)</text>',
         '<text x="30" y="55">Blue: train; orange: validation; purple: test. No metric-distance claim.</text>']
    for lon in range(-114,-98,2):
        x=70+(lon+114)*48
        svg.append(f'<path d="M{x},90 V530" stroke="#ddd"/><text x="{x}" y="555">{lon}</text>')
    for lat in range(32,46,2):
        y=530-(lat-31)*32
        svg.append(f'<path d="M70,{y} H840" stroke="#ddd"/><text x="30" y="{y}">{lat}</text>')
    for s in scenes:
        w,south,e,n=s['wgs84_bounds']; x=70+((w+e)/2+114)*48;y=530-((south+n)/2-31)*32
        svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[s["split"]]}"/><text x="{x+8}" y="{y-7}">{s["region_id"]}</text>')
    svg.append('</g></svg>');(destination/'map.svg').write_text('\n'.join(svg),encoding='utf-8')
    report={'recorded_at':datetime.now(timezone.utc).isoformat(),'manifest':source.relative_to(ROOT).as_posix(),
            'manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'audit':audit,
            'source_license':manifest['license'],'source_citation':manifest['citation'],
            'environment':{'python':platform.python_version(),'rasterio':rasterio.__version__},
            'scope':'Cached file checksums and raster headers only; no pixel reads, training or test metrics. Existing frozen assignments unchanged.',
            'limitations':'Envelope separation is not statistical independence. One test region cannot establish geographic generalization. Angular buffer varies in physical length with latitude.'}
    (ROOT/'reports/mp35-geographic-splits.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(audit['counts'],len(audit['cross_split_pairs']),'cross-split pairs passed')

if __name__=='__main__':main()
