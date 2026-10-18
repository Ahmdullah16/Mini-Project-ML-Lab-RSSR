"""Explicit, bounded data access. No downloads at import or notebook execution."""
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from .core import digest, config_hash, degrade
from .db import connect


def now():
    return datetime.now(timezone.utc).isoformat()


def download(url, destination, expected_sha256, max_bytes=20_000_000):
    """Resume a pinned asset; verify before atomic replacement; bounded retries."""
    destination = Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if digest(destination) != expected_sha256:
            raise ValueError("Existing file checksum differs; refusing silent replacement")
        return 0
    partial = destination.with_suffix(destination.suffix+".part")
    transferred = 0
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            req = urllib.request.Request(url,headers={"Range":f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(req,timeout=45) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resume range")
                total = offset if append else 0
                with partial.open("ab" if append else "wb") as f:
                    while chunk := response.read(65536):
                        total += len(chunk); transferred += len(chunk)
                        if total > max_bytes or transferred > max_bytes*2:
                            raise ValueError("Download byte budget exceeded")
                        f.write(chunk)
            if digest(partial) != expected_sha256:
                raise ValueError("Downloaded checksum mismatch")
            partial.replace(destination)
            return transferred
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def read_raster(path):
    import rasterio
    with rasterio.open(path) as ds:
        x = ds.read().astype(np.float32)
        mask = (ds.read_masks() > 0).all(0) & np.isfinite(x).all(0)
        meta = {"crs":str(ds.crs),"transform":list(ds.transform),"bounds":list(ds.bounds),"nodata":ds.nodata,"bands":list(ds.descriptions),"dtype":ds.dtypes[0],"shape":list(x.shape)}
    return x,mask,meta


def check_splits(scenes, buffer_degrees=.02):
    """Conservative WGS84 bounding-box buffer, NOT a fixed metric distance.

    Research regions must be declared externally; rejects repeat region IDs and
    neighboring boxes across splits, irrespective of acquisition date.
    """
    for i,a in enumerate(scenes):
        if a["split"] not in ("train","validation","test"):
            raise ValueError("Invalid split")
        for b in scenes[i+1:]:
            if a["split"] == b["split"]:
                continue
            if a["region_id"] == b["region_id"]:
                raise ValueError("Region leakage")
            A,B = a["wgs84_bounds"],b["wgs84_bounds"]
            if not (A[2]+buffer_degrees < B[0] or B[2]+buffer_degrees < A[0] or A[3]+buffer_degrees < B[1] or B[3]+buffer_degrees < A[1]):
                raise ValueError("Geographic overlap or buffer violation")


def prepare(manifest_path, root, scale=2, patch=32, limit=16):
    """Prepare nonoverlapping training fixtures AFTER geographic split validation."""
    import rasterio.warp
    root = Path(root); manifest = json.loads(Path(manifest_path).read_text())
    if patch < 8 or patch % scale or limit < 1:
        raise ValueError("Invalid patch/limit")
    scenes = []
    for item in manifest["scenes"]:
        x,mask,meta = read_raster(root/item["path"])
        indices=manifest.get("band_indices",list(range(x.shape[0])))
        if len(set(indices))!=len(indices) or any(i<0 or i>=x.shape[0] for i in indices):
            raise ValueError("Invalid explicit band selection")
        x=x[indices]
        if x.shape[0] != len(manifest["bands"]) or (x[:,mask] < 0).any() or (x[:,mask] > manifest["divisor"]).any():
            raise ValueError("Unexpected bands or radiometric range")
        item = dict(item, **meta)
        item["wgs84_bounds"] = list(rasterio.warp.transform_bounds(meta["crs"],"EPSG:4326",*meta["bounds"]))
        scenes.append(item)
    check_splits(scenes)
    db = connect(root/"catalog.sqlite")
    mh = digest(manifest_path)
    with db:
        db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?,?,?,?,?,?)",(mh,manifest["name"],manifest["version"],manifest["source_url"],manifest["license"],manifest["citation"],mh))
    out = root/"prepared"/f"{mh[:12]}-x{scale}-p{patch}-n{limit}"
    out.mkdir(parents=True,exist_ok=True)
    tiles=[]
    for item in scenes:
        source = root/item["path"]
        if digest(source) != item["sha256"]:
            raise ValueError("Source integrity failure")
        x,mask,_ = read_raster(source)
        x=x[manifest.get("band_indices",list(range(x.shape[0])))]; x = torch.from_numpy(x/manifest["divisor"])
        sid = config_hash({"manifest":mh,"scene":item["id"]})
        with db:
            db.execute("INSERT OR IGNORE INTO scenes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(sid,mh,item["id"],item["region_id"],manifest["sensor"],item.get("acquisition_time"),None,manifest["retrieval_time"],item["crs"],json.dumps(item["transform"]),json.dumps(item["bounds"]),json.dumps(manifest["bands"]),manifest["units"],json.dumps(item["nodata"]),"valid mask required",str(source),item["sha256"]))
        count=0
        for y in range(0,x.shape[1]-patch+1,patch):
            for z in range(0,x.shape[2]-patch+1,patch):
                if count >= limit or not mask[y:y+patch,z:z+patch].all():
                    continue
                hr=x[:,y:y+patch,z:z+patch]; cfg={"scale":scale,"sigma":.8,"noise":0.,"seed":0}
                lr=degrade(hr,**cfg); tid=config_hash({"scene":sid,"y":y,"x":z,"patch":patch,"degradation":cfg})
                path=out/f"{tid}.npz"
                if not path.exists():
                    np.savez(path,lr=lr.numpy(),hr=hr.numpy())
                record={"id":tid,"scene_id":sid,"split":item["split"],"path":str(path.resolve()),"sha256":digest(path)}
                tiles.append(record)
                with db:
                    db.execute("INSERT OR IGNORE INTO pairs_or_tiles VALUES(?,?,?,?,?,?,?,?,?)",(tid,item["split"],str(path),str(path),scale,json.dumps([z,y,patch,patch]),"synthetic, same HR grid",json.dumps(cfg),record["sha256"]))
                    db.execute("INSERT OR IGNORE INTO pair_sources VALUES(?,?)",(tid,sid))
                count+=1
        if count == 0:
            raise ValueError("No valid patches in scene")
    prepared={"source_manifest_sha256":mh,"protocol":"synthetic-naip-v1","bands":manifest["bands"],"scale":scale,"scenes":scenes,"tiles":tiles}
    target=out/"manifest.json"; target.write_text(json.dumps(prepared,indent=2),encoding="utf-8")
    db.close()
    return target
