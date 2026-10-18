import json
from pathlib import Path
import numpy as np
import torch
from .experiment import load_model
from .models import tiled
from .core import digest


def geotiff(checkpoint,source,target,divisor):
    import rasterio
    from affine import Affine
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Positive finite input divisor required")
    model,state=load_model(checkpoint)
    with rasterio.open(source) as ds:
        if ds.count != model.bands or ds.width*ds.height > 4_000_000:
            raise ValueError("Band mismatch or input exceeds 4 million pixels")
        x=ds.read().astype(np.float32)/divisor
        mask=(ds.read_masks()>0).all(0)&np.isfinite(x).all(0)
        x[:,~mask]=0
        profile=ds.profile.copy()
        profile.update(driver="GTiff",count=model.bands,dtype="float32",height=ds.height*model.scale,width=ds.width*model.scale,transform=ds.transform*Affine.scale(1/model.scale),nodata=float("nan"))
    y=tiled(model,torch.from_numpy(x)[None])[0].numpy()
    valid=np.repeat(np.repeat(mask,model.scale,0),model.scale,1); y[:,~valid]=np.nan
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(target,"w",**profile) as ds:
        ds.write(y); ds.update_tags(description="Model estimate; unsupported detail is possible",units="input units divided by divisor",checkpoint_sha256=digest(checkpoint))
    target.with_suffix(".json").write_text(json.dumps({"source_sha256":digest(source),"checkpoint_sha256":digest(checkpoint),"run_id":state["run_id"],"provenance":state["provenance"],"divisor":divisor,"scale":model.scale,"tiling":"128 LR pixels, 32 halo; channel attention may differ from whole-image inference"},indent=2))
    return target
