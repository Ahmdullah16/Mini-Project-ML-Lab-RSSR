"""Bounded CPU training with deterministic resume and traceable measurements."""
import json
import platform
import subprocess
import time
import uuid
import copy
from importlib.metadata import distributions
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import config_hash, digest, metrics, resize, degrade
from .models import SRModel
from .db import connect
from .data import now


def load_tiles(manifest, split, base=None):
    rows = [r for r in manifest["tiles"] if r["split"] == split]
    if not rows:
        raise ValueError(f"No {split} tiles")
    result=[]
    for row in rows:
        path=Path(row['path'])
        if not path.is_absolute():
            if base is None:
                raise ValueError('Relative tile paths require the prepared manifest directory')
            path=Path(base)/path
        if digest(path) != row["sha256"]:
            raise ValueError("Prepared tile checksum mismatch")
        with np.load(path,allow_pickle=False) as d:
            result.append((row,torch.from_numpy(d["lr"]),torch.from_numpy(d["hr"])))
    return result


def save_checkpoint(path, state):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp"); torch.save(state,tmp); tmp.replace(path)


def load_model(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    model=SRModel(**state["config"]["model"])
    model.load_state_dict(state["model"]); model.eval()
    return model,state


def train(config, manifest_path, root, resume=None):
    if config["steps"] < 1 or config["steps"] > 10000 or not 1 <= config["batch_size"] <= 32:
        raise ValueError("Bounded run requires 1..10000 steps and batch size 1..32")
    torch.set_num_threads(2); torch.manual_seed(config["seed"]); torch.use_deterministic_algorithms(True)
    root=Path(root); manifest=json.loads(Path(manifest_path).read_text()); mh=digest(manifest_path)
    weight=config.get('consistency_weight',0)
    if not np.isfinite(weight) or weight<0:
        raise ValueError('Consistency weight must be finite and nonnegative')
    if weight and manifest.get('protocol')!='synthetic-naip-v1':
        raise ValueError('Consistency operator is only justified for synthetic-naip-v1')
    if config["model"]["bands"] != len(manifest["bands"]) or config["model"]["scale"] != manifest["scale"]:
        raise ValueError("Model and dataset bands/scale differ")
    training,validation=load_tiles(manifest,"train",Path(manifest_path).parent),load_tiles(manifest,"validation",Path(manifest_path).parent)
    model=SRModel(**config["model"]); optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    generator=torch.Generator().manual_seed(config["seed"])
    step=0; best=float("inf"); best_model=None; best_step=0
    rid=uuid.uuid4().hex; out=root/"runs"/rid; out.mkdir(parents=True)
    if resume:
        state=torch.load(resume,map_location="cpu",weights_only=True)
        if state.get("inference_only"):
            raise ValueError("Resume from last.pt, not selected inference checkpoint best.pt")
        old=dict(state["config"]); new=dict(config); old.pop("steps"); new.pop("steps")
        if old != new or state["manifest_hash"] != mh:
            raise ValueError("Resume configuration or dataset mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"]); torch.set_rng_state(state["rng"])
        step=state["step"]; best=state["best"]
        best_model=state["best_model"]; best_step=state["best_step"]
        if step >= config["steps"]:
            raise ValueError("Resume target must exceed saved step")
    git=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or "uncommitted"
    dirty=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True).stdout.strip())
    environment={d.metadata["Name"]:d.version for d in distributions() if d.metadata["Name"]}
    hardware={"platform":platform.platform(),"processor":platform.processor(),"device":"cpu","threads":2,"gpu_hours":0}
    db=connect(root/"catalog.sqlite")
    with db:
        db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(rid,git+("+dirty" if dirty else ""),mh,config_hash(config),config["model"]["kind"],config["seed"],json.dumps(environment),json.dumps(hardware),now(),None,"running",None,None))
    started=time.perf_counter(); history=[]
    try:
        while step < config["steps"]:
            model.train()
            ids=torch.randint(len(training),(config["batch_size"],),generator=generator).tolist()
            lr=torch.stack([training[i][1] for i in ids]); hr=torch.stack([training[i][2] for i in ids])
            optimizer.zero_grad(set_to_none=True); pred=model(lr)
            reconstruction=F.l1_loss(pred,hr); loss=reconstruction
            if config.get("consistency_weight",0):
                # This operator is justified ONLY for the recorded synthetic degradation.
                loss=loss+config["consistency_weight"]*F.l1_loss(torch.stack([degrade(p,manifest["scale"]) for p in pred]),lr)
            loss.backward(); optimizer.step(); step+=1
            if step % config["validate_every"] == 0 or step == config["steps"]:
                model.eval()
                with torch.inference_mode():
                    val=float(np.mean([F.l1_loss(model(x[None])[0],y).item() for _,x,y in validation]))
                improved=val < best; best=min(best,val)
                if improved:
                    best_model=copy.deepcopy(model.state_dict()); best_step=step
                state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":config,"step":step,"best":best,"best_model":best_model,"best_step":best_step,"generator":generator.get_state(),"rng":torch.get_rng_state(),"manifest_hash":mh,"run_id":rid,"provenance":"pilot synthetic degradation on NAIP; not a research-validated model"}
                save_checkpoint(out/"last.pt",state)
                history.append({"step":step,"train_l1":reconstruction.item(),"train_total_loss":loss.item(),"validation_l1":val})
        selected=dict(state,model=best_model,selected_step=best_step,inference_only=True)
        save_checkpoint(out/"best.pt",selected)
        report={"run_id":rid,"config":config,"git_commit":git,"dirty":dirty,"manifest_hash":mh,"environment":environment,"hardware":hardware,"seconds":time.perf_counter()-started,"parameters":sum(p.numel() for p in model.parameters()),"history":history,"resume_from":str(resume) if resume else None}
        (out/"training.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        with db:
            db.execute("UPDATE runs SET status='completed',end_time=?,checkpoint_path=?,checkpoint_hash=? WHERE id=?",(now(),str(out/"best.pt"),digest(out/"best.pt"),rid))
            db.execute("INSERT INTO artifacts VALUES(?,?,?,?)",(rid,"training-record",str(out/"training.json"),digest(out/"training.json")))
        return out
    except BaseException as exc:
        with db:
            db.execute("UPDATE runs SET status=?,end_time=? WHERE id=?",("interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",now(),rid))
        raise
    finally:
        db.close()


def evaluate(checkpoint, manifest_path, root, split="validation"):
    model,state=load_model(checkpoint); manifest=json.loads(Path(manifest_path).read_text())
    if digest(manifest_path) != state["manifest_hash"]:
        raise ValueError("Evaluation dataset mismatch")
    rows=load_tiles(manifest,split,Path(manifest_path).parent); output=[]
    torch.set_num_threads(2)
    for row,x,hr in rows:
        with torch.inference_mode():
            predictions={m:resize(x,model.scale,m) for m in ("nearest","bilinear","bicubic")}
            predictions["model"]=model(x[None])[0]
        for name,pred in predictions.items():
            result=metrics(pred,hr,border=model.scale)
            output.append({"tile_id":row["id"],"scene_id":row["scene_id"],"method":name,**result})
    db=connect(Path(root)/"catalog.sqlite")
    with db:
        for row in output:
            for metric,value in row.items():
                if metric in ("tile_id","scene_id","method"):
                    continue
                db.execute("INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?,?)",(state["run_id"],row["tile_id"],row["method"]+"/"+metric,value,"dB" if metric=="psnr" else "rad" if metric=="sam_rad" else "normalized DN","synthetic-naip-v1",split))
    target=Path(checkpoint).parent/f"evaluation-{split}.json"
    target.write_text(json.dumps(output,indent=2),encoding="utf-8"); db.close()
    return target
