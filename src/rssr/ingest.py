"""Bounded STAC metadata ingestion, separate from training and asset downloads."""
import json
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path
from .core import config_hash
from .data import now
from .db import connect


def request_json(url,body=None):
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,data=json.dumps(body).encode() if body else None,headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429,500,502,503,504) or attempt==2:
                raise
            delay=e.headers.get("Retry-After","")
            time.sleep(min(30,float(delay)) if delay.isdigit() else 2**attempt)
        except OSError:
            if attempt==2:
                raise
            time.sleep(2**attempt)


def refresh(config,root,fetch=request_json):
    # Queries overlap intentionally: provider IDs make repeats idempotent. A frozen
    # research manifest never reads this catalog automatically.
    if config.get("max_pages",2) not in range(1,11):
        raise ValueError("max_pages must be 1..10")
    if config.get("limit",10) not in range(1,101):
        raise ValueError("limit must be 1..100")
    root=Path(root); root.mkdir(parents=True,exist_ok=True); db=connect(root/"catalog.sqlite")
    jid=uuid.uuid4().hex; dataset="stac:"+config["endpoint"]
    with db:
        db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?,?,?,?,?,?)",(dataset,"Live STAC metadata","rolling",config["endpoint"],"Asset-specific; verify before downloading","Provider STAC metadata",config_hash(config)))
        db.execute("INSERT INTO ingestion_jobs(id,provider,query_config,start_time,status) VALUES(?,?,?,?,?)",(jid,config["endpoint"],json.dumps(config),now(),"running"))
    url=config["endpoint"].rstrip("/")+"/search"
    body={k:config[k] for k in ("bbox","datetime","collections","limit","query") if k in config}
    found=added=0; cursor=None
    try:
        for page in range(config.get("max_pages",2)):
            result=fetch(url,body)
            for item in result["features"]:
                found+=1; props=item["properties"]; key=item["collection"]+":"+item["id"]
                with db:
                    cur=db.execute("INSERT OR IGNORE INTO scenes VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(dataset+":"+key,dataset,key,props.get("s2:mgrs_tile",key),props.get("platform","unknown"),props.get("datetime"),props.get("created"),now(),str(props.get("proj:epsg","")),json.dumps(props.get("proj:transform")),json.dumps(item.get("bbox")),json.dumps(config.get("bands",[])),"see asset metadata",None,"metadata only",json.dumps(item["assets"]),config_hash(item)))
                    added+=cur.rowcount
            nxt=next((l for l in result.get("links",[]) if l["rel"]=="next"),None)
            if not nxt:
                cursor=None; break
            url=nxt["href"]; body=nxt.get("body") if nxt.get("method","GET")=="POST" else None
            cursor=json.dumps(nxt)
        with db:
            db.execute("UPDATE ingestion_jobs SET end_time=?,status=?,items_found=?,items_added=?,cursor_or_watermark=? WHERE id=?",(now(),"partial" if cursor else "completed",found,added,cursor,jid))
    except Exception as e:
        with db:
            db.execute("UPDATE ingestion_jobs SET end_time=?,status='failed',items_found=?,items_added=?,failures=? WHERE id=?",(now(),found,added,type(e).__name__+": "+str(e),jid))
        raise
    finally:
        db.close()
    return {"job_id":jid,"items_found":found,"items_added":added,"status":"partial" if cursor else "completed","asset_downloads":0}
