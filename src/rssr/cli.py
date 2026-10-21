import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="RSSR laboratory: explicit bounded workflows")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("download"); d.add_argument("manifest"); d.add_argument("--root",default="data")
    d=sub.add_parser("prepare"); d.add_argument("manifest"); d.add_argument("--root",default="data"); d.add_argument("--scale",type=int,default=2); d.add_argument("--patch",type=int,default=32); d.add_argument("--limit",type=int,default=16)
    d=sub.add_parser("train"); d.add_argument("config"); d.add_argument("manifest"); d.add_argument("--root",default="data"); d.add_argument("--resume")
    d=sub.add_parser("evaluate"); d.add_argument("checkpoint"); d.add_argument("manifest"); d.add_argument("--root",default="data"); d.add_argument("--split",choices=["validation","test"],default="validation"); d.add_argument("--frozen-selection",action="store_true")
    d=sub.add_parser("export"); d.add_argument("database"); d.add_argument("directory")
    d=sub.add_parser("refresh"); d.add_argument("config"); d.add_argument("--root",default="data"); d.add_argument("--resume-job")
    d=sub.add_parser("refresh-incremental"); d.add_argument("config"); d.add_argument("--root",default="data")
    d=sub.add_parser("infer"); d.add_argument("checkpoint"); d.add_argument("input"); d.add_argument("output"); d.add_argument("--divisor",type=float,required=True)
    args=p.parse_args()
    if args.command=="download":
        from .data import download
        manifest=json.loads(Path(args.manifest).read_text()); total=0
        for item in manifest["scenes"]:
            total+=download(item["url"],Path(args.root)/item["path"],item["sha256"])
        print(json.dumps({"asset_bytes_transferred":total}))
    elif args.command=="prepare":
        from .data import prepare
        print(prepare(args.manifest,args.root,args.scale,args.patch,args.limit))
    elif args.command=="train":
        from .experiment import train
        print(train(json.loads(Path(args.config).read_text()),args.manifest,args.root,args.resume))
    elif args.command=="evaluate":
        if args.split=="test" and not args.frozen_selection:
            p.error("Final test requires --frozen-selection after recording model-selection decisions")
        from .experiment import evaluate
        print(evaluate(args.checkpoint,args.manifest,args.root,args.split))
    elif args.command=="export":
        from .db import connect,export
        db=connect(args.database); export(db,args.directory); db.close()
    elif args.command=="refresh":
        from .ingest import refresh
        print(refresh(json.loads(Path(args.config).read_text()),args.root,resume_job=args.resume_job))
    elif args.command=="refresh-incremental":
        from .scheduler import tick
        print(tick(json.loads(Path(args.config).read_text()),args.root))
    elif args.command=="infer":
        from .inference import geotiff
        print(geotiff(args.checkpoint,args.input,args.output,args.divisor))


if __name__=="__main__":
    main()
