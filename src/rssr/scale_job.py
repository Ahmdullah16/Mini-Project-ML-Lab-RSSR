"""Portable RGB job; dry-run preflight is the default."""
import argparse,json
from pathlib import Path
from .uploads import decode_image
from .scaling import SCALES,plan,cascade


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('model',help='Pinned hosted-model JSON specification')
    p.add_argument('output');p.add_argument('--scale',type=int,choices=SCALES,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    image=decode_image(Path(args.input).read_bytes());spec=json.loads(Path(args.model).read_text(encoding='utf-8-sig'))
    request=plan(image.shape[1],image.shape[0],args.scale);print(json.dumps(request,indent=2),flush=True)
    if not args.execute:return
    from .hosted_assets import fetch_asset
    from .experiment import load_model
    import tempfile
    path=fetch_asset(spec['checkpoint'],Path(tempfile.gettempdir())/'rssr-public-assets-v1')
    model,state=load_model(path)
    print(json.dumps(cascade(image,model,args.output,args.scale,{'model':spec,'run_id':state['run_id']}),indent=2))

if __name__=='__main__':main()
