"""Execute exported RGB job and Gradio callback without starting a server."""
import io,json,os,sys,tempfile,zipfile,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np,rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'app')]
from rssr.scaling import job_bundle
from rssr.hosted_assets import fetch_asset
spec=json.loads((ROOT/'configs/hosted-model.json').read_text(encoding='utf-8'))
a=np.full((3,5,3),80,np.uint8)
with tempfile.TemporaryDirectory(prefix='rssr-job-check-') as temp:
    folder=Path(temp)
    with zipfile.ZipFile(io.BytesIO(job_bundle(a,spec,8))) as z:z.extractall(folder)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUTF8='1')
    dry=subprocess.run([sys.executable,'run.py'],cwd=folder,env=env,check=True,capture_output=True,text=True)
    assert json.loads(dry.stdout)['output_width']==40 and not (folder/'output').exists()
    subprocess.run([sys.executable,'run.py','--execute'],cwd=folder,env=env,check=True,capture_output=True,text=True)
    meta=json.loads((folder/'output/provenance.json').read_text(encoding='utf-8'))
    with rasterio.open(folder/'output'/meta['output']) as src:assert (src.width,src.height)==(40,24)
    checkpoint=fetch_asset(spec['checkpoint'],Path(tempfile.gettempdir())/'rssr-public-assets-v1')
    os.environ['RSSR_CHECKPOINT']=str(checkpoint)
    import demo
    app=demo.build()
    callback=next(f.fn for f in app.fns.values() if f.api_name=='predict_selected')
    with patch('tempfile.mkdtemp',return_value=str(folder/'gradio')):
        baseline,preview,result,image_path,meta_path=callback(a,'RGB pilot',4)
    with rasterio.open(image_path) as src:assert (src.width,src.height)==(20,12)
    assert preview.shape==(12,20,3) and result['scale']==4
    app.close()
(ROOT/'reports/scale-portable-validation.json').write_text(json.dumps({'scope':'Actual cached pilot; synthetic 5x3 input; no public server.','portable_job_dry_run_no_output':True,'portable_job_x8_output':[40,24],'gradio_callback_x4_output':[20,12]},indent=2),encoding='utf-8')
print('Exported job dry-run/execution and actual Gradio callback passed')
