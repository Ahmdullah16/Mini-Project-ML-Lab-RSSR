"""Actual AppTest and max-size CPU inference for every published catalog model."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import psutil

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from rssr.experiment import load_model
from rssr.hosted_assets import fetch_asset
from rssr.display import predict_rgb
from streamlit.testing.v1 import AppTest

catalog=[json.loads((root/'configs/hosted-model.json').read_text())]+json.loads((root/'configs/hosted-models.json').read_text())['models']
results=[]
with tempfile.TemporaryDirectory(prefix='rssr-catalog-') as folder:
    os.environ['RSSR_ASSET_CACHE']=folder
    app=AppTest.from_file(str(root/'app/streamlit_app.py')).run(timeout=60)
    app.radio(key='section').set_value('Live SR').run(timeout=60)
    for entry in catalog:
        app.selectbox[0].select(entry['label']).run(timeout=60)
        app.button[0].click().run(timeout=120)
        assert not app.exception and not app.error
        output=app.session_state['result'];assert output['sr'].shape==(32,32,3)
        assert output['metadata']['checkpoint_sha256']==entry['checkpoint']['sha256']
        assert output['metadata']['model_name']==entry['label']
        model,state=load_model(fetch_asset(entry['checkpoint'],folder))
        start=time.perf_counter();_,sr,_=predict_rgb(np.zeros((512,512,3),dtype=np.uint8),model,state,entry['checkpoint']['sha256'],max_side=512)
        assert sr.shape==(1024,1024,3) and np.isfinite(sr).all()
        results.append({'model':entry['label'],'checkpoint_sha256':entry['checkpoint']['sha256'],'run_id':state['run_id'],
                        'max_size_seconds':time.perf_counter()-start,'rss_after_inference_bytes':psutil.Process().memory_info().rss,
                        'maximum_output_shape':list(sr.shape),'status':'passed'})
report={'models':results,'public_deployment':False,'memory_note':'RSS after inference, not peak; Windows CPU clean deployment-only environment'}
(root/'reports/model-catalog-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
