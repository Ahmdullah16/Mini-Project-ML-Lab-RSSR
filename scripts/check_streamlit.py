"""Actual model/AppTest inference in two fresh processes and empty asset caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--worker'); p.add_argument('--output')
a=p.parse_args()
if a.worker:
    os.environ['RSSR_ASSET_CACHE']=a.worker
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import psutil
    from streamlit.testing.v1 import AppTest
    from rssr.display import predict_rgb
    from rssr.experiment import load_model
    from rssr.hosted_assets import fetch_asset
    process=psutil.Process(); samples=[]; stop=threading.Event()
    def sample():
        while not stop.wait(.01): samples.append(process.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True); monitor.start()
    try:
        app=AppTest.from_file(str(root/'app/streamlit_app.py')).run(timeout=60)
        app.radio(key='section').set_value('Live SR').run(timeout=60)
        assert not app.exception
        app.button(key='scaled_liveprepare').click().run(timeout=120)
        app.button(key='scaled_liverun').click().run(timeout=120)
        assert not app.exception and not app.error
        result=app.session_state['scaled_liveresults']['outputs'][0]; result['sr']=result['pixels']; assert result['sr'].shape==(32,32,3)
        manifest=json.loads((root/'configs/hosted-model.json').read_text())
        assert result['metadata']['identity']['model']['checkpoint']['sha256']==manifest['checkpoint']['sha256']
        model,state=load_model(fetch_asset(manifest['checkpoint'],a.worker))
        started=time.perf_counter()
        _,large,_=predict_rgb(np.zeros((512,512,3),dtype=np.uint8),model,state,manifest['checkpoint']['sha256'],max_side=1024)
        assert large.shape==(1024,1024,3)
        for image in [np.zeros((1025,1,3),dtype=np.uint8),np.zeros((2,2,4),dtype=np.uint8)]:
            try: predict_rgb(image,model,state,manifest['checkpoint']['sha256'],max_side=1024)
            except ValueError: pass
            else: raise AssertionError('Invalid upload accepted')
        report={'status':'passed','checkpoint_sha256':manifest['checkpoint']['sha256'],
                'sample_output_sha256':hashlib.sha256(result['sr'].tobytes()).hexdigest(),
                'sample_shape':list(result['sr'].shape),'host_boundary_fixture_input_shape':[512,512,3],
                'host_boundary_fixture_output_shape':list(large.shape),'host_boundary_fixture_seconds':time.perf_counter()-started,
                'sampled_peak_rss_bytes':max(samples+[process.memory_info().rss]),'invalid_inputs_rejected':True,
                'fresh_cache_files':len(list(Path(a.worker).iterdir()))}
        Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally: stop.set(); monitor.join()
else:
    with tempfile.TemporaryDirectory(prefix='rssr-restart-test-') as folder:
        reports=[]
        for number in range(2):
            output=Path(folder)/f'result-{number}.json'
            subprocess.run([sys.executable,__file__,'--worker',str(Path(folder)/f'cache-{number}'),'--output',str(output)],check=True,timeout=240)
            reports.append(json.loads(output.read_text()))
        assert reports[0]['sample_output_sha256']==reports[1]['sample_output_sha256']
        record={'status':'passed','verification':'Two independent processes; empty cache each time; public pinned checkpoint downloaded and verified; actual AppTest and configured host-boundary tiled inference',
                'restart_output_equal':True,'runs':reports,'limitations':'Local CPU evidence; no public Streamlit URL or hosted resource limit verification. RSS is sampled, not an exact allocator peak.'}
        (root/'reports/scale-streamlit-restart.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2))
