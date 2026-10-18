"""Explicit local HTTP integration check; start app/demo.py first."""
import json,tempfile,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from gradio_client import Client,handle_file
root=Path(__file__).resolve().parents[1]
url='http://127.0.0.1:7860'
assert urllib.request.urlopen(url,timeout=10).status==200
with tempfile.TemporaryDirectory() as directory:
    source=Path(directory)/'synthetic.png'
    Image.fromarray(np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)).save(source)
    result=Client(url,verbose=False).predict(handle_file(str(source)),api_name='/predict')
    shapes=[list(np.asarray(Image.open(path)).shape) for path in result[:2]]
assert shapes==[[64,64,3],[64,64,3]],shapes
expected=json.loads((root/'reports/demo-checkpoint.json').read_text())
from rssr.core import digest
checkpoint=root/Path(expected['checkpoint'].replace('\\','/'))
assert result[2]['checkpoint_sha256']==digest(checkpoint)
evidence={'transport':'local HTTP Gradio client','http_status':200,'input_shape':[32,32,3],'output_shapes':shapes,'provenance':result[2],'status':'passed','fixture':'synthetic RGB ramp; no fidelity claim'}
(root/'reports/demo-http.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(json.dumps(evidence))
