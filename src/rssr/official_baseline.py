"""Optional, unmodified official SwinIR-S RGB x2 baseline."""
import importlib.util
import json
from pathlib import Path
import torch
from .core import digest
from .data import download


def load_swinir(repository,fetch=False):
    root=Path(repository); config=json.loads((root/'configs/official-swinir.json').read_text())
    checkpoint=root/'data/official-swinir/lightweight-x2.pth'
    if fetch: download(config['url'],checkpoint,config['sha256'],max_bytes=20_000_000)
    if digest(checkpoint)!=config['sha256']: raise ValueError('Official checkpoint hash differs')
    spec=importlib.util.spec_from_file_location('rssr_official_swinir',root/'third_party/swinir/network_swinir.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.SwinIR(upscale=2,in_chans=3,img_size=64,window_size=8,img_range=1.,
                       depths=[6,6,6,6],embed_dim=60,num_heads=[6,6,6,6],mlp_ratio=2,
                       upsampler='pixelshuffledirect',resi_connection='1conv')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['params'],strict=True);model.eval()
    return model,config
