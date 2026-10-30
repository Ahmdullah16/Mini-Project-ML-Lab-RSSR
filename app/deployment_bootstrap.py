"""Refresh project imports once when a long-lived Streamlit worker changes build.

Streamlit reruns the entry script, but Python retains imported helpers. A newer
renderer must not resolve functions against helpers cached from an older build.
"""
import hashlib
import importlib
from pathlib import Path
import sys
import threading

_STATE='_rssr_deployment_import_state'

def prepare_imports(root):
    root=Path(root).resolve()
    # This state lives outside project modules, so refreshing them retains the lock.
    if not hasattr(sys,_STATE):
        setattr(sys,_STATE,{'lock':threading.RLock(),'fingerprint':None})
    state=getattr(sys,_STATE)
    with state['lock']:
        paths=sorted([* (root/'app').glob('*.py'),*(root/'src/rssr').rglob('*.py'),*(root/'configs').glob('hosted-model*.json')])
        digest=hashlib.sha256(str(root).encode())
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode());digest.update(path.read_bytes())
        fingerprint=digest.hexdigest()
        for directory in (root/'src',root/'app'):
            value=str(directory)
            while value in sys.path:sys.path.remove(value)
            sys.path.insert(0,value)
        changed=state['fingerprint']!=fingerprint
        if changed:
            app_names={p.stem for p in (root/'app').glob('*.py')} - {'deployment_bootstrap','streamlit_app'}
            for name,module in list(sys.modules.items()):
                if name=='rssr' or name.startswith('rssr.'):
                    sys.modules.pop(name,None)
                elif name in app_names:
                    origin=getattr(module,'__file__',None)
                    if origin and Path(origin).resolve().parent==root/'app':sys.modules.pop(name,None)
            importlib.invalidate_caches()
            state['fingerprint']=fingerprint
        return {'source_fingerprint':fingerprint,'imports_refreshed':changed}
