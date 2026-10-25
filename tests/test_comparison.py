import io,json,zipfile
import numpy as np
import pytest
from rssr.comparison import input_digest,result_key,validate_result,comparison_bundle

def test_identity_changes_with_input_checkpoint_and_version():
    x=np.zeros((3,4,3),np.uint8);spec={'version':'v1','checkpoint':{'sha256':'a'*64}}
    key=result_key(x,spec);changed=x.copy();changed[0,0,0]=1
    assert key!=result_key(changed,spec)
    assert key!=result_key(x,{**spec,'version':'v2'})
    assert key!=result_key(x,{**spec,'checkpoint':{'sha256':'b'*64}})
    assert key==result_key(x.copy(),spec)
    with pytest.raises(ValueError):input_digest(x.astype(float))
    with pytest.raises(ValueError):input_digest(np.zeros((513,1,3),np.uint8))

def test_native_alignment_and_bundle_labels():
    x=np.zeros((3,4,3),np.uint8);y=np.zeros((6,8,3),np.uint8)
    spec={'version':'v1','checkpoint':{'sha256':'a'*64}};meta={'scale':2,'checkpoint_sha256':'a'*64,'model_name':'actual model'}
    validate_result(x,y,y,meta,spec)
    with pytest.raises(ValueError):validate_result(x,y,y[:5],meta,spec)
    with pytest.raises(ValueError):validate_result(x,y,y,{**meta,'checkpoint_sha256':'wrong'},spec)
    batch={'input':x,'source':{'kind':'synthetic fixture'},'errors':[],'results':[{'key':result_key(x,spec),'baseline':y,'sr':y,'metadata':meta}]}
    with zipfile.ZipFile(io.BytesIO(comparison_bundle(batch))) as archive:
        data=json.loads(archive.read('provenance.json'))
        assert data['results'][0]['model_name']=='actual model'
        assert data['input_sha256']==input_digest(x)
        assert len([name for name in archive.namelist() if name.endswith('.png')])==3
        assert not any('hr' in name for name in archive.namelist())
