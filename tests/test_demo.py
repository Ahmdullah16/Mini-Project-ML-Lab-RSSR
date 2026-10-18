import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('rssr_demo',Path(__file__).resolve().parents[1]/'app/demo.py')
demo=importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)

@pytest.mark.parametrize('image',[None,np.zeros((2,2),np.uint8),np.zeros((2,2,4),np.uint8),np.zeros((1025,1,3),np.uint8),np.zeros((2,2,3),np.float32)])
def test_invalid_upload_fails_before_checkpoint_load(image):
    with pytest.raises(ValueError): demo.predict(image,'nonexistent-checkpoint.pt')
