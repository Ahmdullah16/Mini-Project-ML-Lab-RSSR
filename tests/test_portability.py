import json,shutil
import numpy as np
from rssr.core import digest
from rssr.experiment import load_tiles


def test_relative_manifest_survives_relocation(tmp_path):
    original=tmp_path/'original'; original.mkdir()
    np.savez(original/'tile.npz',lr=np.ones((3,4,4),np.float32),hr=np.ones((3,8,8),np.float32))
    manifest={'tiles':[{'path':'tile.npz','sha256':digest(original/'tile.npz'),'split':'train'}]}
    (original/'manifest.json').write_text(json.dumps(manifest))
    moved=tmp_path/'moved'; shutil.move(original,moved)
    _,lr,hr=load_tiles(json.loads((moved/'manifest.json').read_text()),'train',moved)[0]
    assert lr.shape==(3,4,4) and hr.shape==(3,8,8)
