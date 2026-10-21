import hashlib
import io
import zipfile
import numpy as np
import pytest
from rssr.frozen_portability import frozen_npz_bytes, reconcile_wgs84_roundoff


def archive(creator, value=1):
    buffer = io.BytesIO()
    payload = io.BytesIO()
    np.save(payload, np.array([value], dtype=np.float32))
    with zipfile.ZipFile(buffer, 'w') as out:
        info = zipfile.ZipInfo('lr.npy'); info.create_system = creator
        with out.open(info, 'w', force_zip64=True) as entry:
            entry.write(payload.getvalue())
    return buffer.getvalue()


def test_creator_os_only_normalization_keeps_exact_frozen_hash():
    original = archive(0); sha = hashlib.sha256(original).hexdigest()
    assert frozen_npz_bytes(archive(3), sha) == original
    assert frozen_npz_bytes(original, sha) == original
    with pytest.raises(ValueError, match='checksum mismatch'):
        frozen_npz_bytes(archive(3, value=2), sha)
    with pytest.raises(ValueError):
        frozen_npz_bytes(b'corrupt', sha)


def test_roundoff_is_narrow_and_other_scene_fields_untouched():
    expected = [{'wgs84_bounds': [-113., 44., -112., 45.], 'crs': 'EPSG:32611'}]
    actual = [{'wgs84_bounds': [-113.+1e-14, 44., -112., 45.], 'crs': 'changed'}]
    assert reconcile_wgs84_roundoff(actual, expected) == 1
    assert actual[0]['crs'] == 'changed'  # Later full comparison still rejects it.
    actual[0]['wgs84_bounds'][0] += 1e-8
    with pytest.raises(ValueError): reconcile_wgs84_roundoff(actual, expected)
