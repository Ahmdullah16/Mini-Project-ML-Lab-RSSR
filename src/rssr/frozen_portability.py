"""Narrow metadata compatibility for the original Windows frozen artifacts."""
import hashlib
import io
import math
import struct
import zipfile


def frozen_npz_bytes(data, expected_sha256):
    """Only normalize ZIP creator OS bytes; require exact original file hash.

    NPY payload, compression, CRCs and all other bytes remain untouched. The
    authoritative hash prevents a metadata adjustment from hiding data changes.
    """
    if hashlib.sha256(data).hexdigest() == expected_sha256:
        return data
    normalized = bytearray(data)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            offset = archive.start_dir
            for _ in archive.infolist():
                if normalized[offset:offset+4] != b'PK\x01\x02':
                    raise ValueError('Unexpected NPZ central directory')
                # High byte of ZIP "version made by" is creator OS: Windows=0.
                normalized[offset+5] = 0
                name, extra, comment = struct.unpack_from('<HHH', normalized, offset+28)
                offset += 46 + name + extra + comment
    except (zipfile.BadZipFile, struct.error, IndexError) as error:
        raise ValueError('Invalid frozen NPZ container') from error
    result = bytes(normalized)
    if hashlib.sha256(result).hexdigest() != expected_sha256:
        raise ValueError('Frozen tile checksum mismatch after ZIP metadata normalization')
    return result


def reconcile_wgs84_roundoff(actual, expected):
    """Canonicalize only derived bounds within 1e-12 degrees (~0.1 micrometre).

    Source CRS, affine, native bounds, pixels and split decisions are not changed.
    Full scene dictionaries must still compare equal after this operation.
    """
    if len(actual) != len(expected):
        raise ValueError('Scene count differs')
    changes = 0
    for scene, frozen in zip(actual, expected):
        bounds, target = scene['wgs84_bounds'], frozen['wgs84_bounds']
        if len(bounds) != 4 or len(target) != 4 or not all(
            math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
            for a, b in zip(bounds, target)
        ):
            raise ValueError('Derived WGS84 bounds exceed roundoff tolerance')
        changes += bounds != target
        scene['wgs84_bounds'] = list(target)
    return changes
