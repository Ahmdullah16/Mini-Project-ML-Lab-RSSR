"""Rebuild frozen tiles from pinned public sources, preserving original manifest bytes.

This verifies data, not model quality. It never trains, selects or evaluates a model.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import torch
from rssr.core import digest
from rssr.data import download, prepare
from rssr.db import connect
from rssr.hosted_assets import fetch_asset
from rssr.frozen_portability import frozen_npz_bytes, reconcile_wgs84_roundoff

ASSET_ROOT = 'https://raw.githubusercontent.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/b21b797a2c43f3a45879cd6778be4cd81600f2e7/artifacts/'
ASSETS = {
    'source-4band.json': ('5478ce0faa52cdfc99b46d3216cec636883bf22afac5fe818bf80b06f5e66cad', 5437),
    'prepared-4band.json': ('35043242103a7088274bc8c14c9bb506eec8a8e23d05ce007cced1754f64c09e', 65694),
    'source-3band.json': ('90d092c8bc2d9a54f9c7b674b1eb23720af5493ba048dd82d800e0f4100df6aa', 5477),
    'prepared-3band.json': ('006e268adc51c706d11055a61bbebb51f19bb8291d5f7d8404553ec1f7af69c3', 65682),
}


def rebuild(data_root):
    data_root = Path(data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    fetched = {}
    for name, (sha, size) in ASSETS.items():
        fetched[name] = fetch_asset({'url': ASSET_ROOT + name, 'sha256': sha, 'max_bytes': size}, data_root/'manifest-cache')
    transferred = 0
    results = []
    for bands in (4, 3):
        source_path = fetched[f'source-{bands}band.json']
        source = json.loads(source_path.read_text(encoding='utf-8'))
        for scene in source['scenes']:
            transferred += download(scene['url'], data_root/scene['path'], scene['sha256'])
        expected_path = fetched[f'prepared-{bands}band.json']
        expected = json.loads(expected_path.read_text(encoding='utf-8'))
        prepared_path = prepare(source_path, data_root)
        rebuilt = json.loads(prepared_path.read_text(encoding='utf-8'))
        roundoff_scenes = reconcile_wgs84_roundoff(rebuilt['scenes'], expected['scenes'])
        if len(rebuilt['tiles']) != len(expected['tiles']):
            raise ValueError('Frozen tile count differs')
        normalized_files = 0
        for tile, wanted in zip(rebuilt['tiles'], expected['tiles']):
            if tile['id'] != wanted['id'] or tile['path'] != wanted['path']:
                raise ValueError('Frozen tile identity differs')
            path = prepared_path.parent / tile['path']
            original = path.read_bytes()
            canonical = frozen_npz_bytes(original, wanted['sha256'])
            if canonical != original:
                temporary = path.with_suffix('.canonical.tmp')
                temporary.write_bytes(canonical); temporary.replace(path)
                normalized_files += 1
            tile['sha256'] = digest(path)
        if rebuilt != expected:
            differing = [key for key in expected if rebuilt.get(key) != expected[key]]
            for key in differing:
                if isinstance(expected[key], list):
                    for actual, wanted in zip(rebuilt[key], expected[key]):
                        if actual != wanted:
                            print('First difference', key, {k: {'actual': actual.get(k), 'expected': v} for k, v in wanted.items() if actual.get(k) != v}, flush=True)
                            break
            with zipfile.ZipFile(prepared_path.parent / rebuilt['tiles'][0]['path']) as archive:
                print('NPZ container metadata', [(i.filename, i.create_system, i.external_attr, i.date_time) for i in archive.infolist()], flush=True)
            raise ValueError(f'Rebuilt {bands}-band data differs from frozen record: {differing}; no canonical manifest installed')
        for tile in expected['tiles']:
            if digest(prepared_path.parent/tile['path']) != tile['sha256']:
                raise ValueError('Frozen tile checksum mismatch')
        # JSON newline conversion varies by OS. Only after full semantic and tile
        # verification do we install the published original bytes for frozen hashes.
        shutil.copyfile(expected_path, prepared_path)
        assert digest(prepared_path) == ASSETS[f'prepared-{bands}band.json'][0]
        db = connect(data_root/'catalog.sqlite')
        with db:
            for tile in expected['tiles']:
                db.execute('UPDATE pairs_or_tiles SET checksums=? WHERE id=?', (tile['sha256'], tile['id']))
        db.close()
        results.append({'bands': bands, 'tiles_verified': len(expected['tiles']),
                        'normalized_zip_creator_files': normalized_files,
                        'derived_bounds_roundoff_scenes': roundoff_scenes,
                        'manifest': prepared_path.relative_to(data_root).as_posix(),
                        'manifest_sha256': digest(prepared_path)})
    db = connect(data_root/'catalog.sqlite')
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    db.close()
    report = {'status': 'passed', 'raw_bytes_transferred_this_run': transferred,
              'manifest_asset_bytes_if_empty_cache': sum(size for _, size in ASSETS.values()),
              'results': results, 'database_integrity': 'passed',
              'model_evaluation': 'not performed; existing frozen test results unchanged'}
    (data_root/'rebuild-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, help='New or resumable destination; about 18 MB pinned raw imagery')
    args = parser.parse_args()
    print(json.dumps(rebuild(args.data_root), indent=2))
