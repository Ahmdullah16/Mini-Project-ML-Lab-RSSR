import hashlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
from rssr_ui.passport import create_passport, parse_passport, replay_differences, MAX_BYTES


def example():
    return create_passport(source_hash='a'*64, output_bytes=b'actual output',
        model={'label':'fixture only', 'checkpoint':{'sha256':'b'*64}}, scale=2,
        metadata={'runtime_seconds':.25, 'source_url':'https://private.invalid?token=secret'},
        application={'git_commit':'test-fixture'})


def test_hashes_privacy_and_inert_replay():
    value = example()
    assert value['output_sha256'] == hashlib.sha256(b'actual output').hexdigest()
    assert 'private.invalid' not in json.dumps(value) and 'secret' not in json.dumps(value)
    parsed = parse_passport(json.dumps(value).encode())
    assert not replay_differences(parsed, value)
    parsed['scale'] = 4
    assert replay_differences(parsed, value) == ['scale']
    assert value['completed_at'] and value['run_id']


@pytest.mark.parametrize('data', [b'x', b'[]', b'{}', b'x'*(MAX_BYTES+1), b'\xff', b'['*2000], ids=['malformed','array','missing','oversized','encoding','nested'])
def test_invalid_imports_fail_closed(data):
    with pytest.raises(ValueError): parse_passport(data)


def test_boolean_scale_and_untrusted_fields_rejected_or_ignored():
    value = example()
    value['scale'] = True
    with pytest.raises(ValueError): parse_passport(json.dumps(value).encode())
    value['scale'] = 2
    value['run_command'] = 'never execute this'
    assert 'run_command' not in parse_passport(json.dumps(value).encode())
