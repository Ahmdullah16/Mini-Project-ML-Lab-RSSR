"""Bounded, inert run manifests. Import never fetches files or executes a model."""
import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import uuid4

SCHEMA = 'rssr-experiment-passport-v1'
MAX_BYTES = 64 * 1024
HASH = re.compile(r'^[0-9a-f]{64}$')


def create_passport(*, source_hash, output_bytes, model, scale, metadata, application,
                    input_shape=None, input_encoding='source file bytes'):
    """Call exactly once on successful inference; retain in session with its output."""
    checkpoint = model['checkpoint']['sha256']
    if not HASH.fullmatch(source_hash) or not HASH.fullmatch(checkpoint):
        raise ValueError('A passport needs verified SHA-256 identities')
    if scale not in (2, 4, 8, 16, 32, 64):
        raise ValueError('Unsupported scale')
    return {
        'schema': SCHEMA, 'run_id': str(uuid4()),
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'input_sha256': source_hash,
        'input_shape': input_shape, 'input_encoding': input_encoding,
        'output_sha256': hashlib.sha256(output_bytes).hexdigest(),
        'checkpoint_sha256': checkpoint, 'model_label': model['label'], 'scale': scale,
        'pipeline': metadata.get('pipeline', 'see accompanying provenance'),
        'runtime_seconds': metadata.get('runtime_seconds'),
        'software': application,
        'reference_metrics': 'unavailable: no eligible aligned HR reference',
        'scientific_status': 'executed display-RGB reconstruction; geographic accuracy not validated',
        'replay_status': 'reviewable manifest only; input and checkpoint artifacts still required',
        'privacy': 'Session-only. Export intentionally includes hashes and configuration, not source URLs, coordinates or image bytes.',
    }


def parse_passport(data):
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError('Passport must be JSON of at most 64 KiB')
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('Invalid passport JSON') from error
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('Unsupported passport schema')
    for field in ('input_sha256', 'output_sha256', 'checkpoint_sha256'):
        if not isinstance(value.get(field), str) or not HASH.fullmatch(value[field]):
            raise ValueError('Missing or invalid artifact hash')
    if type(value.get('scale')) is not int or value['scale'] not in (2, 4, 8, 16, 32, 64):
        raise ValueError('Unsupported scale')
    if value.get('input_encoding') not in ('source file bytes', 'RGB uint8 C-order pixels'):
        raise ValueError('Unsupported input encoding')
    if 'input_shape' not in value:
        raise ValueError('Missing input shape convention')
    shape = value.get('input_shape')
    if value['input_encoding'] == 'RGB uint8 C-order pixels' and (
            not isinstance(shape, list) or len(shape) != 3 or
            any(type(v) is not int for v in shape) or shape[2] != 3 or
            not all(1 <= v <= 1024 for v in shape[:2])):
        raise ValueError('Invalid RGB input shape')
    # No paths, URLs, source content or executable arguments from imports are used.
    return {key: value[key] for key in ('schema', 'input_sha256', 'input_shape', 'input_encoding', 'output_sha256', 'checkpoint_sha256', 'scale')}


def replay_differences(imported, current):
    return [key for key in ('input_sha256', 'input_shape', 'input_encoding', 'checkpoint_sha256', 'scale') if imported[key] != current[key]]


def render_passport(passport, key):
    import streamlit as st
    with st.expander('Experiment passport · '+passport['model_label']):
        st.json(passport)
        st.download_button('Download experiment passport', json.dumps(passport, indent=2),
                           'experiment_passport.json', 'application/json', key=key+'download')
        upload = st.file_uploader('Review an earlier passport (JSON, ≤64 KiB)', type=['json'], key=key+'import')
        if upload is not None:
            try:
                if upload.size > MAX_BYTES:
                    raise ValueError('Passport must be JSON of at most 64 KiB')
                earlier = parse_passport(upload.getvalue())
                differences = replay_differences(earlier, passport)
                if differences:
                    st.warning('Not the same request: '+', '.join(differences))
                else:
                    st.info('Input/checkpoint/scale match. This does not verify identical software, protocol or output. No replay has executed.')
            except ValueError as error:
                st.error(str(error))
