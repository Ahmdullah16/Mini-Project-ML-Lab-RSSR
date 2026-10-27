"""Conservative structural mapping; it alone does not verify training provenance."""
def registry_config(legacy):
    allowed={'bands','scale','width','kind','gate','fusion','window','use_attention','constant_gate','zero_residual'}
    if set(legacy)-allowed:
        raise ValueError('Unknown legacy options')
    if legacy.get('gate',False) or legacy.get('constant_gate',1.) != 1.:
        raise ValueError('Residual-gate variant is outside cat-v1')
    if legacy.get('fusion','concat') != 'concat':
        raise ValueError('Non-concat fusion is outside this mapping')
    kind=legacy.get('kind','hybrid'); attention=legacy.get('use_attention',True)
    mapping={'cnn':'M1','attention':'M4' if attention else 'M1',
             'transformer':'M3','hybrid':'M7' if attention else 'M5'}
    if kind not in mapping:
        raise ValueError('No cat-v1 equivalent')
    return {'architecture_id':'cat-v1/'+mapping[kind],
            'bands':legacy.get('bands',3),'scale':legacy.get('scale',2),
            'width':legacy.get('width',16),'window':legacy.get('window',4),
            'zero_residual':legacy.get('zero_residual',False)}
