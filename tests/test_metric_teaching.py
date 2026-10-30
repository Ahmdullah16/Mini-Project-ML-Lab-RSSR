import numpy as np
import pytest
from rssr.metric_teaching import KINDS, demonstrate

@pytest.mark.parametrize('kind',KINDS)
def test_zero_strength_identity(kind):
    r=demonstrate(kind,0)
    np.testing.assert_array_equal(r['reference'],r['distorted'])
    assert r['metrics']['mse']==0 and np.isinf(r['metrics']['psnr'])
    assert r['metrics']['ssim']==pytest.approx(1)
    assert r['metrics']['mae']==r['metrics']['gradient_mae']==0

@pytest.mark.parametrize('kind',KINDS)
def test_controlled_output_and_independent_error_calculation(kind):
    r=demonstrate(kind,1);e=r['distorted'].astype(float)-r['reference'].astype(float)
    assert r['distorted'].shape==(3,64,64)
    assert 0<=r['distorted'].min()<=r['distorted'].max()<=1
    assert r['metrics']['mse']==pytest.approx(np.mean(e**2))
    assert r['metrics']['mae']==pytest.approx(np.mean(abs(e)))
    assert r['metrics']['psnr']==pytest.approx(-10*np.log10(np.mean(e**2)))
    assert 'sam_rad' not in r['metrics']
    np.testing.assert_array_equal(r['distorted'],demonstrate(kind,1)['distorted'])

@pytest.mark.parametrize('kind,strength',[('Fake',0),('Noise',-1),('Noise',float('nan')),('Blur',2)])
def test_invalid(kind,strength):
    with pytest.raises(ValueError):demonstrate(kind,strength)


def test_shared_availability_uses_checkpoint_identity():
    from pathlib import Path
    from rssr.model_availability import availability
    rows=availability(Path(__file__).resolve().parents[1])
    assert len(rows)==7
    assert [r['model'].split(' · ')[0] for r in rows]==[f'M{i}' for i in range(1,8)]
    assert [r['model'].split(' · ')[0] for r in rows if r['rgb_execution']]==[f'M{i}' for i in range(1,8)]
    assert all(r['checkpoint_sha256'] for r in rows)
    assert all('Pinned hosted RGB' in row['status'] for row in rows)
