import pytest
from rssr.legacy_architecture import registry_config


@pytest.mark.parametrize('kind,attention,expected', [('cnn',True,'M1'),('attention',True,'M4'),('attention',False,'M1'),('transformer',True,'M3'),('hybrid',False,'M5'),('hybrid',True,'M7')])
def test_actual_structure_not_label(kind,attention,expected):
    result=registry_config({'kind':kind,'use_attention':attention,'bands':4,'width':8,'zero_residual':True})
    assert result['architecture_id']=='cat-v1/'+expected
    assert result['bands']==4 and result['width']==8 and result['zero_residual']


@pytest.mark.parametrize('config',[{'gate':True},{'constant_gate':.2},{'fusion':'add'},{'kind':'srcnn'},{'unrecorded_option':True}])
def test_no_silent_variant_relabelling(config):
    with pytest.raises(ValueError):registry_config(config)
