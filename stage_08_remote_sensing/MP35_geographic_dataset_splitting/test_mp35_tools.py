import pytest
from mp35_tools import audit_geography


def scenes():
    return [dict(id='a',region_id='a',split='train',wgs84_bounds=[0,0,1,1]),
            dict(id='b',region_id='b',split='test',wgs84_bounds=[2,0,3,1])]


def test_separation_and_exact_boundary():
    source=scenes()
    assert audit_geography(source)['counts']==dict(train=1,validation=0,test=1)
    assert audit_geography(source)['cross_split_pairs'][0]['max_axis_clearance_degrees']==1
    with pytest.raises(ValueError):audit_geography(source,1)
    assert source==scenes()


@pytest.mark.parametrize('change',[{'id':'a'},{'region_id':'a'}, {'split':'unknown'},
    {'wgs84_bounds':[0.5,0,1.5,1]}, {'wgs84_bounds':[float('nan'),0,3,1]},
    {'wgs84_bounds':[179,0,-179,1]}, {'region_id':''}])
def test_reject_invalid_or_leaking(change):
    source=scenes();source[1].update(change)
    with pytest.raises(ValueError):audit_geography(source)


@pytest.mark.parametrize('buffer',[-1,float('nan'),float('inf')])
def test_invalid_buffer(buffer):
    with pytest.raises(ValueError):audit_geography(scenes(),buffer)


def test_same_split_overlap_and_empty():
    source=scenes();source[1].update(split='train',wgs84_bounds=[0,0,1,1])
    assert not audit_geography(source)['cross_split_pairs']
    with pytest.raises(ValueError):audit_geography([])
