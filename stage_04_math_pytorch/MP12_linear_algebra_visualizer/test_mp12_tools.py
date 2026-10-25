from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
from mp12_tools import approximate


def test_diagonal_hand_solution_and_storage():
    x=np.diag([3.,2.,1.])
    y,m=approximate(x,1)
    np.testing.assert_allclose(y,np.diag([3.,0.,0.]))
    assert m['frobenius_error']==pytest.approx(np.sqrt(5))
    assert m['tail_energy']==5 and m['factor_bytes']==56 and m['dense_bytes']==72
    np.testing.assert_allclose(approximate(x,3)[0],x)
    np.testing.assert_allclose(approximate(x,0)[0],0)


def test_rectangular_outer_product_and_invalid_rank():
    x=np.outer([1.,2.,3.],[2.,-1.,4.,0.])
    y,m=approximate(x,1)
    np.testing.assert_allclose(y,x,atol=1e-13)
    for rank in [-1,4,1.5,True]:
        with pytest.raises(ValueError): approximate(x,rank)
    with pytest.raises(ValueError): approximate([[np.nan]],1)
