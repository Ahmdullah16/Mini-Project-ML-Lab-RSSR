import unittest
import numpy as np
from multihead_tools import multihead, projection_parameters
from attention_tools import attention


class MultiheadChecks(unittest.TestCase):
    def test_two_scalar_heads_hand_calculation(self):
        out,maps=multihead(np.eye(2),[np.eye(2)]*4,2)
        p=np.exp(1)/(np.exp(1)+1)
        np.testing.assert_allclose(out,[[p,.5],[.5,p]])
        np.testing.assert_allclose(maps,[[[p,1-p],[.5,.5]],[[.5,.5],[1-p,p]]])

    def test_single_head_reference(self):
        rng=np.random.default_rng(28);x=rng.normal(size=(5,4));w=[rng.normal(size=(4,4)) for _ in range(4)]
        expected,a=attention(x@w[0],x@w[1],x@w[2]);out,maps=multihead(x,w,1)
        np.testing.assert_allclose(out,expected@w[3]);np.testing.assert_allclose(maps[0],a)

    def test_projection_and_parameter_count(self):
        x=np.eye(4);w=[np.eye(4)]*3+[np.zeros((4,4))]
        out,maps=multihead(x,w,4)
        np.testing.assert_array_equal(out,np.zeros_like(x));self.assertEqual(maps.shape,(4,4,4))
        self.assertEqual(sum(v.size for v in w),projection_parameters(4))

    def test_equivariance_and_input_preservation(self):
        rng=np.random.default_rng(28);x=rng.normal(size=(4,4));original=x.copy();w=[np.eye(4)]*4;p=[3,1,0,2]
        out,maps=multihead(x,w,2);yp,ap=multihead(x[p],w,2)
        np.testing.assert_allclose(yp,out[p]);np.testing.assert_allclose(ap,maps[:,p][:,:,p])
        np.testing.assert_array_equal(x,original)

    def test_invalid_contracts(self):
        for heads in [0,-1,3,True,1.5]:
            with self.subTest(heads=heads),self.assertRaises(ValueError):multihead(np.eye(4),[np.eye(4)]*4,heads)
        for weights in [[np.eye(4)]*3,[np.eye(3)]*4,[np.full((4,4),np.nan)]*4]:
            with self.assertRaises(ValueError):multihead(np.eye(4),weights,2)


if __name__=='__main__':unittest.main()
