import unittest
import numpy as np
from window_tools import patchify,unpatchify,connectivity,score_counts


class WindowChecks(unittest.TestCase):
    def test_exact_roundtrip_and_order(self):
        x=np.arange(4*6*3,dtype=np.uint16).reshape(4,6,3);old=x.copy();tokens=patchify(x,2)
        np.testing.assert_array_equal(tokens[0],x[:2,:2].reshape(-1));np.testing.assert_array_equal(tokens[1],x[:2,2:4].reshape(-1))
        np.testing.assert_array_equal(unpatchify(tokens,x.shape,2),x);np.testing.assert_array_equal(x,old)
        self.assertEqual(tokens.dtype,x.dtype)

    def test_boundary_connectivity(self):
        local=connectivity(4,4,2);shifted=connectivity(4,4,2,1)
        self.assertFalse(local[1,2]);self.assertTrue(shifted[1,2]);self.assertFalse(shifted[0,3])
        self.assertTrue(np.diag(shifted).all());np.testing.assert_array_equal(shifted,shifted.T)
        self.assertEqual(int(local.sum()),score_counts(4,4,2)['local_pairs'])

    def test_two_layer_reachability(self):
        local=connectivity(4,4,2);shifted=connectivity(4,4,2,1)
        reached=(shifted.astype(int)@local.astype(int))>0
        self.assertTrue(((reached|local)==reached).all())
        self.assertGreater(int(reached.sum()),int(local.sum()))
        self.assertLess(int(reached.sum()),4**4)

    def test_hand_memory(self):
        counts=score_counts(4,4,2)
        self.assertEqual(counts,{'tokens':16,'global_pairs':256,'local_pairs':64,'global_float32_bytes':1024,'local_float32_bytes':256})

    def test_invalid(self):
        for image,size in [(np.zeros((3,4,1)),2),(np.zeros((4,4)),2),(np.zeros((4,4,1)),0),(np.full((4,4,1),np.nan),2)]:
            with self.assertRaises(ValueError):patchify(image,size)
        with self.assertRaises(ValueError):unpatchify(np.zeros((4,4)),(4,4,3),2)
        with self.assertRaises(ValueError):connectivity(4,4,2,2)
        with self.assertRaises(ValueError):connectivity(128,128,4)
        with self.assertRaises(ValueError):score_counts(3,4,2)


if __name__=='__main__':unittest.main()
