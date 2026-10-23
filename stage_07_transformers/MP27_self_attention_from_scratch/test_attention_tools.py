import unittest
import numpy as np
from attention_tools import attention, softmax, score_bytes, entropy


class AttentionChecks(unittest.TestCase):
    def test_hand_example(self):
        output, weights = attention([[1, 0]], [[1, 0], [0, 1]], [[10], [20]])
        p = np.exp(1/np.sqrt(2))/(np.exp(1/np.sqrt(2))+1)
        np.testing.assert_allclose(weights, [[p, 1-p]])
        np.testing.assert_allclose(output, [[10*p+20*(1-p)]])

    def test_stability_and_translation(self):
        np.testing.assert_allclose(softmax([[1000, 1001]]), softmax([[0, 1]]))
        np.testing.assert_allclose(softmax([[0, 0, 0]]), [[1/3]*3])

    def test_masks_and_shapes(self):
        q = np.eye(2); v = np.array([[3, 4, 5], [6, 7, 8]])
        out, weights = attention(q, q, v, np.eye(2, dtype=bool))
        np.testing.assert_array_equal(out, v)
        np.testing.assert_array_equal(weights, np.eye(2))
        self.assertEqual(out.shape, (2, 3))

    def test_permutation_and_immutability(self):
        rng = np.random.default_rng(27); q = rng.normal(size=(5, 3)); original = q.copy()
        v = rng.normal(size=(5, 2)); p = [2, 0, 4, 1, 3]
        out, _ = attention(q, q, v)
        permuted, _ = attention(q[p], q[p], v[p])
        np.testing.assert_allclose(permuted, out[p]); np.testing.assert_array_equal(q, original)

    def test_invalid(self):
        for q, k, v in [([], [[1]], [[1]]), ([[np.nan]], [[1]], [[1]]),
                        ([[1, 2]], [[1]], [[1]]), ([[1]], [[1]], [[1], [2]]),
                        ([[1e308]], [[1e308]], [[1]])]:
            with self.subTest(q=q), self.assertRaises(ValueError): attention(q, k, v)
        for mask in [np.zeros((1, 2), dtype=bool), np.ones((1, 2)), np.ones((2, 1), dtype=bool)]:
            with self.subTest(mask=mask), self.assertRaises(ValueError): softmax([[1, 2]], mask)

    def test_memory_and_entropy(self):
        self.assertEqual(score_bytes(128), 65536)
        self.assertEqual(score_bytes(256), 4*score_bytes(128))
        np.testing.assert_allclose(entropy([[.5, .5], [1, 0]]), [np.log(2), 0])
        for n in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError): score_bytes(n)


if __name__ == '__main__': unittest.main()
