import unittest
import torch
from encoder_tools import Encoder,positions


class EncoderChecks(unittest.TestCase):
    def test_layernorm_hand_example(self):
        block=Encoder(2,1).double()
        actual=block.norm1(torch.tensor([[[1.,3.]]],dtype=torch.float64))
        expected=torch.tensor([[[-1.,1.]]],dtype=torch.float64)/(1+1e-5)**.5
        torch.testing.assert_close(actual,expected)

    def test_zero_branches_preserve_residual(self):
        block=Encoder().double()
        with torch.no_grad():
            for p in block.parameters():p.zero_()
        x=torch.arange(12,dtype=torch.float64).reshape(1,3,4)
        y,trace=block(x);torch.testing.assert_close(y,x)
        self.assertEqual(trace['ffn_hidden'],[1,3,8])

    def test_no_position_permutation(self):
        torch.manual_seed(29);block=Encoder().double().eval();x=torch.randn(1,5,4,dtype=torch.float64);p=[3,0,4,1,2]
        y,_=block(x);yp,_=block(x[:,p]);torch.testing.assert_close(yp,y[:,p])
        located,_=block(x+positions(5,4));moved,_=block(x[:,p]+positions(5,4))
        self.assertGreater(float((moved-located[:,p]).abs().max().detach()),.01)

    def test_positions_and_gradients(self):
        torch.testing.assert_close(positions(2,4)[0],torch.tensor([0.,1.,0.,1.],dtype=torch.float64))
        block=Encoder().double();x=torch.ones(1,3,4,dtype=torch.float64,requires_grad=True)
        y,_=block(x);y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()))

    def test_invalid(self):
        for width,heads in [(4,3),(0,1),(4,False)]:
            with self.assertRaises(ValueError):Encoder(width,heads)
        for tokens,width in [(0,4),(2,3),(True,4)]:
            with self.assertRaises(ValueError):positions(tokens,width)
        for x in [torch.ones(3,4),torch.ones(1,0,4),torch.full((1,3,4),float('nan'))]:
            with self.assertRaises(ValueError):Encoder()(x)


if __name__=='__main__':unittest.main()
