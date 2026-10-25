from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import pytest
import torch
from mp23_tools import InspectableAttention
from rssr.models import Attention


def test_inspection_preserves_production_gate_and_gradients():
    torch.manual_seed(23)
    original = Attention(8)
    inspected = InspectableAttention(8)
    inspected.load_state_dict(original.state_dict())
    x = torch.randn(2, 8, 5, 7, requires_grad=True)
    y, channel, spatial = inspected(x)
    torch.testing.assert_close(y, original(x))
    assert channel.shape == (2, 8, 1, 1)
    assert spatial.shape == (2, 1, 5, 7)
    assert torch.all((channel > 0) & (channel < 1))
    assert torch.all((spatial > 0) & (spatial < 1))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_disabled_and_uniform_have_explicit_meaning():
    gate = InspectableAttention(4)
    x = torch.randn(1, 4, 3, 5)
    torch.testing.assert_close(gate(x, "off")[0], x)
    torch.testing.assert_close(gate(x, "uniform")[0], x * 0.5)
    with pytest.raises(ValueError):
        gate(x, "unknown")
    with pytest.raises(ValueError):
        gate(torch.zeros(1, 3, 4, 4))
