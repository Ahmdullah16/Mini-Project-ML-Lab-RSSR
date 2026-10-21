"""Inspectable educational attention; does not modify frozen research models."""
import torch
from torch import nn
from .models import Attention


class InspectableAttention(Attention):
    """Return gates as well as features, with explicit controlled alternatives."""

    def forward(self, x, mode="combined"):
        if x.ndim != 4 or x.shape[1] != self.channel[1].in_channels:
            raise ValueError("Expected NCHW with configured channels")
        if mode not in ("off", "uniform", "channel", "spatial", "combined"):
            raise ValueError("Unknown attention mode")
        channel = torch.ones_like(x[:, :, :1, :1])
        spatial = torch.ones_like(x[:, :1])
        if mode == "uniform":
            channel = channel * 0.5
        elif mode in ("channel", "combined"):
            channel = self.channel(x)
        gated = x * channel
        if mode in ("spatial", "combined"):
            pooled = torch.cat([gated.mean(1, keepdim=True), gated.amax(1, keepdim=True)], 1)
            spatial = self.spatial(pooled).sigmoid()
        return gated * spatial, channel, spatial


class AttentionReconstruction(nn.Module):
    """Same-grid RGB reconstruction for controlled teaching comparisons."""

    def __init__(self, mode, width=8):
        super().__init__()
        self.mode = mode
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.attention = InspectableAttention(width)
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x):
        features = self.head(x).relu()
        gated, _, _ = self.attention(features, self.mode)
        return self.tail(gated)
