"""Local MP02 API. Algorithms remain shared in rssr.array_lab; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.array_lab import validate_hwc, brighten, masked_channel_stats

__all__ = ['validate_hwc', 'brighten', 'masked_channel_stats']
