"""Local MP04 API. Algorithms remain shared in rssr.image_foundations; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.image_foundations import resize_hwc, compare_hwc

__all__ = ['resize_hwc', 'compare_hwc']
