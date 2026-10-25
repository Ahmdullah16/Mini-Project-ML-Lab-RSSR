"""Local MP05 API. Algorithms remain shared in rssr.degradation_lab; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.degradation_lab import gaussian, filter_hwc, degrade_hwc, spectrum

__all__ = ['gaussian', 'filter_hwc', 'degrade_hwc', 'spectrum']
