"""Local MP12 API. Algorithms remain shared in rssr.linear_lab; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.linear_lab import approximate

__all__ = ['approximate']
