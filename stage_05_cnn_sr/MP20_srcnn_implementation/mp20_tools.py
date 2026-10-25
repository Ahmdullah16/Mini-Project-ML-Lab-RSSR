"""Local MP20 API. Algorithms remain shared in rssr.models; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.models import Attention, WindowBlock, SRModel, tiled

__all__ = ['Attention', 'WindowBlock', 'SRModel', 'tiled']
