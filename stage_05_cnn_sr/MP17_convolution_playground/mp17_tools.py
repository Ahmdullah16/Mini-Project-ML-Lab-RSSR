"""Local MP17 API. Algorithms remain shared in rssr.convolution_lab; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.convolution_lab import integer, output_size, correlate, receptive_field

__all__ = ['integer', 'output_size', 'correlate', 'receptive_field']
