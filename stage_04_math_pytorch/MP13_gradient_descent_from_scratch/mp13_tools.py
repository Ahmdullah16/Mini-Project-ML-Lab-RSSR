"""Local MP13 API. Algorithms remain shared in rssr.training_lab; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.training_lab import regression_arrays, linear_loss, initialize, forward, mlp_loss, finite_difference_error

__all__ = ['regression_arrays', 'linear_loss', 'initialize', 'forward', 'mlp_loss', 'finite_difference_error']
