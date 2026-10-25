"""Local MP10 API. Algorithms remain shared in rssr.metric_lab; no duplicate implementation."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.metric_lab import paired, mse_psnr, ssim_map

__all__ = ['paired', 'mse_psnr', 'ssim_map']
