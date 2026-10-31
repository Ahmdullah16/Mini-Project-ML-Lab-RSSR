"""Project-local API reexports the canonical shared MP16 numerical pipeline.

No duplicate loader or sample-weighted metric formula.
"""
from pathlib import Path
import sys

ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents)
          if (p/"src/rssr").is_dir())
sys.path.insert(0,str(ROOT/"src"))
from rssr.pipeline_lab import make_loader,weighted_mean,tiny_regression

__all__=["make_loader","weighted_mean","tiny_regression"]
