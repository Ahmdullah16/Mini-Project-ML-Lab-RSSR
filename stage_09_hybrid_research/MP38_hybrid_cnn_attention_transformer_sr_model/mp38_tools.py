"""MP38 local numerical API; full lesson acceptance remains in README."""
from pathlib import Path
import sys
ROOT=next(p for p in (Path(__file__).resolve().parent,*Path(__file__).resolve().parents) if (p/'src/rssr').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from rssr.models import SRModel

def build_hybrid(bands=4,width=8,gate=True):
    return SRModel(bands=bands,width=width,kind='hybrid',gate=gate)
