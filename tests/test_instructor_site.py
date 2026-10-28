import importlib.util
import json
from pathlib import Path
from rssr.learning_content import visitor_prose

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagrams',ROOT/'app/architecture_diagrams.py')
diagrams=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagrams)


def test_diagrams_represent_all_seven_distinct_paths():
    assert len(set(diagrams.diagram(k) for k in diagrams.MODULES))==7
    for key,(c,a,t) in diagrams.MODULES.items():
        graph=diagrams.diagram(key)
        assert ('E -> L;' in graph)==bool(c)
        assert ('E -> T;' in graph)==bool(t)
        assert (' -> A;' in graph)==bool(a)
        assert ('2d→d' in graph)==bool(c and t)
        assert 'Input -> Bicubic -> Add;' in graph


def test_visitor_links_preserve_references_without_source_navigation():
    text='[Notebook](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/a.ipynb) [Report](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR/blob/main/a.md) [RCAN](https://arxiv.org/abs/1807.02758)'
    result=visitor_prose(text)
    assert 'github.com' not in result and 'ipynb' not in result
    assert '[RCAN](https://arxiv.org/abs/1807.02758)' in result


def test_conclusion_frozen_numbers_match_recorded_rows():
    text=(ROOT/'paper/main.tex').read_text(encoding='utf-8')
    rows=json.loads((ROOT/'reports/final-heldout.json').read_text())['summary']
    for model in ('cnn','hybrid','bicubic'):
        row=next(r for r in rows if r['bands']==3 and r['model']==model)
        assert f"{row['psnr']:.4f}" in text
