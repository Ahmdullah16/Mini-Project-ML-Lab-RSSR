"""Foundation guards: provenance, scoped styling and accessible token contrast."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'app'))
from rssr_ui.tokens import PALETTES, contrast, css
from rssr_ui.facts import fact_records
from rssr.evidence import build_index


def test_both_palettes_have_readable_text_and_control_borders():
    for p in PALETTES.values():
        for surface in ('card', 'bg'):
            for text in ('text', 'muted', 'accent'):
                assert contrast(p[text], p[surface]) >= 4.5, (text, surface)
            assert contrast(p['line'], p[surface]) >= 3


def test_css_is_scoped_and_motion_is_optional():
    for mode in PALETTES:
        s = css(mode)
        assert 'data-testid' not in s and '@import' not in s and 'url(' not in s
        assert 'prefers-reduced-motion' in s and 'translateY' not in s
        assert 'translateY' in css(mode, reduced=False)
        assert 'filter:' not in s and 'opacity:' not in s


def test_facts_are_registry_derived_and_not_mastery_or_performance_claims():
    index = build_index(ROOT)
    rows = {f['id']: f for f in fact_records(ROOT, index)}
    assert rows['lessons']['value'] == len(index['projects']) == 40
    assert rows['stages']['value'] == len({p['stage'] for p in index['projects']}) == 9
    assert rows['families']['value'] == 7
    assert rows['scales']['value'] == 6
    assert all(r['source'] and r['limitations'] for r in rows.values())
    smaller = dict(index, projects=index['projects'][:2])
    assert next(r for r in fact_records(ROOT, smaller) if r['id']=='lessons')['value'] == 2
