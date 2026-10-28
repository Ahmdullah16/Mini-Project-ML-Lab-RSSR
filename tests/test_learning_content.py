from pathlib import Path
import pytest
from rssr.learning_content import load_lesson, prose

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('i',range(1,41))
def test_each_lesson_is_readable_without_code_execution(i):
    lesson=load_lesson(ROOT,f'MP{i:02d}')
    assert lesson['overview'] and lesson['explanations'] and lesson['validation']
    assert len(lesson['source_hashes'])==3
    assert '```' not in lesson['overview']+lesson['explanations']
    assert 'not yet individually accepted' in lesson['coverage']


def test_math_and_links_survive_code_removal():
    text='Equation $x^2$.\n\n```python\nraise RuntimeError()\n```\n\n[Source](../../docs/protocol.md)'
    result=prose(text,'stage/MP01/README.md')
    assert '$x^2$' in result and 'RuntimeError' not in result
    assert 'blob/main/docs/protocol.md' in result
    with pytest.raises(ValueError):load_lesson(ROOT,'../../secrets')
