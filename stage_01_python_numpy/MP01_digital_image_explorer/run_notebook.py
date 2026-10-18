"""Execute this lesson's plain Python cells without Jupyter; capture outputs.

This is deliberately not a general Jupyter runner: magics, widgets, and rich
IPython display hooks are unsupported. Exceptions stop execution with a nonzero
exit status. Run in a fresh Python process for a fresh namespace.
"""
import base64
from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile


def main():
    project = Path(__file__).resolve().parent
    os.chdir(project)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mp01-mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    notebook = json.loads((project / 'digital_image_explorer.ipynb').read_text())
    namespace = {'__name__': '__main__'}
    plots = 0
    executed = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        stdout, stderr = StringIO(), StringIO()
        outputs = []

        def capture_show(*args, **kwargs):
            nonlocal plots
            for number in plt.get_fignums():
                buffer = BytesIO()
                plt.figure(number).savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                outputs.append({'output_type': 'display_data', 'metadata': {},
                                'data': {'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}})
                plots += 1
            plt.close('all')

        plt.show = capture_show
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        # A failure propagates: never produce a success report from a partial run.
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(source, f'notebook-cell-{index}', 'exec'), namespace)
        for name, stream in [('stdout', stdout), ('stderr', stderr)]:
            if stream.getvalue():
                outputs.append({'output_type': 'stream', 'name': name, 'text': stream.getvalue()})
        executed += 1
        cell['execution_count'] = executed
        cell['outputs'] = outputs
    output = project / 'outputs'
    output.mkdir(exist_ok=True)
    (output / 'digital_image_explorer.executed.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'PASS: {executed} Python cells executed sequentially; {plots} figures captured; no cell exceptions.')
    print('Method: plain Python shared namespace, Agg plots; not a Jupyter kernel execution.')
    print('Optional user image skipped by default. Generated artifacts are in outputs/.')


if __name__ == '__main__':
    main()
