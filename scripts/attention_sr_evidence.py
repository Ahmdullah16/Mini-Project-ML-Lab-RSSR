"""Export MP26 curves from completed development runs; never retrain or access test."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
records = json.loads((root/'reports/submission-study.json').read_text())
rows = [r for r in records if r['bands'] == 4 and r['variant'] in ('cnn', 'attention')]
assert len(rows) == 6
curves = []
for row in rows:
    report = json.loads((root/'data/runs'/row['run_id']/'training.json').read_text())
    assert report['config'] == row['config']
    curves.append({'model': row['variant'], 'seed': row['seed'], 'run_id': row['run_id'],
                   'history': report['history']})
benchmark = json.loads((root/'reports/development-benchmark.json').read_text())
summary = []
for name in ('cnn', 'attention'):
    group = [r for r in rows if r['variant'] == name]
    summary.append({'model': name, 'mean_validation_delta_db': statistics.mean(r['scene_mean_delta_db'] for r in group),
                    'seed_sd_delta_db': statistics.stdev(r['scene_mean_delta_db'] for r in group),
                    'parameters': group[0]['parameters']})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
for row in curves:
    for ax, key in zip(axes, ['train_l1', 'validation_l1']):
        ax.plot([r['step'] for r in row['history']], [r[key] for r in row['history']],
                label=f"{row['model']} seed {row['seed']}")
        ax.set(xlabel='Step', ylabel=key); ax.legend(fontsize=6)
fig.savefig(root/'reports/mp26-attention-sr.png', dpi=120); plt.close(fig)
result = {'summary': summary, 'curves': curves,
          'efficiency': [r for r in benchmark['efficiency'] if r['bands'] == 4 and r['variant'] in ('cnn', 'attention')],
          'protocol': 'Existing four-train/four-validation scene synthetic NAIP x2, seeds 17/23/41, 1000 updates Adam .001 batch4 L1; select minimum validation L1 every100 steps. No new training or test access.',
          'limitations': 'Small educational network; channel gate uses average pooling only. Unequal parameters despite equal updates. Seed SD is not geographic uncertainty. Timing differences can be noise; partial MAC count excludes pooling/activations/interpolation.'}
(root/'reports/mp26-attention-sr.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
