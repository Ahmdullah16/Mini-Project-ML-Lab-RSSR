"""Filter-dependent explanations of recorded evidence; never reruns evaluation."""
import math
import statistics

DEFINITIONS = {
    'psnr': ('PSNR', 'higher', '10 log10(R²/MSE), with declared range R. Compare only identical data, masks and processing; no universal good-score threshold.'),
    'delta_psnr': ('PSNR change over bicubic', 'higher', 'Model PSNR minus matched bicubic PSNR in dB. Positive means lower reconstruction error under this protocol; it does not prove physical detail recovery.'),
    'ssim': ('Structural similarity', 'higher', 'Shared evaluator: 7×7 uniform windows, population moments, constants (0.01R)² and (0.03R)²; only fully valid windows contribute. Unavailable when no valid window exists. Check source conventions for teaching records.'),
    'mse': ('Mean squared error', 'lower', 'Mean squared pixel difference. Units are squared input units; scaling both images by k scales MSE by k².'),
    'rmse': ('Root mean squared error', 'lower', 'Square root of MSE in the input units. Normalized DN is not calibrated reflectance.'),
    'sam_rad': ('Spectral angle', 'lower', 'Angle between predicted and reference band vectors, in radians. Shared evaluator excludes norm products ≤1e-12 and clips cosine to [-1,1]; no valid spectra means unavailable. Small angle does not imply correct brightness.'),
    'parameters': ('Trainable parameter count', 'lower cost', 'Number of learned scalars; fewer parameters do not necessarily mean faster inference or better reconstruction.'),
    'median_seconds': ('Median CPU latency', 'lower cost', 'Median forward time: float32, 64×64 LR, two CPU threads, five warmups and twenty timed repetitions on the recorded Windows host. Excludes acquisition and hosted end-to-end latency.'),
    'p95_seconds': ('95th-percentile CPU latency', 'lower cost', 'Tail of twenty warmed forward timings on the recorded host; a small timing sample, not a service-level guarantee.'),
    'sampled_process_peak_rss_bytes': ('Sampled process memory', 'lower cost', 'Largest sampled process resident-set size in bytes; includes interpreter/model memory and can miss short peaks. Not exact allocator peak or GPU memory.'),
    'conv_linear_attention_macs': ('Partial multiply–accumulate count', 'lower cost', 'Counts implemented convolution, linear and attention products. Omits normalization, activations, pooling and interpolation. MACs are not complete FLOPs; no automatic ×2 conversion is claimed.'),
    'train_l1': ('Training absolute error', 'lower', 'Minibatch mean absolute error in normalized DN. Individual training points are not full-training-set estimates.'),
    'validation_l1': ('Validation absolute error', 'lower', 'Recorded validation mean absolute error in normalized DN. Falling training loss alone does not establish generalization; persistent divergence requires evidence from comparable curves.'),
    'train_mse': ('Training squared error', 'lower', 'Mean squared training error in normalized DN squared; compare the same loss, scaling and sample aggregation.'),
}

CONTEXT = ('study','dataset','data_kind','task','bands','split','condition','protocol','metric','unit')

def metric_definition(metric):
    key = 'rmse' if metric.startswith('rmse_band_') else metric
    title, direction, explanation = DEFINITIONS.get(key, (metric, 'unspecified', 'Definition unavailable; inspect the source before interpreting this value.'))
    if metric.startswith('rmse_band_'):
        band = int(metric.rsplit('_',1)[1])
        title += ' · '+{0:'red',1:'green',2:'blue',3:'near infrared'}.get(band,f'band {band}')
    return {'title':title,'direction':direction,'explanation':explanation}

def explain_selection(rows):
    """Summarize one comparable slice; reject silent pooling across protocols."""
    if not rows:
        return {'available':False,'message':'No recorded values match this selection.'}
    first=rows[0]
    if any(any(r[k]!=first[k] for k in CONTEXT) for r in rows):
        raise ValueError('Select one study, dataset, task, band count, split, condition, protocol and metric')
    definition=metric_definition(first['metric'])
    groups={}
    for row in rows:
        value=row['value']
        if value is None or not math.isfinite(value):
            continue
        groups.setdefault(row['model'],[]).append(value)
    summaries=[{'model':model,'value':statistics.mean(values),'records':len(values)} for model,values in groups.items()]
    return {'available':bool(summaries),**definition,'summary':summaries,
            'question':f"How do the selected methods compare in {definition['title'].lower()} under this recorded condition?",
            'context':{k:first[k] for k in CONTEXT},
            'sources':sorted({r['source'] for r in rows}),
            'observation':'; '.join(f"{r['model']}: {r['value']:.6g} {first['unit']} ({r['records']} source rows)" for r in summaries) or 'Values unavailable.',
            'aggregation':'Bars are arithmetic means of the displayed source rows, not confidence intervals. Seeds and repeated bicubic records are not additional independent scenes. Scene/tile counts in the table belong to each source row and must not be summed blindly.',
            'limits':'Observed differences do not isolate a causal mechanism. Keep architecture, training budget and pretraining differences in view. Synthetic aerial SR does not establish cross-sensor or downstream validity.'}
