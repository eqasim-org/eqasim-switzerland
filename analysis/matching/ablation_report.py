"""Rebuild four-model ablation figures from saved scores without refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ablation_experiment import VARIANTS


def configure(context):
    context.stage('analysis.matching.ablation_experiment')


def execute(context):

    # Render saved selection and test results without performing additional fits.

    return render(context.stage('analysis.matching.ablation_experiment'))


def render(directory):
    root = Path(directory)
    scores = pd.read_csv(root / 'scores.csv')
    membership = pd.read_csv(root / 'membership.csv').set_index('mz_id')
    manifest = json.loads((root / 'manifest.json').read_text())
    variants = manifest.get('variants', VARIANTS)
    metrics = [('energy', 'Joint behaviour'), ('chain_brier', 'Activity and mode chains'),
               ('crps_distance_km', 'Daily distance'), ('crps_trip_count', 'Total trips'),
               ('crps_count_work', 'Work visits'), ('crps_count_shop', 'Shopping visits'),
               ('crps_count_leisure', 'Leisure visits'), ('crps_count_education', 'Education visits')]
    # Shared coverage cohort across every method and seed, separately per phase.
    cohort = scores.groupby(['phase', 'mz_id']).covered.min()
    valid = set(cohort[cohort.eq(1)].index)
    scores = scores.loc[[(p, i) in valid for p, i in zip(scores.phase, scores.mz_id)]]
    summary = []
    for (phase, method, seed), group in scores.groupby(['phase', 'method', 'seed']):
        for key, label in metrics:
            summary.append(dict(phase=phase, method=method, seed=seed, metric=key,
                                score=np.average(group[key], weights=group.survey_weight)))
    summary = pd.DataFrame(summary)
    summary.to_csv(root / 'summary.csv', index=False)
    selected = summary.loc[(summary.phase == 'selection') & (summary.metric == 'energy')].groupby('method').score.mean().idxmin()
    intervals = []
    rng = np.random.default_rng(manifest['seed'])
    for phase in ['selection', 'test']:
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        for ax, (key, label) in zip(axes.flat, metrics):
            values = summary.loc[(summary.phase == phase) & (summary.metric == key)]
            for j, name in enumerate(variants):
                data = values.loc[values.method == name, 'score'].to_numpy()
                ax.bar(j, data.mean(), color='#0072B2' if name == selected else '#999999')
                ax.scatter(np.full(len(data), j), data, color='black', s=18)
            ax.set_title(label + '\nLower is better')
            ax.set_xticks(range(len(variants)), variants, rotation=30, ha='right')
        fig.tight_layout(); fig.savefig(root / f'{phase}.png', dpi=180); plt.close(fig)
    test = scores[scores.phase.eq('test')]
    for name in list(variants)[1:]:
        for key, _ in metrics:
            pivot = test.groupby(['mz_id','method'])[key].mean().unstack()
            weight = test.groupby('mz_id').survey_weight.first().reindex(pivot.index)
            delta = pivot[name] - pivot['Current']
            grouped = pd.DataFrame({'household':membership.reindex(pivot.index).household,
                'num':delta*weight, 'den':weight}).groupby('household')[['num','den']].sum().to_numpy()
            bootstrap = []
            for _ in range(1000):
                sums = grouped[rng.integers(len(grouped), size=len(grouped))].sum(0)
                bootstrap.append(sums[0]/sums[1])
            low, high = np.quantile(bootstrap,[.025,.975])
            intervals.append(dict(method=name,metric=key,difference=np.average(delta,weights=weight),low=low,high=high))
    intervals=pd.DataFrame(intervals);intervals.to_csv(root/'paired_intervals.csv',index=False)
    fig, axes=plt.subplots(2,4,figsize=(17,8))
    for ax,(key,label) in zip(axes.flat,metrics):
        rows=intervals[intervals.metric.eq(key)]
        ax.errorbar(rows.difference,rows.method,xerr=[rows.difference-rows.low,rows.high-rows.difference],fmt='o',capsize=4)
        ax.axvline(0,color='gray',linestyle='--');ax.set_title(label+'\nLeft of zero favours the addition')
    fig.tight_layout();fig.savefig(root/'differences.png',dpi=180);plt.close(fig)
    (root/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Four matching models</title>
<style>body{{font:17px sans-serif;max-width:1500px;margin:30px auto;line-height:1.5}}img{{width:100%}}</style>
<h1>Four matching models: counts and feature reconstruction</h1>
<p>All variants share inputs, loss weights, retrieval rules and household splits. Three fitting seeds
and {manifest['draws']} donor draws per recipient are used. Current retains the existing model;
Counts replaces its count head with one five-output head; Features replaces projected-input reconstruction
with masked reconstruction of age, income, household composition, density, commute distance, sex,
car resources, public transport accessibility, employment, student and residential context; the fourth does both.
Total-trip loss and the average of four purpose-count losses receive the same count weight.
Counts cover full diaries and include secondary work/education. Missing features have no reconstruction error.</p>
<p>Approximately 60% of households fit selection-stage models, 20% select the variant and 20% are reserved
for testing. Each fit uses an internal 10% early-stopping group, at least ten epochs and the existing
maximum epochs/patience. Test-stage models fit the 80% development group. This is a follow-up experiment
informed by earlier results, not external validation. Scores use a common fully matched cohort.</p>
<h2>Variant selection — lower is better</h2><p>Bars are averages over fitting seeds; dots show individual fits.
Blue identifies the variant selected by mean selection energy score: <b>{selected}</b>.</p><img src="selection.png">
<h2>Household test — lower is better</h2><p>Test outcomes do not select the variant. The same daily-behaviour
scores evaluate all methods, regardless of their training losses.</p><img src="test.png">
<h2>Do the additions improve on the current model?</h2><p>Differences average over three fits per respondent.
Intervals resample test households 1,000 times. Left of zero means lower error; crossing zero means
this comparison does not establish an improvement. Intervals condition on these fitted models and split;
they are exploratory and are not adjusted for multiple comparisons.</p><img src="differences.png">
<p>No production model is changed automatically. Results should support improvement across relevant outcomes
before integration. <a href="summary.csv">Scores</a> · <a href="paired_intervals.csv">Paired intervals</a> ·
<a href="runtime.csv">Runtime</a> · <a href="manifest.json">Settings</a></p>''')
    return str(root/'report.html')
