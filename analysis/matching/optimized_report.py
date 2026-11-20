"""Compare optimized BSLM with RHDM/OHDM without replacing existing experiments."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import optimize_weights, repeat_experiments

STAGE = 'analysis.matching.optimize_weights'


def configure(context):
    repeat_experiments.configure(context)
    context.stage(STAGE)


def execute(context):

    # Run the established replay and population comparisons in an isolated analysis directory.

    root = Path(context.config('analysis_path')) / 'matching' / 'loss_weight_search'
    proxy = OptimizedContext(context, context.stage(STAGE), root / 'application')
    report = repeat_experiments.execute(proxy)

    # Summarise the independent test and search as figures with explicit interpretation.

    render(root, report)
    return str(root / 'report.html')


def render(root, application_report):
    root = Path(root)
    trials = pd.read_csv(root / 'trials.csv')
    test = pd.read_csv(root / 'test_scores.csv')
    selection = json.loads((root / 'selection.json').read_text())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(trials.trial, trials.energy)
    ax.plot(trials.trial, trials.energy.cummin(), label='Best so far')
    ax.set(xlabel='Trial (0 = current weights)', ylabel='Selection energy score', title='Loss-weight search: lower is better')
    ax.legend()
    fig.tight_layout(); fig.savefig(root / 'search.png', dpi=180); plt.close(fig)
    metrics = [('energy', 'Joint behaviour (energy)'), ('chain_brier', 'Purpose and mode chain (Brier)'),
               ('crps_distance_km', 'Daily distance (CRPS, km)'), ('crps_count_shop', 'Shopping visits (CRPS)')]
    common = test.groupby('mz_id').covered.min()
    cohort = common.index[common.eq(1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, label) in zip(axes.flat, metrics):
        means = test.loc[test.mz_id.isin(cohort)].groupby('method').apply(
            lambda x: np.average(x[key], weights=x.survey_weight), include_groups=False)
        ax.barh(means.index, means, color=['#999999', '#0072B2', '#009E73', '#E69F00'][:len(means)])
        ax.set_title(label + ': lower is better')
    fig.tight_layout(); fig.savefig(root / 'test.png', dpi=180); plt.close(fig)
    weights = pd.DataFrame({'Behavioural outcome': optimize_weights.KEYS,
        'Selected weight': [selection['weights'][k] for k in optimize_weights.KEYS]}).to_html(index=False)
    relative = Path(application_report).resolve().relative_to(root.resolve())
    text = f'''<!doctype html><html><meta charset="utf-8"><title>Matching loss-weight optimization</title>
<style>body{{font:17px sans-serif;max-width:1150px;margin:35px auto;line-height:1.5}}img{{width:100%}}</style>
<h1>Matching loss-weight optimization</h1><p>We change only the seven behavioural loss weights.
Households are separated into fitting (about 60%), selection (20%) and untouched testing (20%).
Each neural fit retains its internal 10% early-stopping group and trains at least ten epochs.
The baseline and {selection['trials'] - 1} log-scale candidates use the same seeds and selection donors.
The weighted multivariate energy score selects the weights; the training loss is not compared across trials.
This finite search finds the best tested candidate, not a proven global optimum.</p>
<h2>Search: lower is better</h2><p>Points show selection errors; the line shows the best candidate found so far.</p>
<img src="search.png">{weights}
<h2>Independent household test: lower is better</h2><p>After selection, models are refitted on the 80% development group.
All four methods use that same donor group and {selection['draws']} draws per test person.
The figures use the common fully matched test cohort ({len(cohort):,} people), with survey weights.
These are predictive scores against observed diaries, not evidence that survey-wide frequencies are population truth.
Coverage per method: {html.escape(str(test.groupby('method').covered.mean().round(4).to_dict()))}.</p>
<img src="test.png"><h2>Full-survey replay and synthetic population</h2>
<p>The selected model is then refitted on the complete eligible survey. This separate application uses
50 donor-restricted replay draws and 10 synthetic-population assignments per method. Replay is reconstruction,
not independent testing. Population differences are sensitivity results, not known forecast errors.</p>
<p><a href="{html.escape(str(relative))}">Open the full national, canton, distance and activity-chain comparison</a></p></html>'''
    (root / 'report.html').write_text(text)


class OptimizedContext:
    def __init__(self, parent, matcher, output):
        self.parent, self.matcher, self.output = parent, matcher, output

    def config(self, name):
        return str(self.output) if name == 'analysis_path' else self.parent.config(name)

    def stage(self, name, *args, **kwargs):
        return self.matcher if name == repeat_experiments.MODEL else self.parent.stage(name, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.parent, name)
