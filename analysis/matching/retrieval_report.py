"""Plot the fixed-encoder retrieval experiment without additional matching."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .probability_figure import BINS


def configure(context):
    context.stage('analysis.matching.retrieval_experiment')


def execute(context):

    # Read completed candidate scores and render the selection/test comparisons.

    return render(context.stage('analysis.matching.retrieval_experiment'))


def render(directory):
    root=Path(directory)
    summary=pd.read_csv(root/'summary.csv')
    manifest=json.loads((root/'manifest.json').read_text())
    best=json.loads((root/'selection.json').read_text())['candidate']
    baseline=manifest['baseline']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,phase in zip(axes,('selection','test')):
        grid=summary[summary.phase.eq(phase)].groupby(['k','temperature']).event_brier_mean.mean().unstack()
        im=ax.imshow(grid,aspect='auto',cmap='viridis_r')
        ax.set(xticks=range(len(grid.columns)),xticklabels=grid.columns,yticks=range(len(grid.index)),yticklabels=grid.index,
               xlabel='Sampling temperature',ylabel='Number of neighbours',title=phase.title()+': event Brier (lower is better)')
        for i in range(len(grid)):
            for j in range(len(grid.columns)):
                ax.text(j,i,f'{grid.iloc[i,j]:.4f}',ha='center',va='center',color='white',fontsize=9)
        fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(root/'sweep.png',dpi=180);plt.close(fig)
    data={}
    for name in dict.fromkeys((baseline,best)):
        parts=[pd.read_csv(root/'test'/name/f'seed_{seed}.csv.gz') for seed in range(manifest['seeds'])]
        for seed,frame in enumerate(parts):frame['seed']=seed
        data[name]=pd.concat(parts,ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    binrows=[]
    for ax,event in zip(axes,('shop','leisure')):
        for name,frame in data.items():
            # Average stochastic predictions across fits, retaining one observed diary/person.
            grouped=frame.groupby('mz_id').agg(probability=('prob_'+event,'mean'),truth=('truth_'+event,'first'),weight=('survey_weight','first'))
            codes=np.searchsorted(BINS[1:-1],grouped.probability,side='right')
            x,y=[],[]
            for j in range(6):
                g=grouped[codes==j]
                if g.empty:continue
                px=np.average(g.probability,weights=g.weight);oy=np.average(g.truth,weights=g.weight)
                x.append(px);y.append(oy)
                binrows.append(dict(candidate=name,event=event,bin=j,predicted=px,observed=oy,people=len(g)))
            ax.plot(x,y,'o-',label=name)
        ax.plot([0,1],[0,1],'k--',alpha=.5)
        ax.set(title=event.title(),xlabel='Mean predicted probability',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(root/'probabilities.png',dpi=180);plt.close(fig)
    pd.DataFrame(binrows).to_csv(root/'probability_bins.csv',index=False)
    metrics=['event_brier_mean','brier_shop','brier_leisure','energy','chain_brier','crps_distance_km','crps_count_shop','crps_count_leisure']
    results=summary[summary.phase.eq('test')&summary.candidate.isin([baseline,best])].groupby('candidate')[metrics].mean()
    results.to_csv(root/'selected_test_summary.csv')
    (root/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Neighbourhood and temperature experiment</title>
<style>body{{font:17px sans-serif;max-width:1250px;margin:30px auto;line-height:1.5}}img{{width:100%}}</style>
<h1>Neighbourhood size and sampling temperature</h1><p>We reuse three fitted baseline encoders, with no neural retraining.
Only neighbour count and sampling temperature change. Each recipient has 50 donor draws.
Households, seeds, compatibility rules and behavioural weights stay fixed. Selection encoders were fitted on 60%
of households; test encoders on 80%. This follow-up reuses a previously examined split, so it is exploratory.</p>
<h2>Which settings improve event probabilities?</h2><p>Lower shopping/leisure Brier scores are better.
These scores assess probability accuracy, combining agreement with discrimination. Selection uses the mean of both
event scores, not appearance of the curve. Chosen setting: <b>{best}</b>; baseline: <b>{baseline}</b>.
The test grid is diagnostic and does not select settings. Each grid has its own colour scale; compare numbers.</p>
<img src="sweep.png"><h2>Do gains extend to other outcomes?</h2><p>Test scores average over the three fitted encoders.
Lower is better for every score. Wider pools can improve probability agreement while weakening behavioural similarity.</p>
{results.round(5).to_html()}<h2>Shopping and leisure probabilities</h2><p>Closer to the diagonal means better agreement.
Points use common bins, with actual mean probabilities on the x axis and observed frequencies on the y axis.
For this display, each person's probability is averaged over three fitting seeds before binning; the score table
instead averages scores computed separately for each fit. The two summaries serve different purposes.</p>
<img src="probabilities.png"><p>No synthesis model or paper results are replaced by this experiment.
<a href="summary.csv">All scores and coverage</a> · <a href="manifest.json">Settings</a></p>''')
    return str(root/'report.html')
