"""Report count-head effects at the selected retrieval settings, without fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .probability_figure import BINS
from .metrics import clustered_interval

METRICS=[('brier_shop','Shopping probability'),('brier_leisure','Leisure probability'),
         ('energy','Joint behaviour'),('chain_brier','Activity/mode chain'),
         ('crps_count_shop','Shopping visits'),('crps_count_leisure','Leisure visits'),
         ('crps_distance_km','Daily distance'),('crps_trip_count','Total trips')]


def configure(context):
    context.stage('analysis.matching.counts_retrieval')


def execute(context):

    # Render all comparisons from saved donor-matching scores.

    return render(context.stage('analysis.matching.counts_retrieval'))


def render(directory):
    root=Path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    summary=pd.read_csv(root/'summary.csv')
    frames=[]
    for label,folder in manifest['models'].items():
        for seed in range(manifest['seeds']):
            frame=pd.read_csv(root/folder/'test/k200_t0.1'/f'seed_{seed}.csv.gz')
            frame['model'],frame['seed']=label,seed
            frames.append(frame)
    scores=pd.concat(frames,ignore_index=True)
    valid=scores.groupby('mz_id').covered.min()
    scores=scores[scores.mz_id.isin(valid[valid.eq(1)].index)]
    membership=pd.read_csv(root/'Current/membership.csv').set_index('mz_id')
    means=scores.groupby(['model','seed']).apply(lambda g:pd.Series({key:np.average(g[key],weights=g.survey_weight)
        for key,_ in METRICS}),include_groups=False)
    means.to_csv(root/'test_scores_by_seed.csv')
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for ax,(key,label) in zip(axes.flat,METRICS):
        for j,name in enumerate(manifest['models']):
            data=means.loc[name,key]
            ax.bar(j,data.mean(),color=['#777777','#0072B2'][j])
            ax.scatter(np.full(len(data),j),data,c='black',s=18)
        ax.set_title(label+'\nLower is better');ax.set_xticks([0,1],list(manifest['models']))
    fig.tight_layout();fig.savefig(root/'scores.png',dpi=180);plt.close(fig)
    intervals=[]
    rng=np.random.RandomState(519)
    for key,_ in METRICS:
        wide=scores.groupby(['mz_id','model'])[key].mean().unstack()
        weights=scores.groupby('mz_id').survey_weight.first().reindex(wide.index)
        estimate,low,high,_,_=clustered_interval(wide['Count head']-wide.Baseline,weights,
            membership.reindex(wide.index).household,rng,1000)
        intervals.append(dict(metric=key,difference=estimate,lower=low,upper=high))
    intervals=pd.DataFrame(intervals);intervals.to_csv(root/'paired_intervals.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,event in zip(axes,['shop','leisure']):
        for name in manifest['models']:
            frame=scores[scores.model.eq(name)].groupby('mz_id').agg(
                p=('prob_'+event,'mean'),y=('truth_'+event,'first'),w=('survey_weight','first'))
            codes=np.searchsorted(BINS[1:-1],frame.p,side='right');x=[];y=[]
            for j in range(6):
                group=frame[codes==j]
                if not group.empty:
                    x.append(np.average(group.p,weights=group.w));y.append(np.average(group.y,weights=group.w))
            ax.plot(x,y,'o-',label=name)
        ax.plot([0,1],[0,1],'k--',alpha=.5)
        ax.set(title=event.title(),xlabel='Mean predicted probability',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(root/'probabilities.png',dpi=180);plt.close(fig)
    (root/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Count-head matching at 200 neighbours</title>
<style>body{{font:17px sans-serif;max-width:1400px;margin:30px auto;line-height:1.5}}img{{width:100%}}</style>
<h1>Does the count head help with 200 neighbours and temperature 0.1?</h1>
<p>We reuse three baseline encoders and three counts-only encoders. The latter have one head predicting
five counts: all trips, shopping, leisure, education and work. Feature reconstruction is unchanged.
Both variants use 200 neighbours, temperature 0.1, identical household partitions and seeds,
and 100 donor draws per recipient. No neural retraining is performed. Test recipients were excluded
from fitting, but this split has already been examined in previous experiments: this is exploratory.</p>
<h2>Test scores: lower is better</h2><p>Bars average three fitting seeds; dots show each seed.
All scores use the common fully matched cohort ({len(valid[valid.eq(1)]):,} respondents).</p><img src="scores.png">
<h2>Differences from the baseline</h2><p>Negative differences favour the count head.
Pointwise 95% intervals resample households 1,000 times, after averaging each person's scores across fits.
Intervals condition on the fitted models and are not adjusted for multiple comparisons.</p>
{intervals.round(6).to_html(index=False)}
<h2>Shopping and leisure probability agreement</h2><p>Closer to the diagonal indicates better agreement.
For this display, probabilities are averaged over fitting seeds before common-bin aggregation.
Score comparisons instead average scores calculated for each fit separately.</p><img src="probabilities.png">
<p>No production defaults, larger replay experiment or paper results are changed.</p></html>''')
    return str(root/'report.html')
