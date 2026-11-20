"""Auditable probability-agreement figure from fixed saved donor assignments."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import clustered_interval

EVENTS = [('work','Any work',['work','work_secondary']),
          ('education','Any education',['education','education_secondary']),
          ('shop','Any shopping',['shop']),('leisure','Any leisure',['leisure']),
          ('work_secondary','Secondary work',['work_secondary'])]
LABELS = {'V1':'RHDM','V2':'BSLM','V3':'OHDM'}
COLORS = {'V1':'#df8a2e','V2':'#269568','V3':'#427fc0'}
BINS = np.array([0,.1,.2,.3,.5,.75,1.])


def merge_sparse_bins(probabilities, minimum=20, initial=BINS):
    """Coarsen common boundaries using prediction counts only, never outcomes.

    Every method must have at least `minimum` people in every resulting bin.
    Empty bins are merged too. Ties favour the narrower adjacent interval,
    then the left interval, so results are deterministic.
    """
    if minimum < 1 or any(len(values) < minimum for values in probabilities):
        raise ValueError("Each method needs at least the minimum number of respondents")
    edges = np.asarray(initial, dtype=float).copy()
    while len(edges) > 2:
        counts = np.array([np.bincount(np.searchsorted(edges[1:-1], values, side="right"),
                            minlength=len(edges)-1) for values in probabilities])
        smallest = counts.min(axis=0)
        if smallest.min() >= minimum:
            break
        j = int(smallest.argmin())
        if j == 0:
            boundary = 1
        elif j == len(edges)-2:
            boundary = j
        else:
            boundary = j if edges[j+1]-edges[j-1] <= edges[j+2]-edges[j] else j+1
        edges = np.delete(edges, boundary)
    return edges


def generate(results, trips_path, paper, min_people=20):
    root, out = Path(results), Path(paper)
    people = pd.read_csv(root/'microcensus/respondents.csv')
    ids = pd.Index(people.mz_id)
    trips = pd.read_pickle(trips_path)[0]
    eligible = trips.loc[trips.person_id.isin(ids)].sort_values(['person_id','trip_id'])
    digest = hashlib.sha256(pd.util.hash_pandas_object(eligible,index=True).to_numpy().tobytes()).hexdigest()
    assert digest == json.loads((root/'experiment_identity.json').read_text())['trips']
    truth = np.column_stack([ids.isin(trips.loc[trips.purpose.isin(purposes)|trips.origin_purpose.isin(purposes),'person_id'])
                             for _,_,purposes in EVENTS])
    rows = []
    rng = np.random.RandomState(518745)
    predictions = {}
    for name in LABELS:
        probabilities = np.full(truth.shape,np.nan)
        seen = np.zeros(len(ids),dtype=int)
        for fold in range(3):
            with np.load(root/f'microcensus/fold_{fold}/{name}_donor_rows.npz') as saved:
                target = ids.get_indexer(saved['test_ids'])
                assert (target>=0).all() and (saved['rows']>=0).all()
                donors = ids.get_indexer(saved['source_ids'][saved['rows']].ravel()).reshape(saved['rows'].shape)
                assert (donors>=0).all()
                probabilities[target] = truth[donors].mean(1)
                seen[target] += 1
        assert (seen==1).all() and np.isfinite(probabilities).all()
        predictions[name] = probabilities
    event_bins = {}
    for col,(event,_,_) in enumerate(EVENTS):
        edges = merge_sparse_bins([predictions[name][:,col] for name in LABELS], min_people)
        event_bins[event] = edges.tolist()
        for name in LABELS:
            probabilities = predictions[name]
            codes = np.searchsorted(edges[1:-1],probabilities[:,col],side='right')
            for j in range(len(edges)-1):
                mask=codes==j
                w=people.survey_weight.to_numpy()[mask]
                if mask.any():
                    observed,low,high,households,ess=clustered_interval(truth[mask,col].astype(float),w,
                        people.household_group.to_numpy()[mask],rng,500)
                    predicted=np.average(probabilities[mask,col],weights=w)
                else:
                    observed=low=high=predicted=np.nan;households=ess=0
                rows.append(dict(algorithm=name,event=event,bin=j,lower=edges[j],upper=edges[j+1],
                    midpoint=(edges[j]+edges[j+1])/2,predicted=predicted,observed=observed,
                    observed_lower=low,observed_upper=high,people=int(mask.sum()),weight=float(w.sum()),households=households,ess=ess))
    table=pd.DataFrame(rows)
    assert table.people.ge(min_people).all()
    table.to_csv(out/'tables/latest_probability_agreement.csv',index=False)
    (out/'tables/latest_probability_bins.json').write_text(json.dumps(dict(minimum_people_per_method=min_people,
        rule='Adjacent merging based only on prediction counts; boundaries shared by methods within each event',
        initial=BINS.tolist(),by_event=event_bins),indent=2))
    fig=plt.figure(figsize=(13,9))
    outer=fig.add_gridspec(2,3,wspace=.32,hspace=.45)
    legend=None
    for cell,(event,title,_) in zip(outer,EVENTS):
        inner=cell.subgridspec(2,1,height_ratios=[3,1],hspace=.1)
        ax=fig.add_subplot(inner[0]);hist=fig.add_subplot(inner[1],sharex=ax)
        for a in LABELS:
            data=table.loc[table.algorithm.eq(a)&table.event.eq(event)&table.people.gt(0)]
            ax.plot(data.predicted,data.observed,'o-',color=COLORS[a],label=LABELS[a],markersize=4,lw=1)
            ax.vlines(data.predicted,data.observed_lower,data.observed_upper,color=COLORS[a],lw=1)
            mass=100*data.weight/people.survey_weight.sum()
            hist.stairs(mass,np.r_[data.lower.to_numpy(),data.upper.iloc[-1]],color=COLORS[a],lw=1.4)
        ax.plot([0,1],[0,1],'k--',alpha=.5,lw=1)
        ax.set(title=title+'\nBins: '+', '.join(f'{x:g}' for x in event_bins[event]),ylabel='Observed frequency',xlim=(0,1),ylim=(0,1))
        ax.tick_params(labelbottom=False);ax.grid(alpha=.15)
        hist.set(xlabel='Mean predicted probability / probability bins',ylabel='People (%)',xticks=[0,.25,.5,.75,1])
        hist.tick_params(labelsize=8);hist.grid(axis='y',alpha=.15)
        legend=ax.get_legend_handles_labels()
    key=fig.add_subplot(outer[1,2]);key.axis('off')
    key.legend(*legend,loc='upper left',fontsize=11)
    key.text(0,.65,'Closer to the diagonal: better agreement.\n\nPoints: survey-weighted bin means.\nLines: connect populated bins only.\nIntervals: 95% household bootstrap.\nLower panels: weighted population shares.\n\nAdjacent bins merged until each\nmethod has at least 20 people per bin.\nBoundaries shared within each panel.\nMerging uses predictions, not outcomes.',va='top',fontsize=10,linespacing=1.5)
    fig.savefig(out/'figures/latest_probability_agreement.pdf',bbox_inches='tight')
    plt.close(fig)
    return table
