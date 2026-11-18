"""Analysis-only neighbourhood/temperature sweep with fixed held-out encoders."""
import json
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from synthesis.population.matching.V2 import model as model_stage
from synthesis.population.matching.V2.configuration import MatchingOptions
from synthesis.population.matching.V2.inputs import prepare_inputs
from synthesis.population.matching.V2.wrappers import DiaryMatcher
from synthesis.population.matching.V2.donors import DonorSelector
from synthesis.population.matching import opening_hours
from .metrics import diary_outcomes, outcome_scale, score_ensembles

PRIOR='analysis.matching.ablation_experiment'
NEIGHBOURS=(10,20,50,100,200)
TEMPERATURES=(.01,.1,.5,2.)


def configure(context):
    model_stage.configure(context)
    context.config('analysis_path')
    context.stage(PRIOR)


def execute(context, *, model_variant="Current", neighbours=None, temperatures=None,
            output_name="retrieval_sweep", draws=None):

    # Reuse the three baseline encoders and exact household splits from the completed experiment.

    neighbours = NEIGHBOURS if neighbours is None else tuple(neighbours)
    temperatures = TEMPERATURES if temperatures is None else tuple(temperatures)
    prior=Path(context.stage(PRIOR))
    root=Path(context.config('analysis_path'))/'matching'/output_name
    root.mkdir(parents=True,exist_ok=True)
    membership=pd.read_csv(prior/'membership.csv')
    old=json.loads((prior/'manifest.json').read_text())
    old['draws'] = old['draws'] if draws is None else int(draws)
    source,population,trips=prepare_inputs(context,context.stage('data.constants'))
    source,_=opening_hours.add_regions(context,source,population)
    if not np.array_equal(source.mz_id,membership.mz_id):
        raise ValueError('Prepared donor identity/order differs from fitted encoders')
    membership.to_csv(root/'membership.csv',index=False)
    options=replace(MatchingOptions.from_context(context),overwrite=False)
    if options.training['loss_weights']!=old['weights']:
        raise ValueError('Configured behavioural weights differ from the baseline experiment')
    observed,labels=diary_outcomes(source,trips)
    truth=np.column_stack([source.mz_id.isin(trips.loc[trips.purpose.eq(p)|trips.origin_purpose.eq(p),'person_id'])
                           for p in ('shop','leisure')])
    manifest=dict(seed=old['seed'],seeds=old['seeds'],draws=old['draws'],neighbours=neighbours,model_variant=model_variant,
        temperatures=temperatures,baseline='k20_t0.5',selection='Mean shopping/leisure event Brier',
        interpretation='Fixed encoders; follow-up on a previously examined household split; exploratory')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]

    # Vary retrieval alone, using common seeds and fifty draws for every candidate.

    for phase in ('selection','test'):
        donor_rows=np.flatnonzero(membership.split.eq('fit') if phase=='selection' else membership.split.ne('test'))
        recipient_rows=np.flatnonzero(membership.split.eq(phase))
        donors=source.iloc[donor_rows].reset_index(drop=True)
        target=source.iloc[recipient_rows].reset_index(drop=True).assign(person_id=source.iloc[recipient_rows].mz_id.to_numpy())
        _,scaled,_=outcome_scale(observed,donor_rows,source.person_weight.to_numpy())
        for repetition in range(old['seeds']):
            payload=torch.load(prior/phase/model_variant.replace(' ', '_')/str(repetition)/'model.pt',map_location='cpu',weights_only=False)
            matcher=DiaryMatcher(options)
            matcher.model_,matcher.processor_,matcher.training_report_=payload['model'],payload['processor'],payload['report']
            matcher.donors_=donors
            matcher.donor_embeddings_=model_stage.encode(matcher.model_,matcher.processor_.transform(donors))
            for k in neighbours:
                for temperature in temperatures:
                    name=f'k{k}_t{temperature:g}'
                    directory=root/phase/name
                    directory.mkdir(parents=True,exist_ok=True)
                    path=directory/f'seed_{repetition}.csv.gz'
                    started=time.perf_counter()
                    if path.exists():
                        scores=pd.read_csv(path)
                    else:
                        matcher.selector_=DonorSelector(**{**options.retrieval,'top_k':k,'temperature':temperature})
                        selected=matcher.match(target,n_draws=old['draws'],random_seed=old['seed']+10000+repetition).donor_rows
                        global_rows=np.where(selected>=0,donor_rows[np.maximum(selected,0)],-1)
                        scores=score_ensembles(observed,labels,recipient_rows,global_rows,scaled)
                        valid=(global_rows>=0).all(1)
                        for col,event in enumerate(('shop','leisure')):
                            probability=truth[np.maximum(global_rows,0),col].mean(1)
                            scores['prob_'+event]=np.where(valid,probability,np.nan)
                            scores['truth_'+event]=truth[recipient_rows,col].astype(int)
                            scores['brier_'+event]=np.where(valid,(probability-truth[recipient_rows,col])**2,np.nan)
                        keep=['covered','energy','chain_brier','crps_distance_km','crps_trip_count',
                              'crps_count_shop','crps_count_leisure','prob_shop','truth_shop','brier_shop',
                              'prob_leisure','truth_leisure','brier_leisure']
                        scores=scores[keep].copy()
                        scores['mz_id']=source.iloc[recipient_rows].mz_id.to_numpy()
                        scores['survey_weight']=source.iloc[recipient_rows].person_weight.to_numpy()
                        temporary=path.with_name(path.name+'.tmp')
                        scores.to_csv(temporary,index=False,compression='gzip');temporary.replace(path)
                    record=dict(phase=phase,seed=repetition,candidate=name,k=k,temperature=temperature,
                                seconds=time.perf_counter()-started,coverage=float(np.average(scores.covered,weights=scores.survey_weight)))
                    for key in ('energy','chain_brier','crps_distance_km','crps_trip_count','crps_count_shop','crps_count_leisure','brier_shop','brier_leisure'):
                        record[key]=float(np.average(scores[key],weights=scores.survey_weight))
                    record['event_brier_mean']=(record['brier_shop']+record['brier_leisure'])/2
                    records.append(record)
                    pd.DataFrame(records).to_csv(root/'summary.csv',index=False)
                    print('Finished retrieval candidate',record,flush=True)
    summary=pd.DataFrame(records)
    eligible=summary[summary.phase.eq('selection')].groupby('candidate').filter(lambda x: x.coverage.eq(1).all())
    best=eligible.groupby('candidate').event_brier_mean.mean().idxmin()
    (root/'selection.json').write_text(json.dumps(dict(candidate=best,criterion=manifest['selection']),indent=2))
    return str(root)
