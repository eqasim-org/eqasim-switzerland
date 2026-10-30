"""Matched survey and population comparisons for the distance-loss weight sweep."""
import gc
import html
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cohorts import population, survey, PT_CLASSES
from figures import ecdf, save
from metrics import mean_ci, summaries, standardized_contrast
from replay import prepare_inputs, select_cohort, replay
from loss_sweep_tools import FrozenCache, write_json

LOG = logging.getLogger(__name__)
BINS = [.01, .1, .25, .5, 1., 1.5, 2., 3., 5., 20., np.inf]
BIN_LABELS = ['10–100 m', '100–250 m', '250–500 m', '0.5–1 km', '1–1.5 km',
              '1.5–2 km', '2–3 km', '3–5 km', '5–20 km', '>20 km']
COLORS = ['#0072B2', '#009E73', '#E69F00', '#CC79A7', '#D55E00', '#56B4E9', '#666666']



def display_trial(label):
    if label == 'survey':
        return 'Microcensus'
    if label.startswith('short '):
        return 'Short ' + label.partition(' ')[2]
    return 'Global weight ' + label


def replay_errors(predictions, observed, inputs):
    target = observed[['person_id', 'trip_id', 'destination_x', 'destination_y']].rename(
        columns={'destination_x': 'observed_x', 'destination_y': 'observed_y'})
    target = target.merge(inputs[['person_id', 'trip_id', 'target_distance']],
                          on=['person_id', 'trip_id'], validate='one_to_one')
    result = predictions.merge(target, on=['person_id', 'trip_id'], validate='many_to_one')
    result['observed_km'] = result.target_distance / 1000
    result['signed_error_km'] = result.distance_km - result.observed_km
    result['absolute_error_km'] = result.signed_error_km.abs()
    result['spatial_error_km'] = np.hypot(result.destination_x-result.observed_x,
                                         result.destination_y-result.observed_y) / 1000
    result['distance_band'] = pd.cut(result.observed_km, BINS, labels=BIN_LABELS,
                                     include_lowest=True)
    return result


def error_summary(errors):
    rows = []
    for trial, frame in errors.groupby('trial', sort=False):
        nonstationary = frame.loc[frame.observed_km >= .01]
        scopes = [('All nonstationary', nonstationary)]
        scopes.extend((str(band), group) for band, group in nonstationary.groupby('distance_band', observed=True))
        for label, group in scopes:
            row = dict(trial=trial, distance_band=label,
                       people=group.person_id.nunique(),
                       trips=group[['person_id', 'trip_id']].drop_duplicates().shape[0],
                       observed_km=float(np.average(group.observed_km, weights=group.weight)),
                       predicted_km=float(np.average(group.distance_km, weights=group.weight)))
            for column in ['signed_error_km', 'absolute_error_km', 'spatial_error_km']:
                mean, _, low, high = mean_ci(group, column)
                row[column], row[column+'_low'], row[column+'_high'] = mean, low, high
            rows.append(row)
    return pd.DataFrame(rows)


def paired_improvement(errors):
    # Average draws before pairing so repeated predictions are not independent people.

    trips = errors.groupby(['trial', 'person_id', 'trip_id'], observed=True).agg(
        error=('absolute_error_km', 'mean'), weight=('weight', 'sum'),
        observed_km=('observed_km', 'first')).reset_index()
    reference = trips.loc[trips.trial.eq('0.1'), ['person_id', 'trip_id', 'error']].rename(columns={'error':'baseline_error'})
    rows = []
    for trial, frame in trips.loc[trips.trial.ne('0.1')].groupby('trial', sort=False):
        frame = frame.merge(reference, on=['person_id','trip_id'], validate='one_to_one')
        frame['improvement_km'] = frame.baseline_error-frame.error
        for label, group in [('All nonstationary', frame.loc[frame.observed_km >= .01]),
                             ('10 m–1.5 km', frame.loc[frame.observed_km.between(.01,1.5)]),
                             ('>1.5 km', frame.loc[frame.observed_km > 1.5])]:
            if group.empty:
                continue
            mean, _, low, high = mean_ci(group, 'improvement_km')
            rows.append(dict(trial=trial, distance_band=label, improvement_km=mean,
                             low_km=low, high_km=high, people=group.person_id.nunique(), trips=len(group)))
    return pd.DataFrame(rows)


def render_figures(datasets, summary, errors, directory):
    labels = list(datasets)[1:]
    if len(labels) > len(COLORS):
        raise ValueError('The report palette must include every trial')

    # Compare complete population distributions, with a close view of short distances.

    fig, axes = plt.subplots(1,2,figsize=(8,3.2))
    for ax in axes:
        for color, (label, frame) in zip(['#30343B', *COLORS], datasets.items()):
            ecdf(ax,frame.distance_km,frame.weight,display_trial(label),color)
        ax.axvline(1.5,color='#999',linestyle=':',linewidth=1)
        ax.set(ylabel='Weighted cumulative share',xlabel='Secondary-leg distance (km)',ylim=(0,1))
    axes[0].set(xscale='log',xlim=(.01,200))
    axes[1].set(xlim=(0,3))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig,directory,'01_population_distance_distributions')

    # Show signed and absolute replay errors against identical observed distance bands.

    fig, axes = plt.subplots(1,2,figsize=(9,3.4))
    for color,label in zip(COLORS,labels):
        frame=summary.loc[summary.trial.eq(label)].set_index('distance_band').reindex(BIN_LABELS)
        for ax,column in zip(axes,['signed_error_km','absolute_error_km']):
            ax.plot(range(len(BIN_LABELS)),frame[column],marker='o',markersize=3,color=color,label=display_trial(label))
    for ax in axes:
        ax.axvline(4.5,color='#999',linestyle=':',linewidth=1)
        ax.set(xticks=range(len(BIN_LABELS)),xticklabels=BIN_LABELS,xlabel='Observed trip distance')
        ax.tick_params(axis='x',rotation=55)
        ax.grid(axis='y',color='#ddd',linewidth=.5)
    axes[0].axhline(0,color='#555',linewidth=.7)
    axes[0].set_ylabel('Mean predicted − observed distance (km)')
    axes[1].set_ylabel('Mean absolute distance error (km)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig,directory,'02_replay_errors_by_observed_distance')

    # Inspect local errors at the expert switch without letting long-trip scales hide them.

    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for color,label in zip(COLORS,labels):
        frame=summary.loc[summary.trial.eq(label)].set_index('distance_band').reindex(BIN_LABELS[:7])
        axes[0].plot(range(7),frame.signed_error_km,marker='o',markersize=3,color=color,label=display_trial(label))
        axes[1].plot(range(7),frame.absolute_error_km,marker='o',markersize=3,color=color,label=display_trial(label))
    for ax in axes:
        ax.axvline(4.5,color='#999',linestyle=':',linewidth=1)
        ax.set(xticks=range(7),xticklabels=BIN_LABELS[:7],xlabel='Observed trip distance')
        ax.tick_params(axis='x',rotation=50)
        ax.grid(axis='y',color='#ddd',linewidth=.5)
    axes[0].axhline(0,color='#555',linewidth=.7)
    axes[0].set_ylabel('Mean signed distance error (km)')
    axes[1].set_ylabel('Mean absolute distance error (km)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig,directory,'03_short_range_and_threshold_errors')

    fig, axes=plt.subplots(1,2,figsize=(8,3.2),sharey=True)
    for ax, car in zip(axes,['No car access','Car access']):
        for color,(label,frame) in zip(['#30343B',*COLORS],datasets.items()):
            means=[]
            for quality in PT_CLASSES:
                group=frame.loc[frame.car_group.eq(car)&frame.home_pt.eq(quality)]
                means.append(np.average(group.distance_km,weights=group.weight) if len(group) else np.nan)
            ax.plot(range(5),means,marker='o',markersize=3,color=color,label=display_trial(label))
        ax.set(title=car,xticks=range(5),xticklabels=PT_CLASSES,xlabel='Residential public-transport quality')
    axes[0].set_ylabel('Mean secondary-leg distance (km)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig,directory,'04_car_access_and_residential_accessibility')


def compare(plan):
    from pyproj import Transformer
    from run import replay_input_fingerprint

    output=Path(plan['output'])
    tables=output/'tables'
    tables.mkdir(exist_ok=True)
    settings=yaml.safe_load(Path(__file__).with_name('comparison.yml').read_text())
    transform=Transformer.from_crs(4326,2056,always_xy=True)
    for region in settings['regions'].values():
        x,y=transform.transform(*region['center_lon_lat'])
        x,y=round(x/500)*500,round(y/500)*500
        width=region['half_width_m']
        region['bounds']=[x-width,y-width,x+width,y+width]
    baseline=FrozenCache(plan)
    baseline.verify()
    observed,audit=survey(baseline)
    cohort=select_cohort(observed,settings['regions'],plan['replay_people'],baseline.config('random_seed'))
    if cohort.empty:
        raise ValueError('No common city-origin survey cohort')
    inputs=prepare_inputs(baseline,cohort)
    observed.to_parquet(output/'survey.parquet',index=False)
    cohort.to_parquet(output/'replay_observed.parquet',index=False)
    datasets={'survey':observed}
    replays=[]
    manifest=dict(baseline_root=baseline.root, source_fingerprint=plan['source_fingerprint'],
                  common_input_sha256=replay_input_fingerprint(inputs),
                  people=int(cohort.person_id.nunique()),trips=len(cohort),draws=plan['draws'],runs={},survey_audit=audit)
    detour=None

    # Reuse the same survey people, inputs, seeds and draws for every fitted model.

    for weight in [.1,*plan['weights']]:
        label=f'{weight:g}'
        cache=FrozenCache(plan,weight)
        cache.variant.mkdir(parents=True,exist_ok=True)
        LOG.info('Comparing distribution weight=%s',label)
        frame,audit=population(cache)
        frame.to_parquet(cache.variant/'population.parquet',index=False)
        datasets[label]=frame
        predictions,detour=replay(cache,cohort,inputs,plan['draws'],cache.config('random_seed'),min(plan['threads'],4),detour)
        keys = ['person_id', 'trip_id', 'draw']
        if len(predictions) != len(cohort) * plan['draws'] or predictions.duplicated(keys).any():
            raise ValueError('Replay did not produce one prediction per common trip and draw')
        expected = set(cohort[['person_id', 'trip_id']].itertuples(index=False, name=None))
        actual = set(predictions[['person_id', 'trip_id']].itertuples(index=False, name=None))
        if expected != actual:
            raise ValueError('Replay differs from the common survey cohort')
        predictions.to_parquet(cache.variant/'replay.parquet',index=False)
        predictions['trial']=label
        replays.append(replay_errors(predictions,cohort,inputs))
        manifest['runs'][label]=dict(audit=audit,provenance=cache.provenance())
        cache.verify()
    del detour
    gc.collect()
    note=('All four distance-enabled experts use the indicated distribution-loss weight. The population, destinations, '
          'survey inputs and seed are frozen to the completed 0.1 run. The 1.5 km expert switch and final distance '
          'weighting are unchanged. Replay uses observed origins and identical people/draw seeds; it is a fitting-data '
          'diagnostic, not independent held-out validation. Positive paired improvement means lower absolute error than 0.1. '
          'Stationary trips below 10 m are excluded from replay error tables. Person-cluster intervals omit model-fitting uncertainty. '
          'Loss values across different weights are not directly comparable; evaluate distance and behavioral metrics instead.')
    write_comparison_report(datasets, pd.concat(replays, ignore_index=True), manifest, output, note)


def write_comparison_report(datasets, errors, manifest, output, note, trial_settings=None, training_summary=None):
    """Render old and focused trials together without rerunning saved predictions."""
    output = Path(output)
    tables = output / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    errors.to_parquet(output/'replay_errors.parquet',index=False)
    summary=error_summary(errors)
    summary.to_csv(tables/'replay_errors_by_distance.csv',index=False)
    improvement=paired_improvement(errors)
    improvement.to_csv(tables/'paired_improvement_over_0.1.csv',index=False)
    population_summary=summaries(datasets)
    population_summary.to_csv(tables/'population_distance_summary.csv',index=False)
    for name,dimension,low,high,strata in [
        ('car','car_group','No car access','Car access',['purpose','age_group','home_pt_tier']),
        ('accessibility','home_pt_tier','Low (D/none)','High (A/B)',['purpose','age_group','car_group'])]:
        standardized_contrast(datasets,dimension,low,high,strata,10).to_csv(tables/f'standardized_{name}_contrast.csv',index=False)
    render_figures(datasets,summary,errors,output/'figures')
    write_json(output/'comparison_manifest.json',manifest)
    overview=population_summary.loc[population_summary.dimension.eq('all'),['dataset','trips','persons','mean_km','median_km','short_share','long_share_20km']]
    document=['<!doctype html><html><head><meta charset="utf-8"><title>Secondary NN loss-weight comparison</title>',
              '<style>body{font:15px/1.5 system-ui;max-width:1150px;margin:35px auto;padding:20px}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd}</style></head><body>',
              '<h1>Secondary NN distribution-loss weight comparison</h1><p>'+html.escape(note)+'</p>',
              '<h2>Population distances</h2>',overview.to_html(index=False,float_format=lambda x:f'{x:.3f}'),
              '<h2>Paired improvement over weight 0.1</h2>',improvement.to_html(index=False,float_format=lambda x:f'{x:.3f}'),
              '<h2>Replay errors by observed distance</h2>',summary.to_html(index=False,float_format=lambda x:f'{x:.3f}')]
    if trial_settings is not None:
        trial_settings.to_csv(tables / 'trial_settings.csv', index=False)
        document.extend(['<h2>Loss settings by expert</h2>', trial_settings.to_html(index=False)])
    if training_summary is not None:
        training_frame = training_summary if isinstance(training_summary, pd.DataFrame) else pd.DataFrame([training_summary])
        training_frame.to_csv(tables / 'training_summary.csv', index=False)
        document.extend(['<h2>New short-expert training</h2>', training_frame.to_html(index=False),
            '<p>The minimum is an exploration period before stopping. An earlier epoch can still be selected if its cross-validated score is best.</p>'])
    for image in sorted((output/'figures').glob('*.png')):
        document.extend(['<h2>'+html.escape(image.stem.replace('_',' '))+'</h2>',
                         f'<img src="figures/{image.name}"><p><a href="figures/{image.stem}.pdf">PDF</a> · <a href="figures/{image.stem}.svg">SVG</a></p>'])
    document.append('<p><a href="comparison_manifest.json">Provenance</a> · <a href="tables/replay_errors_by_distance.csv">Replay error table</a> · <a href="tables/population_distance_summary.csv">Population table</a></p></body></html>')
    (output/'report.html').write_text('\n'.join(document))
    LOG.info('Sweep report: %s',output/'report.html')
