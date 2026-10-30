#!/usr/bin/env python3
"""Compare secondary locations from existing Synpp caches; never run simulations."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import html
import json
import logging
from pathlib import Path
import sys
import uuid

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

from cache_io import RunCache
from cohorts import population, survey, LABELS
from metrics import summaries, standardized_contrast, activity_spaces, transition_table, paired_replay_summary, selector_diagnostics

LOG = logging.getLogger('secondary_comparison')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def caches_for(settings):
    caches = []
    for path in settings['configs']:
        caches.append(RunCache(REPOSITORY / path))
    if len({c.kind for c in caches}) != len(caches):
        raise ValueError('Configuration list must contain distinct secondary-location methods')
    return caches


def prepare(caches, output):
    manifest = {'generation': str(uuid.uuid4()), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Weekday domestic secondary arrival legs, age >=6; border-crossing persons excluded from population', 'runs': {}, 'audit': {}}
    for cache in caches:
        LOG.info('Extracting %s secondary-trip cohort', cache.kind)
        frame, audit = population(cache)
        frame.to_parquet(output / f'population_{cache.kind}.parquet', index=False)
        manifest['audit'][cache.kind] = audit
        manifest['runs'][cache.kind] = cache.provenance()
        LOG.info('%s: %s', cache.kind, audit)
        del frame
        gc.collect()
    reference = next((c for c in caches if c.kind == 'rda'), caches[0])
    frame, audit = survey(reference)
    frame.to_parquet(output / 'survey.parquet', index=False)
    manifest['audit']['survey'] = audit
    manifest['survey_source'] = reference.provenance()
    write_json(output / 'prepared_manifest.json', manifest)
    return manifest



def replay_input_fingerprint(inputs):
    # Hash array-valued chain features by content without changing replay inputs.

    frame = inputs.copy()
    for column in ['activity_chain', 'chain_state']:
        if column in frame:
            frame[column] = frame[column].map(tuple)
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def replay_models(caches, settings, output, prepared):
    from replay import select_cohort, replay, prepare_inputs
    observed = pd.read_parquet(output / 'survey.parquet')
    cohort = select_cohort(observed, settings['regions'], settings['max_replay_people'], settings['seed'])
    if cohort.empty:
        raise ValueError('No survey people with trips originating in the selected regions')
    cohort.to_parquet(output / 'replay_observed.parquet', index=False)
    reference = next((c for c in caches if c.kind == 'rda'), caches[0])
    inputs = prepare_inputs(reference, cohort)
    input_sha256 = replay_input_fingerprint(inputs)
    predictions, detour = [], None
    provenance = {}
    for cache in caches:
        if cache.kind == 'rda':
            continue
        frame, detour = replay(cache, cohort, inputs, settings['survey_draws'], settings['seed'], settings['threads'], detour)
        predictions.append(frame)
        provenance[cache.kind] = cache.provenance()
    del detour
    gc.collect()
    if not predictions:
        raise ValueError('Survey replay requires the two NN configurations')
    pd.concat(predictions, ignore_index=True).to_parquet(output / 'replay_predictions.parquet', index=False)
    write_json(output / 'replay_manifest.json', dict(
        generation=prepared['generation'], seed=settings['seed'], draws=settings['survey_draws'],
        max_replay_people=settings['max_replay_people'], selected_people=int(cohort.person_id.nunique()),
        selected_trips=len(cohort), regions=settings['regions'], runs=provenance,
        common_survey_inputs=reference.provenance(), common_input_sha256=input_sha256,
        paired_controls='Identical survey demographics, income imputation, home/work coordinates, origins, purposes and observed chain features for both NNs',
        interpretation='Observed-origin diagnostic replay on existing survey records; not independent held-out validation'))


CAPTIONS = {
'01_distance_distributions_and_purpose': 'Weighted cumulative distributions and mean secondary-arrival-leg distances by purpose. Microcensus is a behavioral reference, not a calibration target. Distances are straight-line LV95 distances, not routed distances. Intervals are approximate 95% person-cluster intervals for mean distances; they exclude simulation-seed uncertainty and the full survey sampling design. The dotted line marks 1.5 km, not a fitted optimum.',
'02_car_access_and_residential_accessibility': 'Distance gradients by residential ÖV class and car access. Class A is the highest accessibility; no class denotes locations outside the A–D polygons. Car access harmonizes survey always/sometimes (0/1) versus never (2) with the synthetic 1/0 indicator; it does not denote vehicle ownership or guarantee a driving licence. Purpose and age composition can differ in these raw profiles.',
'03_standardized_behavioral_contrasts': 'Behavioral associations standardized to a shared microcensus covariate mix. Car contrasts adjust for purpose, age group and home accessibility tier; accessibility contrasts adjust for purpose, age group and car access. Only strata with at least the configured number of distinct people in both contrasted groups of every dataset are retained. The table reports retained weighted support. Person-cluster influence-function intervals treat reference stratum weights as fixed. These are descriptive associations, not causal effects.',
'04_home_to_destination_accessibility': 'Residential-to-destination accessibility transition probabilities, separately by car access. Each home-class row sums to 100% before independent rounding; absent cells are blank rather than evidence of a zero probability. Every panel uses the same color scale. Destination accessibility is evaluated from the same cached ÖV polygon source.',
'05_person_level_activity_spaces': 'Secondary-destination radius of gyration and repeated-visit share among people with at least two secondary arrival legs. Radius is computed around each person’s mean secondary destination, counting repeated visits. Revisited cells use a fixed 250 m LV95 grid. Each person contributes once with their person weight, rather than once per trip. These statistics describe secondary destinations, not a complete daily activity space.',
'06_observed_origin_survey_replay': 'Saved NN models evaluated on exactly the same sampled survey people and trips, resetting each trip to its observed origin and using observed chain features. Multiple destination draws are averaged for paired NN differences and have total weight equal to the observed trip weight. Histograms pool differences beyond ±40 km into the outer bins. The same income imputation and other observed inputs are held fixed across both NNs. Existing survey records may have been used to fit the models: this is a conditional diagnostic, not an out-of-sample performance estimate. It does not reproduce sequential synthetic-population rollout.',
'08_model_selection_diagnostic': 'Observed-origin replay diagnostic: share of nonstationary survey trips routed to the short-range branch, by observed distance and car access. The distance-input pipeline may use direct distance-conditioned facility sampling in that branch; consult the recorded run policy. The dotted separator marks 1.5 km. The learned selector does not receive the plotted trip distance as an input; distance is used here only for retrospective stratification. This is not held-out classifier performance. Lines average fitted local-expert probabilities when available, with historical decisions as a fallback. The table counts actual weighted draws; routing is probabilistic when distance is absent. Stationary trips below 10 m are excluded.',
'07_map': 'Spatial distribution of secondary destinations for the indicated purpose and city-origin cohort. Top: the same sampled microcensus trips, observed destinations versus observed-origin NN replay. Bottom: independently generated synthetic populations, RDA versus the two NNs. Every panel uses identical LV95 extent, 500 m cells and a shared square-root color normalization. Cell values are weighted destination mass per 1,000 eligible city-origin trips; out-of-frame destinations remain in the denominator and their visible share is annotated. Survey draw weights sum to the observed trip weight. Rows compare distributional patterns rather than identical population compositions. Zero cells are transparent. Roads, lakes and municipal boundaries are local inputs; no tile server is contacted.'}


def figures_and_report(caches, settings, output, prepared):
    import figures
    datasets = {'survey': pd.read_parquet(output / 'survey.parquet')}
    datasets.update({cache.kind: pd.read_parquet(output / f'population_{cache.kind}.parquet') for cache in caches})
    tables, images = output / 'tables', output / 'figures'
    tables.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    summary = summaries(datasets)
    summary.to_csv(tables / 'distance_summary.csv', index=False)
    LOG.info('Computing standardized car-access contrast on shared support')
    car = standardized_contrast(datasets, 'car_group', 'No car access', 'Car access',
                                ['purpose', 'age_group', 'home_pt_tier'], settings['minimum_people_per_stratum'])
    LOG.info('Computing standardized accessibility contrast on shared support')
    pt = standardized_contrast(datasets, 'home_pt_tier', 'Low (D/none)', 'High (A/B)',
                               ['purpose', 'age_group', 'car_group'], settings['minimum_people_per_stratum'])
    car.to_csv(tables / 'standardized_car_contrast.csv', index=False)
    pt.to_csv(tables / 'standardized_accessibility_contrast.csv', index=False)
    transition = transition_table(datasets)
    transition.to_csv(tables / 'accessibility_transitions.csv', index=False)
    persons = {name: activity_spaces(f) for name, f in datasets.items()}
    person_summary = []
    for name, frame in persons.items():
        frame.to_parquet(output / f'person_activity_spaces_{name}.parquet', index=False)
        f = frame.loc[frame.secondary_legs >= 2]
        person_summary.append(dict(dataset=name, persons=len(f),
            mean_radius_gyration_km=float(np.average(f.radius_gyration_km, weights=f.weight)),
            mean_revisit_share_250m=float(np.average(f.revisit_share_250m, weights=f.weight))))
    pd.DataFrame(person_summary).to_csv(tables / 'person_activity_spaces.csv', index=False)
    LOG.info('Rendering behavioral figures')
    figures.distance_figure(datasets, summary, images)
    figures.behavior_figure(summary, datasets, images)
    figures.contrast_figure(car, pt, images)
    figures.transition_figure(transition, datasets, images)
    figures.activity_figure(persons, images)
    replay_path = output / 'replay_manifest.json'
    replay_manifest = json.loads(replay_path.read_text()) if replay_path.exists() else None
    if replay_manifest and replay_manifest['generation'] == prepared['generation']:
        if replay_manifest['regions'] != settings['regions']:
            raise RuntimeError('Map regions changed since replay; rerun --phase replay before plotting')
        observed = pd.read_parquet(output / 'replay_observed.parquet')
        predictions = pd.read_parquet(output / 'replay_predictions.parquet')
        replay_summary, paired = paired_replay_summary(observed, predictions)
        replay_summary.to_csv(tables / 'paired_replay_summary.csv', index=False)
        paired.to_parquet(output / 'paired_replay_distances.parquet', index=False)
        figures.replay_figure(observed, predictions, images)
        selector_table, selector_trips = selector_diagnostics(observed, predictions)
        selector_table.to_csv(tables / 'selector_diagnostic.csv', index=False)
        figures.selector_figure(selector_trips, images)
        LOG.info('Preparing offline basemap layers')
        basemap = figures.prepare_basemap(caches, settings['regions'])
        write_json(output / 'basemap_provenance.json', {**basemap[3], 'projection': 'EPSG:2056', 'grid_metres': 500})
        figures.maps(datasets, observed, predictions, settings['regions'], settings['map_purposes'], basemap, images)
    else:
        LOG.warning('No replay from this prepared-data generation; survey replay and maps are omitted. Run --phase replay then --phase figures.')
    captions = []
    gallery = []
    # Only present figures generated for this cohort generation.
    stems = list(CAPTIONS)[:5]
    if replay_manifest and replay_manifest['generation'] == prepared['generation']:
        stems += ['06_observed_origin_survey_replay', '08_model_selection_diagnostic']
        stems += [f'07_map_{city}_{purpose}' for city in settings['regions'] for purpose in settings['map_purposes']]
    for stem in stems:
        if not (images / (stem + '.png')).exists(): continue
        caption = CAPTIONS['07_map'] if stem.startswith('07_map') else CAPTIONS[stem]
        captions.append(f'## {stem}\n\n{caption}\n')
        gallery.append(f'<section><h2>{html.escape(stem.replace("_", " "))}</h2><img loading="lazy" src="figures/{stem}.png"><p>{html.escape(caption)}</p><a href="figures/{stem}.pdf">PDF</a> · <a href="figures/{stem}.svg">SVG</a></section>')
    (output / 'captions.md').write_text('\n'.join(captions))
    findings = ['# Secondary-location comparison', '', 'Descriptive results from the completed spatial-assignment caches; the final MATSim stage is not required.', '']
    overall = summary.loc[summary.dimension == 'all'].copy()
    for row in overall.itertuples():
        findings.append(f'- {LABELS[row.dataset]}: mean {row.mean_km:.2f} km, median {row.median_km:.2f} km; {row.short_share:.1%} within 1.5 km, {row.long_share_20km:.1%} beyond 20 km ({row.persons:,} people, {row.trips:,} secondary legs).')
    findings += ['', '## Standardized behavioral contrasts', '']
    for label, table in [('Car access', car), ('Home accessibility', pt)]:
        for row in table.itertuples():
            if row.status != 'ok':
                findings.append(f'- {label}, {LABELS[row.dataset]}: insufficient common support.')
                continue
            findings.append(f'- {label}, {LABELS[row.dataset]}: {row.difference_km:+.2f} km (95% person-cluster interval {row.low_km:+.2f} to {row.high_km:+.2f}); {row.retained_weight_share:.1%} of contrasted-trip weight retained.')
    caveats = [
        'The survey reference period and synthetic scaling year can differ. Consult cohort coverage and recorded run settings; matching a historical marginal distribution is not the scientific objective.',
        'The microcensus supplies a behavioral reference, not a distribution-fitting objective. Raw population contrasts may reflect different population composition; standardized results restrict to shared support.',
        'These are straight-line secondary-arrival-leg distances and assigned pre-simulation locations, not MATSim routed distances, travel times or a complete daily travel budget.',
        'NN learned-selector means target distance is excluded from model inputs and model selection. Observed target distances still enter stationary-trip handling and distance-loss training; it is not a fully distance-unconstrained experiment.',
        'The saved NNs were estimated in separate runs whose survey income imputations differ. Replay holds one common input realization fixed, but it cannot remove training-data differences; this compares fitted pipelines rather than a controlled causal effect of removing the distance input alone.',
        'Survey replay uses existing fitted models and observed trip origins/chain features. It is not independent held-out validation and not a free-running rollout. Replicated draws quantify conditional stochastic destination variation, not parameter or training uncertainty.',
        'The compared simulations have one seed each. Person-cluster intervals do not measure between-seed model variability or the complete complex-survey design; very narrow population intervals should not be read as model certainty.',
        'Population weights are uniform within each sampled synthetic population. Survey trip statistics use person weights per trip; activity-space summaries use one person weight per person. Household clustering and survey strata are unavailable to this comparison.',
        'Car access is not car ownership. ÖV A/B indicates high accessibility and D/no-class low accessibility; no class is not treated as a missing value. Extent clipping excludes international secondary legs from statistical comparisons.']
    distance_cache = next((cache for cache in caches if cache.kind == 'nn_distance'), None)
    if distance_cache is not None and distance_cache.config('secondary_nn_short_sampling') == 'distance':
        threshold = distance_cache.config('secondary_nn_short_trip_threshold_m', 1500.)
        caveats.insert(0, f'The distance-input NN pipeline samples short-trip facilities directly by compatibility with the supplied distance, up to {threshold/1000:g} km. Purpose eligibility and the origin H3 neighbourhood are preserved. Its short-distance distribution is therefore constrained by input distances, rather than independently learned. Longer trips use the neural hierarchy; the distance-free pipeline retains its learned experts and selector.')
    for cache in caches:
        if cache.kind.startswith('nn'):
            weight = cache.config('secondary_nn_no_car_no_pt_weight', 1.)
            if weight < 1:
                caveats.insert(0, f'{LABELS[cache.kind]} uses a prediction-time accessibility rule: without both car availability and a driving licence, cell probabilities receive weight {weight:g} when no purpose-eligible A–D facility exists. The rule is soft and preserves positive mass. Accessibility patterns therefore reflect both learned attractiveness and this specified mobility rule.')
    findings += ['', '## Interpretation and limits', ''] + ['- ' + line for line in caveats]
    (output / 'findings.md').write_text('\n'.join(findings) + '\n')
    overview = overall[['dataset', 'trips', 'persons', 'mean_km', 'median_km', 'p90_km', 'short_share', 'long_share_20km']]
    document = '<!doctype html><html><head><meta charset="utf-8"><title>Secondary-location behavior comparison</title><style>body{font:15px/1.6 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#222}h1,h2{line-height:1.25}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:6px 9px;border-bottom:1px solid #ddd}section{margin:50px 0}.note{background:#f3f5f7;padding:18px}a{color:#0072b2}</style></head><body><h1>Secondary-location behavior comparison</h1><p>Reproducible analysis of three completed spatial-assignment caches.</p>'
    document += overview.to_html(index=False, float_format=lambda v: f'{v:.3f}')
    document += '<h2>Cohort coverage and exclusions</h2>' + pd.DataFrame(prepared['audit']).T.to_html()
    document += '<div class="note"><h2>How to interpret this comparison</h2><ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in caveats) + '</ul></div>'
    document += '<p><a href="findings.md">Numerical findings</a> · <a href="captions.md">Figure captions</a> · <a href="prepared_manifest.json">Data provenance</a></p>'
    document += '<h2>Standardized car contrast</h2>' + car.to_html(index=False, float_format=lambda v: f'{v:.3f}')
    document += '<h2>Standardized accessibility contrast</h2>' + pt.to_html(index=False, float_format=lambda v: f'{v:.3f}')
    document += ''.join(gallery) + '<h2>Machine-readable tables</h2><ul>' + ''.join(f'<li><a href="tables/{p.name}">{p.name}</a></li>' for p in sorted(tables.glob('*.csv'))) + '</ul></body></html>'
    (output / 'report.html').write_text(document)
    effective_settings = dict(settings)
    if replay_manifest and replay_manifest['generation'] == prepared['generation']:
        effective_settings['actual_replay'] = {key: replay_manifest[key] for key in ['seed', 'draws', 'max_replay_people', 'selected_people', 'selected_trips']}
    write_json(output / 'analysis_settings.json', effective_settings)
    write_json(output / 'analysis_provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version, packages={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'geopandas', 'matplotlib', 'torch', 'scikit-learn']}, code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}, pipeline_code_sha256={name: hashlib.sha256((REPOSITORY/'synthesis/population/spatial/secondary_nn'/name).read_bytes()).hexdigest() for name in ['mz_chains.py', 'model_wrappers.py', 'feature_encoding.py', 'h3.py', 'tools.py', 'locations.py', 'choice_model.py', 'location_helpers.py']}))
    LOG.info('Report: %s', output / 'report.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, default=Path(__file__).with_name('comparison.yml'))
    parser.add_argument('--configs', nargs='+', metavar='YAML',
                        help='Pipeline configurations to compare (overrides the settings file list)')
    parser.add_argument('--phase', choices=['all', 'status', 'prepare', 'replay', 'figures'], default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--survey-draws', type=int)
    parser.add_argument('--max-replay-people', type=int)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    settings = yaml.safe_load(args.settings.read_text())
    if args.configs is not None:
        settings['configs'] = args.configs
    for option in ['survey_draws', 'max_replay_people', 'threads']:
        if getattr(args, option) is not None: settings[option] = getattr(args, option)
        if settings[option] < 1: parser.error(f'{option} must be at least 1')
    transformer = Transformer.from_crs(4326, 2056, always_xy=True)
    for region in settings['regions'].values():
        x,y = transformer.transform(*region['center_lon_lat'])
        width = region['half_width_m']
        # Snap extent to the same 500 m grid used for destination densities.
        x,y = round(x/500)*500,round(y/500)*500
        region['bounds'] = [x-width,y-width,x+width,y+width]
    caches = caches_for(settings)
    if args.phase == 'status':
        for cache in caches:
            print(cache.kind, cache.directory, 'completed assignment root:', cache.root)
        return
    output = args.output.resolve() if args.output else REPOSITORY / settings['output']
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in ['all', 'prepare']:
        prepared = prepare(caches, output)
    else:
        prepared = json.loads((output / 'prepared_manifest.json').read_text())
        for cache in caches:
            if prepared['runs'][cache.kind]['root'] != cache.root:
                raise RuntimeError('Prepared data belong to a different cache generation; run --phase prepare again')
    if args.phase in ['all', 'replay']:
        replay_models(caches, settings, output, prepared)
    if args.phase in ['all', 'figures']:
        figures_and_report(caches, settings, output, prepared)


if __name__ == '__main__':
    main()
