"""Tables, standalone scientific figures and a self-contained fitted-model report."""
import base64
from html import escape
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import log_softmax

from dmc.models import model_jax
from dmc.vot.functions import vot_utils
from dmc.parameters import NAMES_CONVERSION
from . import data, elasticities, replay

LABELS = {'estimated': 'Fitted model', 'exported': 'Exported application parameters'}
COLORS = {'estimated': '#087f8c', 'exported': '#c06b25'}
MODE_LABELS = {'car': 'Car driver', 'pt': 'Public transport', 'bike': 'Bicycle',
               'walk': 'Walking', 'car_passenger': 'Car passenger'}


class VariantContext:
    def __init__(self, parent, references):
        self.parent, self.references = parent, references

    def config(self, name, *args, **kwargs):
        return self.references[name] if name in self.references else self.parent.config(name, *args, **kwargs)


def generate(context, dataset, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    repetitions = int(context.config('analysis.dmc.repetitions'))
    step = float(context.config('analysis.dmc.elasticity_step'))
    seed = context.config('random_seed')
    if seed is None or repetitions < 2 or not 0 < step < 1:
        raise ValueError('A configured random seed, at least two draws and a valid elasticity step are required')
    frame, modes = dataset.frame, dataset.modes
    arrays = model_jax.dataframe_to_arrays(frame)
    rows = replay.layout(dataset.keys)
    weights = dataset.survey_weights
    observed = frame['mode'].to_numpy(int)
    predictions = {}; samples = {}; valuations = {}
    scores = []; shares = []; intervals = []; sensitivity = []; checks = []; scenarios = []
    valuations_summary = []
    exported = dataset.keys.copy()
    exported['observed_mode'] = np.asarray(modes)[observed]
    exported['survey_weight'] = weights
    exported['fit_weight'] = frame.person_weight.to_numpy()
    for variant, beta in dataset.coefficients.items():
        current = VariantContext(context, dataset.export_references if variant == 'exported' else {})
        evaluate = replay.predictor(current, beta, modes)
        prediction = replay.predict(arrays, evaluate, beta, modes, rows)
        predictions[variant] = prediction
        draws, sampled = replay.sampling_shares(prediction, rows, weights, seed, repetitions)
        samples[variant] = sampled
        exported[variant + '_sampled_mode'] = np.asarray(modes)[sampled]
        for index, mode in enumerate(modes):
            exported[f'{variant}_probability_{mode}'] = prediction.probabilities[:, index]
            for weighting, values in [('survey', weights), ('fit', frame.person_weight.to_numpy())]:
                shares.append({'variant': variant, 'weighting': weighting, 'mode': mode,
                               'observed': np.average(observed == index, weights=values),
                               'expected': np.average(prediction.probabilities[:, index], weights=values),
                               'sampled': np.average(sampled == index, weights=values)})
            intervals.append({'variant': variant, 'mode': mode, 'draws': repetitions,
                              'expected': np.average(prediction.probabilities[:, index], weights=weights),
                              'sampling_mean': draws[:, index].mean(), 'first_sample': draws[0, index],
                              'sampling_lower': np.quantile(draws[:, index], .025),
                              'sampling_upper': np.quantile(draws[:, index], .975)})
        utility, available = evaluate(arrays)
        log_p = log_softmax(np.where(np.asarray(available) > 0, np.asarray(utility), -np.inf), axis=1)
        chosen_log = log_p[np.arange(len(frame)), observed]
        marginal = prediction.probabilities[np.arange(len(frame)), observed]
        conditional_ll = float(np.sum(chosen_log * frame.person_weight))
        if variant == 'estimated' and not np.isclose(conditional_ll, dataset.audit['fitted_log_likelihood'], atol=1e-5, rtol=0):
            raise ValueError('The cached likelihood does not match current utilities; estimate the current DMC model before reporting')
        scores.append({'variant': variant,
                       'conditional_mean_negative_LL_fit_weights': -np.average(chosen_log, weights=frame.person_weight),
                       'conditional_weighted_LL': conditional_ll,
                       'autonomous_marginal_mean_negative_LL_survey_weights': -np.average(np.log(np.maximum(marginal, np.finfo(float).tiny)), weights=weights)})
        result = SimpleNamespace(get_beta_values=lambda: beta)
        car, car_mean = vot_utils.get_car_vot(current, frame, result, modes)
        pt, pt_mean, iv, access, wait = vot_utils.get_pt_vot(current, frame, result, modes)
        valuations[variant] = {'car': car, 'pt': pt}
        for mode, values in [('car', car), ('pt', pt)]:
            use_weights = current.config('use_weights_for_vot')
            value_weights = frame.person_weight.loc[values.index].to_numpy() if use_weights else np.ones(len(values))
            quantiles = weighted_quantile(values.to_numpy(), value_weights, [.1, .5, .9, .95])
            valuations_summary.append({'variant': variant, 'mode': mode, 'trips': len(values),
                                       'mean_CHF_per_hour': car_mean if mode == 'car' else pt_mean,
                                       'p10': quantiles[0], 'median': quantiles[1], 'p90': quantiles[2], 'p95': quantiles[3],
                                       'weighted': use_weights})
            column = np.full(len(frame), np.nan); column[values.index] = values.to_numpy()
            exported[f'{variant}_{mode}_VOT_CHF_per_hour'] = column
        for name, values in [('in_vehicle', iv), ('access_egress', access), ('waiting', wait)]:
            column = np.full(len(frame), np.nan); column[values.index] = values.to_numpy()
            exported[f'{variant}_pt_{name}_VOT_CHF_per_hour'] = column
        values, comparisons, finite = elasticities.compute(arrays, evaluate, beta, modes, rows, weights, variant, step)
        sensitivity.append(values); checks.append(comparisons); scenarios.append(finite)

    # Save full numeric diagnostics before rendering figures and the HTML.
    shares = pd.DataFrame(shares); intervals = pd.DataFrame(intervals); scores = pd.DataFrame(scores)
    valuations_summary = pd.DataFrame(valuations_summary)
    sensitivity = pd.concat(sensitivity, ignore_index=True)
    for name, table in [('mode_shares', shares), ('sampling_intervals', intervals), ('likelihood', scores),
                        ('value_of_time_summary', valuations_summary), ('elasticities', sensitivity),
                        ('elasticity_step_checks', pd.concat(checks)), ('finite_changes', pd.concat(scenarios))]:
        table.to_csv(output / (name + '.csv'), index=False)
    exported.to_parquet(output / 'microcensus_replay.parquet', index=False)
    coefficients = pd.DataFrame([{'parameter': name, 'exported_name': NAMES_CONVERSION[name],
                                  'estimate': dataset.coefficients['estimated'][name],
                                  'exported_value': dataset.coefficients['exported'][name],
                                  'fixed': declaration.fixed, 'lower_bound': declaration.lower_bound,
                                  'upper_bound': declaration.upper_bound}
                                 for name, declaration in dataset.declarations.items()])
    coefficients.to_csv(output / 'coefficients.csv', index=False)
    plot_shares(output, shares, intervals, modes)
    plot_vot(output, valuations, frame, context.config('use_weights_for_vot'))
    curves = plot_curves(output, dataset, predictions, samples)
    plot_elasticities(output, sensitivity, modes)
    audit = {**dataset.audit, 'random_seed': int(seed), 'sampling_repetitions': repetitions,
             'elasticity_step': step, 'passenger_history': 'generated any-earlier and previous-trip states, reset per person-day',
             'vot_constraints': context.config('dmc_vot_constraints'),
             'vot_car_target': context.config('dmc_vot_car_target'), 'vot_pt_target': context.config('dmc_vot_pt_target'),
             'source_sha256': {str(path): data.fingerprint(path) for path in source_files()}}
    (output / 'provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
    write_html(output, audit, scores, shares, intervals, valuations_summary, coefficients, curves, sensitivity)
    return str(output / 'index.html')


def weighted_quantile(values, weights, quantiles):
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    position = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return np.interp(quantiles, position, values)


def finish(figure, output, name):
    figure.savefig(output / (name + '.png'), dpi=160, bbox_inches='tight')
    figure.savefig(output / (name + '.svg'), bbox_inches='tight', metadata={'Date': None})
    plt.close(figure)


def plot_shares(output, shares, intervals, modes):
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    x = np.arange(len(modes)); width = .24
    actual = shares.loc[(shares.variant == 'estimated') & (shares.weighting == 'survey')].set_index('mode').loc[modes]
    for axis, field in zip(axes, ['expected', 'sampled']):
        axis.bar(x - width, 100 * actual.observed, width, color='black', label='Observed MZ')
        for i, variant in enumerate(LABELS):
            part = shares.loc[(shares.variant == variant) & (shares.weighting == 'survey')].set_index('mode').loc[modes]
            if field == 'expected':
                bounds = intervals.loc[intervals.variant == variant].set_index('mode').loc[modes]
                # Draw the percentile range directly, even for highly skewed samples.
                axis.errorbar(x + i * width, 50 * (bounds.sampling_lower + bounds.sampling_upper),
                              yerr=50 * (bounds.sampling_upper - bounds.sampling_lower),
                              fmt='none', color=COLORS[variant], capsize=3)
            axis.bar(x + i * width, 100 * part[field], width, color=COLORS[variant], label=LABELS[variant])
        axis.set_xticks(x, [MODE_LABELS[m] for m in modes], rotation=18)
        axis.set_ylabel('MZ survey-weighted mode share (%)'); axis.grid(axis='y', alpha=.2)
    axes[0].set_title('Exact generated-history probabilities; sampled 95% ranges')
    axes[1].set_title('One common-seed sampled replay'); axes[0].legend(fontsize=8)
    finish(figure, output, 'mode_shares')


def plot_vot(output, valuations, frame, weighted):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for column, mode in enumerate(['car', 'pt']):
        pooled = np.concatenate([item[mode].to_numpy() for item in valuations.values()])
        limit = max(1., float(np.quantile(pooled, .995))) if len(pooled) else 1.
        bins = np.linspace(0, limit, 41)
        for variant, values in valuations.items():
            series = values[mode]
            if series.empty:
                continue
            weight = frame.person_weight.loc[series.index].to_numpy() if weighted else np.ones(len(series))
            shown = series.to_numpy() <= limit
            axes[0, column].hist(series.to_numpy()[shown], bins=bins, weights=weight[shown], density=True,
                                 histtype='step', linewidth=2, color=COLORS[variant], label=LABELS[variant],
                                 linestyle='-' if variant == 'estimated' else '--')
            order = np.argsort(series.to_numpy())
            axes[1, column].step(np.r_[0., series.to_numpy()[order]], np.r_[0., np.cumsum(weight[order]) / weight.sum()],
                                 where='post', color=COLORS[variant], label=LABELS[variant],
                                 linestyle='-' if variant == 'estimated' else '--')
        axes[0, column].set(title=MODE_LABELS[mode] + ': histogram through pooled 99.5th percentile', ylabel='Density')
        axes[1, column].set(title=MODE_LABELS[mode] + ': full empirical distribution', ylabel='Cumulative fraction', ylim=(0, 1))
        axes[1, column].set_xscale('symlog', linthresh=10.)
        for axis in axes[:, column]:
            axis.set_xlabel('Marginal willingness to pay for time saving (CHF/hour)'); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    finish(figure, output, 'value_of_time')


def curve_definitions(frame, keys):
    all_rows = np.ones(len(frame), bool)
    return [
        ('age', frame.age, [0, 18, 25, 35, 45, 55, 65, 75, 90, np.inf], all_rows, 'Age (years)'),
        ('income', frame.income, [0, 1500, 2500, 3500, 5000, 7500, 10000, 15000, np.inf], all_rows, 'Equivalent monthly income (CHF)'),
        ('distance', frame.euclidean_distance_km, [0, .5, 1, 2, 3, 5, 7, 10, 15, 25, 50, 100, np.inf], all_rows, 'Euclidean OD distance (km)'),
        ('transfers', frame.pt_transfers, [0, 1, 2, 3, 4, 5, np.inf], frame.pt_availability.to_numpy() > 0, 'PT transfers; PT available'),
        ('pt_time', frame.pt_in_vehicle_time_min, [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, np.inf], frame.pt_availability.to_numpy() > 0, 'PT in-vehicle minutes; PT available'),
        ('car_time', frame.car_travel_time_min, [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, np.inf], frame.car_availability.to_numpy() > 0, 'Car travel minutes; car available'),
        ('waiting', np.maximum(frame.pt_transfer_time_min, 0), [0, 1, 3, 5, 10, 20, 30, 60, np.inf], frame.pt_availability.to_numpy() > 0, 'PT waiting minutes, clipped at zero; PT available'),
        ('trip_position', keys.groupby('person_id', sort=False).cumcount() + 1, [1, 2, 3, 4, 5, 6, 7, 8, np.inf], all_rows, 'Trip position within selected day prefix'),
        *([('passenger_distance', frame.car_passenger_distance_km, [0, 5, 15, 25, 35, 50, 75, 100, np.inf], frame.car_passenger_availability.to_numpy() > 0, 'Routed passenger distance (km); passenger available')]
          if 'car_passenger_distance_km' in frame else []),
    ]


def plot_curves(output, dataset, predictions, samples):
    frame, modes, weight = dataset.frame, dataset.modes, dataset.survey_weights
    observed = frame['mode'].to_numpy(int)
    figures = []
    for name, values, bins, domain, xlabel in curve_definitions(frame, dataset.keys):
        category = pd.cut(np.asarray(values), bins, right=False, include_lowest=True, labels=False)
        records = []
        for group in range(len(bins) - 1):
            selected = domain & (category == group)
            if not selected.any():
                continue
            label = f'{bins[group]:g}–{bins[group + 1]:g}' if np.isfinite(bins[group + 1]) else f'≥{bins[group]:g}'
            for variant, prediction in predictions.items():
                for index, mode in enumerate(modes):
                    records.append({'variant': variant, 'mode': mode, 'group': group, 'label': label,
                                    'trips': int(selected.sum()), 'weight': float(weight[selected].sum()),
                                    'observed': np.average(observed[selected] == index, weights=weight[selected]),
                                    'expected': np.average(prediction.probabilities[selected, index], weights=weight[selected]),
                                    'sampled': np.average(samples[variant][selected] == index, weights=weight[selected])})
        table = pd.DataFrame(records)
        table.to_csv(output / ('shares_by_' + name + '.csv'), index=False)
        if table.empty:
            continue
        figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
        for index, mode in enumerate(modes):
            axis = axes.flat[index]
            actual = table.loc[(table.variant == 'estimated') & (table['mode'] == mode)]
            axis.plot(actual.group, 100 * actual.observed, 'ko-', label='Observed MZ', markersize=3)
            for variant in LABELS:
                part = table.loc[(table.variant == variant) & (table['mode'] == mode)]
                axis.plot(part.group, 100 * part.expected, 'o-', color=COLORS[variant], label=LABELS[variant] + ' probabilities', markersize=3)
                axis.plot(part.group, 100 * part.sampled, 'x--', color=COLORS[variant], alpha=.7, label=LABELS[variant] + ' sampled', markersize=3)
            axis.set_xticks(actual.group, [f'{label}\nn={n}' for label, n in zip(actual.label, actual.trips)], rotation=35, fontsize=8)
            axis.set(title=MODE_LABELS[mode], xlabel=xlabel, ylabel='MZ weighted mode share (%)', ylim=(0, 100)); axis.grid(alpha=.2)
        for axis in list(axes.flat)[len(modes):]:
            axis.axis('off')
        axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc='center', fontsize=8)
        finish(figure, output, 'shares_by_' + name)
        figures.append(('shares_by_' + name, xlabel))
    return figures


def plot_elasticities(output, values, modes):
    direct = []
    for variant in LABELS:
        for mode in modes:
            for feature, driver in [('Income', 'income'), ('Own travel time', 'time_' + mode),
                                    ('Own monetary cost', 'cost_' + mode if mode in ('car', 'pt') else None),
                                    ('All distances', 'distance'),
                                    ('Own passenger distance', 'distance_car_passenger' if mode == 'car_passenger' else None),
                                    ('Own PT transfers', 'transfers' if mode == 'pt' else None)]:
                selected = values.loc[(values.variant == variant) & (values.response_mode == mode) & (values.driver == driver)]
                direct.append({'variant': variant, 'mode': mode, 'feature': feature,
                               'elasticity': selected.elasticity.iloc[0] if len(selected) else np.nan})
    direct = pd.DataFrame(direct)
    direct.to_csv(output / 'direct_elasticities.csv', index=False)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    x = np.arange(len(modes)); width = .35
    for axis, feature in zip(axes.flat, direct.feature.unique()):
        for index, variant in enumerate(LABELS):
            part = direct.loc[(direct.variant == variant) & (direct.feature == feature)].set_index('mode').loc[modes]
            axis.bar(x + (index - .5) * width, part.elasticity, width, color=COLORS[variant], label=LABELS[variant])
        absent = direct.loc[direct.feature == feature].groupby('mode').elasticity.count()
        for index, mode in enumerate(modes):
            if absent[mode] == 0:
                axis.text(index, .8, 'N/A', ha='center', transform=axis.get_xaxis_transform(), fontsize=9)
        axis.set_xticks(x, [MODE_LABELS[m] for m in modes], rotation=20, fontsize=9)
        axis.set(title=feature, ylabel='Expected mode-share elasticity'); axis.axhline(0, color='black', linewidth=.7)
        axis.grid(axis='y', alpha=.2)
    axes.flat[0].legend(fontsize=8)
    finish(figure, output, 'direct_elasticities')
    drivers = values.driver.drop_duplicates().tolist()
    matrix = values.pivot(index=['variant', 'response_mode'], columns='driver', values='elasticity')
    finite = values.elasticity[np.isfinite(values.elasticity)]
    limit = max(.01, float(finite.abs().max())) if len(finite) else 1.
    figure, axes = plt.subplots(2, 1, figsize=(17, 8), layout='constrained')
    for axis, variant in zip(axes, LABELS):
        array = matrix.loc[variant].reindex(modes)[drivers].to_numpy()
        picture = axis.imshow(np.ma.masked_invalid(array), aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        for row in range(len(modes)):
            for column in range(len(drivers)):
                value = array[row, column]
                axis.text(column, row, f'{value:+.3f}' if np.isfinite(value) else 'N/A', ha='center', va='center', fontsize=7)
        axis.set_yticks(range(len(modes)), [MODE_LABELS[m] for m in modes])
        axis.set_xticks(range(len(drivers)), drivers, rotation=25, ha='right', fontsize=9)
        axis.set_title(LABELS[variant])
    figure.colorbar(picture, ax=list(axes), label='Expected mode-share elasticity')
    finish(figure, output, 'elasticities')


def source_files():
    import dmc.models.data
    import dmc.models.declarations
    import dmc.models.utility
    import dmc.parameters
    import dmc.vot.functions
    return [*sorted(Path(__file__).parent.glob('*.py')), Path(model_jax.__file__),
            Path(model_jax.__file__).with_name('model_biogeme.py'), Path(dmc.models.data.__file__), Path(dmc.models.declarations.__file__),
            Path(dmc.models.utility.__file__), Path(dmc.parameters.__file__), Path(dmc.vot.functions.__file__)]


def write_html(output, audit, scores, shares, intervals, vot, coefficients, curves, sensitivities):
    def table(value, precision=4):
        return '<div class="table">' + value.to_html(index=False, float_format=lambda v: f'{v:.{precision}f}', na_rep='N/A') + '</div>'

    def image(name):
        content = base64.b64encode((output / (name + '.png')).read_bytes()).decode()
        return f'<img alt="{escape(name)}" src="data:image/png;base64,{content}">'

    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DMC Microcensus report</title>
<style>body{{font:17px/1.6 Georgia,serif;max-width:1350px;margin:40px auto;padding:0 24px;color:#172d35}}h1,h2,h3{{font-family:system-ui}}img{{max-width:100%;height:auto}}.table{{overflow:auto}}table{{border-collapse:collapse;font:13px system-ui}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}.notice{{padding:18px;background:#fff1d6;border-left:4px solid #a87119}}code{{background:#eef4f5}}</style></head><body>
<h1>Fitted DMC model: Microcensus replay and sensitivities</h1>
<p>This report evaluates one production model from <code>dmc.models</code>. Analysis does not estimate coefficients or define replacement utilities. The two views are the full-precision fitted model and its actual exported application parameters, which include four-decimal rounding and a threefold passenger long-distance coefficient beyond 25 routed kilometres.</p>
<div class="notice"><b>In-sample diagnostics.</b> {audit['reported_trips']:,} trips from {audit['people']:,} people are evaluated on the estimator's exact selected cohort. {audit['complete_days']:,} person-days are complete relative to the processed MZ diaries; {audit['retained_final_trips']:,} final trips are retained. {audit['excluded_training_trips']:,} training trips lie outside the selected sequences. Interrupted days are starting prefixes, never stitched across removed trips. These results are not held-out validation or complete-day forecasts for the incomplete prefixes.</div>
<h2>Replay and mode shares</h2><p>Mode-share figures use original MZ person weights, joined by person/trip identifier. The likelihood uses the pipeline's existing mode-adjusted training weights; a second aggregate table uses those fit weights. This difference can explain aggregate share differences even with fitted constants. Conditional likelihood legitimately uses observed history to score the observed sequence. Autonomous replay starts every person with no passenger history and propagates generated choices only. Exact probabilities use a three-state forward recursion (never passenger, earlier passenger with another last mode, or immediately previous passenger); one common-seed sampled replay and ranges over {audit['sampling_repetitions']} replays show choice-draw noise, not survey or parameter uncertainty.</p>
{table(scores,6)}{image('mode_shares')}{table(shares)}{table(intervals)}
<p>The replay is sequential conditional MNL. It does not enforce car/bike location consistency or match passengers to drivers. It is not the production whole-tour sampler and does not claim that daily-history integration into population/Java choice is complete.</p>
<h2>Mode shares by traveller and trip characteristics</h2><p>Each figure compares observed MZ shares, exact probabilities and one generated-history sample within the same bins. These are descriptive conditional shares, not individual causal response curves. Bins show sample counts; sparse bins should be interpreted cautiously. Transfers, waiting and PT-time figures require PT availability. Car/passenger travel attributes similarly require that alternative to be available.</p>'''
    for name, title in curves:
        body += '<h3>' + escape(title) + '</h3>' + image(name)
    body += f'''<h2>Value of time: car and public transport</h2><p>Values are marginal willingness to pay, <code>60 × (∂utility/∂time_min)/(∂utility/∂cost_CHF)</code>, on observed available car/PT trips. Combined PT VOT saves one minute proportionally across nonnegative in-vehicle, access/egress and waiting durations. The histogram display excludes the extreme pooled 0.5% tail; the ECDF and exported trip-level values retain the full distribution. Summary and plotting weights follow the estimation's <code>use_weights_for_vot</code> option. The configured car {audit['vot_car_target']:g} and combined PT {audit['vot_pt_target']:g} CHF/hour targets are external calibration assumptions when enabled in JAX, not unconstrained empirical estimates. VOT constraints enabled in this fit: {audit['vot_constraints']}. Rounding exported coefficients can shift those means slightly.</p>
{table(vot)}{image('value_of_time')}
<h2>Elasticities and finite changes</h2><p>Each entry is <code>d log(expected weighted mode share) / d log(input factor)</code>, using symmetric ±{100*audit['elasticity_step']:g}% changes and a half-step check. An elasticity of −0.3 means approximately a 0.3% relative share decrease for a 1% input increase, not 0.3 percentage points. Coefficients, person weights and availability are fixed. Generated-history states are recomputed under every intervention. These are local model sensitivities, not causal estimates. Undefined zero-share elasticities are N/A.</p>
<p><code>distance</code> scales Euclidean and all routed distances together, holding times fixed; <code>distance_car_passenger</code> changes only passenger routed kilometres, including the 25 km hinge. Income affects both preference curves and its cost interaction. Time changes scale nonnegative physical components while preserving negative cached corrections. <code>pt_waiting</code> changes only waiting. Transfer elasticities use a continuous count relaxation; the separate <code>finite_changes.csv</code> includes the physically discrete scenario of one extra transfer and +10% attribute scenarios. No own monetary-cost attribute exists for bicycle, walking or passenger; only car and PT cost drivers are reported.</p>
{image('direct_elasticities')}{image('elasticities')}<details><summary>Numeric elasticities</summary>{table(sensitivities,5)}</details>
<h2>Coefficients and reproducibility</h2><p>Estimated and exported values below come from the actual fitted result and its YAML artifacts. All active utility equations, including both any-earlier and previous-trip passenger history and the additional long-distance penalty, are evaluated through <code>dmc.models.model_jax.build_utilities</code>. Full coefficients, trip probabilities, one sampled replay, VOT values, bins, finite changes and step-size checks are exported as CSV/Parquet beside this self-contained HTML. PNG and SVG scientific figures are available separately. Source and artifact fingerprints are recorded in <code>provenance.json</code>.</p>
<details><summary>Full coefficient table</summary>{table(coefficients,6)}</details></body></html>'''
    (output / 'index.html').write_text(body)
