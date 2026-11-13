"""Statistics on complete request cohorts and successful matched journey pairs."""
from itertools import combinations
from pathlib import Path
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mode_choice.defaults import UNROUTABLE_TIME_MIN

METRICS = {'pt_total_travel_time_min': 'Total time (min)', 'pt_in_vehicle_time_min': 'In-vehicle time (min)',
           'pt_access_egress_time_min': 'Access + egress (min)', 'pt_transfer_time_min': 'Transfer time (min)',
           'pt_transfers': 'Transfers', 'pt_in_vehicle_distance_km': 'In-vehicle distance (km)',
           'pt_total_distance_km': 'Total distance (km)', 'component_sum_min': 'Sum of time components (min)'}
LABELS = {'rust': 'Eqasim Rust', 'raptor': 'SwissRailRaptor', 'flash': 'FLASH-TB', 'google': 'Saved Google'}


def classify(trips, routes, backend):
    if routes['trip_id'].null_count() or routes['trip_id'].n_unique() != routes.height:
        raise ValueError(f'{backend}: duplicate/null trip identifiers')
    if routes.select('trip_id').join(trips.select('trip_id'), on='trip_id', how='anti').height:
        raise ValueError(f'{backend}: returned unknown trip identifiers')
    routes = trips.select('trip_id').join(routes, on='trip_id', how='left', validate='1:1', maintain_order='left')
    for column in METRICS:
        if column not in routes.columns:
            routes = routes.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    routes = routes.with_columns((pl.col('pt_in_vehicle_time_min') + pl.col('pt_access_egress_time_min') +
                                  pl.col('pt_transfer_time_min')).alias('component_sum_min'))
    valid_total = (pl.col('pt_total_travel_time_min').is_finite() & (pl.col('pt_total_travel_time_min') >= 0) &
                   (pl.col('pt_total_travel_time_min') < UNROUTABLE_TIME_MIN)).fill_null(False)
    valid_components = pl.all_horizontal([
        (pl.col(c).is_finite() & (pl.col(c) >= 0) & (pl.col(c) < UNROUTABLE_TIME_MIN))
        for c in ['pt_in_vehicle_time_min', 'pt_access_egress_time_min', 'pt_transfer_time_min', 'pt_transfers']]).fill_null(False)
    status = pl.when(~valid_total).then(pl.lit('invalid_or_unroutable')).when(~valid_components).then(
        pl.lit('invalid_components')).when(pl.col('pt_in_vehicle_time_min') <= 0).then(pl.lit('walk_only')).otherwise(pl.lit('success'))
    if backend == 'google':
        status = pl.when(pl.col('google_trip_id').is_null()).then(pl.lit('unverified_original_trip')).when(
            ~pl.col('reference_present').fill_null(False)).then(pl.lit('not_in_cleaned_reference')).when(
                ~pl.col('expected_pt_used').fill_null(False)).then(pl.lit('expected_pt_not_used')).otherwise(status)
    else:
        status = pl.when(~pl.col('route_row_present').fill_null(False)).then(pl.lit('missing_route_row')).otherwise(status)
    found = valid_total & pl.col('pt_in_vehicle_time_min').is_finite() & (pl.col('pt_in_vehicle_time_min') > 0) & (pl.col('pt_in_vehicle_time_min') < UNROUTABLE_TIME_MIN)
    if backend == 'google':
        found = found & pl.col('google_trip_id').is_not_null() & pl.col('reference_present').fill_null(False) & pl.col('expected_pt_used').fill_null(False)
    else:
        found = found & pl.col('route_row_present').fill_null(False)
    routes = routes.with_columns(status.alias('status'), found.fill_null(False).alias('journey_found')).with_columns(
        (pl.col('status') == 'success').alias('successful'))
    # Keep the original stage output separately; invalid individual measurements
    # must not discard valid total-time pairs or enter transfer/distance averages.
    routes = routes.with_columns([pl.when(pl.col(c).is_finite() & (pl.col(c) >= 0) & (pl.col(c) < UNROUTABLE_TIME_MIN))
                                  .then(pl.col(c)).otherwise(None).alias(c) for c in METRICS if c != 'component_sum_min'])
    return routes.with_columns((pl.col('pt_in_vehicle_time_min') + pl.col('pt_access_egress_time_min') +
                               pl.col('pt_transfer_time_min')).alias('component_sum_min'))


def statistics(trips, routes):
    summary, distributions, segments = [], [], []
    for backend, frame in routes.items():
        success = frame.filter('journey_found')
        weighted = frame.select('trip_id', 'successful').join(trips.select('trip_id', 'weight_person'), on='trip_id', validate='1:1')
        total_weight = weighted['weight_person'].sum()
        row = {'backend': backend, 'requested': frame.height, 'pt_journeys_found': success.height,
               'pt_journeys_not_found': frame.height - success.height, 'pt_journey_found_pct': success.height / frame.height * 100,
               'complete_valid_journeys': frame['successful'].sum(),
               'weighted_complete_valid_pct': float(weighted.filter('successful')['weight_person'].sum() / total_weight * 100) if total_weight else None}
        row.update({status: int(count) for status, count in frame.group_by('status').len().iter_rows()})
        summary.append(row)
        for column in METRICS:
            values = success[column].drop_nulls().to_numpy()
            values = values[np.isfinite(values)]
            distributions.append({'backend': backend, 'metric': column, 'n': len(values),
                                  'mean': float(values.mean()) if len(values) else None,
                                  'median': float(np.median(values)) if len(values) else None,
                                  'p90': float(np.quantile(values, .9)) if len(values) else None,
                                  'p95': float(np.quantile(values, .95)) if len(values) else None})
        segmented = frame.select('trip_id', 'successful').join(trips.select(
            'trip_id', 'mode_detailed', 'trip_crossing_border', 'trip_outside_ch',
            (pl.col('departure_time') / 3600).floor().cast(pl.Int64).alias('departure_hour'),
            pl.when(pl.col('crowfly_distance') < 1000).then(pl.lit('<1 km')).when(pl.col('crowfly_distance') < 5000).then(
                pl.lit('1–5 km')).when(pl.col('crowfly_distance') < 20000).then(pl.lit('5–20 km')).otherwise(pl.lit('>=20 km')).alias('distance_band')),
            on='trip_id', validate='1:1')
        for category in ['mode_detailed', 'trip_crossing_border', 'trip_outside_ch', 'departure_hour', 'distance_band']:
            for value, count, successes in segmented.group_by(category).agg(pl.len().alias('n'), pl.col('successful').sum()).iter_rows():
                segments.append({'backend': backend, 'category': category, 'value': str(value), 'n': count,
                                 'successful': successes, 'success_pct': successes / count * 100})
    return pl.DataFrame(summary, infer_schema_length=None), pl.DataFrame(distributions), pl.DataFrame(segments)


def paired_statistics(routes):
    rows = []
    for left, right in combinations(routes, 2):
        a, b = routes[left].filter('journey_found'), routes[right].filter('journey_found')
        joined = a.select('trip_id', *METRICS).join(b.select('trip_id', *METRICS), on='trip_id', suffix='_other', validate='1:1')
        for column in METRICS:
            x, y = joined[column].to_numpy(), joined[column + '_other'].to_numpy()
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            difference = y - x
            rows.append({'left': left, 'right': right, 'metric': column, 'matched_trips': len(x),
                         'left_mean': float(x.mean()) if len(x) else None, 'right_mean': float(y.mean()) if len(x) else None,
                         'mean_difference_right_minus_left': float(difference.mean()) if len(x) else None,
                         'mean_absolute_difference': float(abs(difference).mean()) if len(x) else None,
                         'median_absolute_difference': float(np.median(abs(difference))) if len(x) else None,
                         'p90_absolute_difference': float(np.quantile(abs(difference), .9)) if len(x) else None,
                         'rmse': float(np.sqrt(np.mean(difference ** 2))) if len(x) else None,
                         'correlation': float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else None,
                         'exact_agreement_pct': float(np.isclose(x, y, rtol=0, atol=1e-8).mean() * 100) if len(x) else None})
    return pl.DataFrame(rows)


def generate(trips, routes, timings, metadata, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    overview, distributions, segments = statistics(trips, routes)
    pairs = paired_statistics(routes)
    for name, frame in [('success', overview), ('distributions', distributions), ('segments', segments), ('pairwise', pairs)]:
        frame.write_csv(output / (name + '.csv'))
    wide = trips
    for backend, frame in routes.items():
        wide = wide.join(frame.select('trip_id', *METRICS, 'status', 'successful', 'journey_found').rename({
            column: backend + '_' + column for column in [*METRICS, 'status', 'successful', 'journey_found']}), on='trip_id', validate='1:1')
    wide.write_parquet(output / 'comparison.parquet')
    wide.group_by([name + '_journey_found' for name in routes]).len().write_csv(output / 'success_patterns.csv')
    (output / 'timings.json').write_text(json.dumps(timings, indent=2) + '\n')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    figures(routes, timings, output)
    content = ["<!doctype html><html><head><meta charset='utf-8'><title>Microcensus PT routers</title>",
               '<style>body{font:15px system-ui;margin:2em;max-width:1500px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd;text-align:right}th{background:#eee}img{max-width:100%}.scroll{overflow:auto}p{max-width:1050px}</style></head><body>',
               '<h1>Microcensus public-transport router comparison</h1>',
               '<p>Identical observed PT requests; all trips by default. No Google API calls. Survey person and trip IDs, coordinates, departure times and weights are retained. The Google join uses original WEGNR verified by departure time and both endpoints within 1 m, because survey cleaning can renumber or split trips.</p>',
               '<p>Scenario service day: ' + html.escape(str(metadata['service_date'])) + '; eqasim commit: ' + html.escape(str(metadata['eqasim_commit'])) + '.</p>',
               '<h2>Selection</h2><pre>' + html.escape(json.dumps(metadata['selection'], indent=2)) + '</pre>',
               '<h2>PT journeys found and component validity</h2>', table(overview),
               '<p>A PT journey is found when total and positive in-vehicle time are valid and PT was used. Complete valid journeys additionally require valid components. Pure walking results, missing rows and invalid components are shown separately. In particular, negative Google transfer times are excluded from that metric but do not discard valid Google total-time comparisons. Google noncoverage means absent from the cleaned reference or an unverified original key; it is not evidence of an API routing failure. The reference has already removed trips without two alternatives or a routed selected mode. Weighted percentages use microcensus person weights; the remaining statistics count trips.</p>',
               '<img src="reachability.png">', '<h2>Routing performance</h2>', timing_table(timings), '<img src="runtime.png">',
               '<p>Servers run sequentially with the same thread allocation. Eqasim startup includes Rust preprocessing or Raptor initialization. FLASH preprocessing is separate and shared with population routing. The first FLASH routing call includes native timetable loading; later calls reuse it. Backend timings include their Python adapter and file I/O, exclude fares, and are not isolated native-algorithm benchmarks. Original Google runtime is unavailable.</p>',
               '<h2>Journey distributions on each backend’s found PT journeys</h2>', table(distributions), '<img src="distributions.png">',
               '<h2>Paired comparisons on jointly found PT journeys</h2>', table(pairs), '<img src="google_pairs.png">',
               '<p>Pairwise rows use identical trips with PT found in both backends and finite nonnegative values for each metric; denominators can differ by metric. Differences are right minus left. Google and survey journeys may have different query dates, departure choices or routing objectives. Total time may include waiting differently; compare the component sums alongside total time. Google total distance is not inferred from in-vehicle distance. FLASH distance follows stop geometry; eqasim distance follows the mapped network. Differences are diagnostic, not a ground-truth accuracy score.</p>',
               '<h2>Success by distance, border status, detailed survey mode and departure hour</h2>', table(segments),
               '<p>The survey’s PT label also includes planes, taxis and coaches. These remain in the requested cohort and are identified in the detailed-mode rows; a transit-only router may not serve them.</p>',
               '<h2>Files</h2><p>requests.parquet, per-backend routes.parquet, comparison.parquet, success.csv, success_patterns.csv, distributions.csv, pairwise.csv, segments.csv, timings.json and metadata.json preserve the inputs and results for further analysis.</p></body></html>']
    target = output / 'index.html'
    target.write_text('\n'.join(content))
    return str(target)


def table(frame):
    rows = ['<div class="scroll"><table><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in frame.columns) + '</tr>']
    for row in frame.iter_rows():
        cells = [f'{value:.3f}' if isinstance(value, float) else ('—' if value is None else str(value)) for value in row]
        rows.append('<tr>' + ''.join('<td>' + html.escape(value) + '</td>' for value in cells) + '</tr>')
    return '\n'.join([*rows, '</table></div>'])


def timing_table(timings):
    rows = []
    for backend, values in timings.items():
        routing = values.get('routing_seconds', [])
        rows.append({'backend': backend, 'startup_seconds': values.get('startup_seconds', values.get('python_setup_seconds')),
                     'routing_seconds_by_repeat': ', '.join(f'{v:.3f}' for v in routing),
                     'warm_trips_per_second': values.get('requests', 0) / routing[-1] if len(routing) > 1 and routing[-1] else None,
                     'flash_original_preprocessing_seconds': values.get('preprocessing', {}).get('preprocessing_seconds'),
                     'flash_preprocessing_reused': values.get('preprocessing', {}).get('reused')})
    return table(pl.DataFrame(rows))


def figures(routes, timings, output):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([LABELS[b] for b in routes], [frame['journey_found'].sum() / frame.height * 100 for frame in routes.values()])
    ax.set(ylabel='PT journeys found (%)', ylim=(0, 105))
    save(fig, output / 'reachability.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    names, first, warm = [], [], []
    for b, timing in timings.items():
        if timing.get('routing_seconds'):
            names.append(LABELS[b]);first.append(timing['routing_seconds'][0]);warm.append(timing['routing_seconds'][-1])
    positions = np.arange(len(names))
    ax.bar(positions - .2, first, .4, label='First routing call')
    ax.bar(positions + .2, warm, .4, label='Last routing call')
    ax.set(xticks=positions, xticklabels=names, ylabel='Complete routing call (seconds)');ax.legend()
    save(fig, output / 'runtime.png')
    columns = ['pt_total_travel_time_min', 'pt_transfers', 'pt_in_vehicle_distance_km']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for column, ax in zip(columns, axes):
        for backend, frame in routes.items():
            values = frame.filter('journey_found')[column].drop_nulls().to_numpy()
            values = np.sort(values[np.isfinite(values)])
            if len(values): ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=LABELS[backend])
        ax.set(xlabel=METRICS[column], ylabel='Cumulative share', ylim=(0, 1))
        if ax.has_data(): ax.legend(fontsize=8)
    save(fig, output / 'distributions.png')
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    google = routes['google'].filter('journey_found')
    for row, backend in enumerate(('rust', 'raptor', 'flash')):
        paired = routes[backend].filter('journey_found').join(google.select('trip_id', *columns), on='trip_id', suffix='_google', validate='1:1')
        for col, column in enumerate(columns):
            ax = axes[row, col]
            x, y = paired[column + '_google'].to_numpy(), paired[column].to_numpy()
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            ax.scatter(x, y, s=3, alpha=.15)
            limit = max(float(x.max(initial=0)), float(y.max(initial=0)), 1)
            ax.plot([0, limit], [0, limit], '--', color='grey')
            ax.set(xlabel='Google: ' + METRICS[column], ylabel=LABELS[backend], title=f'{len(x):,} matched trips')
    save(fig, output / 'google_pairs.png')


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
