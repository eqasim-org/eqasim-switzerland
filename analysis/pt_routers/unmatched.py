"""Saved-result diagnostics for journeys found by FLASH but unavailable in Java."""
from pathlib import Path
import html
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pyproj import Transformer

from analysis.pt_routers.report import gtfs_mode, reachable, save_figure

SAMPLE_SIZE = 50
SAMPLE_FILENAME = f'flash_only_sample_{SAMPLE_SIZE}.csv'
METRICS = {
    'pt_total_travel_time_min': 'Total travel time (min)',
    'pt_in_vehicle_time_min': 'In-vehicle time (min)',
    'pt_access_egress_time_min': 'Access + egress walking (min)',
    'pt_transfer_time_min': 'Transfers: walking + waiting (min)',
    'initial_waiting_min': 'Initial waiting, derived (min)',
    'pt_total_distance_km': 'Total geometric distance (km)',
    'pt_in_vehicle_distance_km': 'In-vehicle geometric distance (km)',
    'walking_distance_km': 'Total geometric walking distance (km)',
    'transfer_walk_min': 'Walking during transfers (min)',
    'longest_transfer_walk_km': 'Longest walking transfer (km)',
    'pt_transfers': 'Transfers',
    'euclidean_distance_km': 'Origin–destination straight-line distance (km)',
    'direct_walk_estimate_min': 'Estimated direct walking time (min)',
}


def analyse(output, requests, seed, java_path, leg_paths, walk_speed, walk_detour):
    """Build cohort tables without invoking routing or changing any source caches."""
    output = Path(output)
    java, flash = (pl.read_parquet(output / f'{name}_routes.parquet') for name in ('java', 'flash'))
    statuses = java.select('trip_id').with_columns(pl.Series('java_reachable', reachable(java))).join(
        flash.select('trip_id').with_columns(pl.Series('flash_reachable', reachable(flash))), on='trip_id', validate='1:1')
    cohorts = statuses.filter(pl.col('flash_reachable')).with_columns(
        pl.when(pl.col('java_reachable')).then(pl.lit('both')).otherwise(pl.lit('flash_only')).alias('cohort'))
    request_columns = ['trip_id', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']
    request_columns += [name for name in ('person_id', 'trip_index') if name in requests.columns]
    trips = cohorts.select('trip_id', 'cohort').join(flash, on='trip_id', validate='1:1').join(
        requests.select(request_columns), on='trip_id', how='left', validate='1:1')
    if trips.filter(pl.any_horizontal(pl.col(request_columns[1:6]).is_null())).height:
        raise ValueError('FLASH-only analysis requires coordinates and departure times for every trip')

    # Walking legs are separate from vehicle boardings; endpoint waiting placeholders are excluded.

    columns = ['trip_id', 'leg', 'from', 'to', 'departure', 'arrival', 'vehicle', 'distance_m', 'gtfs_route_type']
    frames = [pl.read_parquet(path, columns=columns) for path in leg_paths]
    legs = pl.concat(frames) if frames else pl.DataFrame(schema={
        'trip_id': pl.String, **{c: pl.Int64 for c in columns[1:] if c != 'distance_m'}, 'distance_m': pl.Float64})
    if legs.join(statuses.select('trip_id'), on='trip_id', how='anti').height:
        raise ValueError('Selected FLASH legs contain unknown trip IDs')
    legs = legs.join(cohorts.select('trip_id', 'cohort'), on='trip_id', how='inner', validate='m:1').with_columns(
        pl.col('leg').min().over('trip_id').alias('first_leg'),
        pl.col('leg').max().over('trip_id').alias('last_leg'))
    vehicle = legs.filter(pl.col('vehicle') == 1)
    modes = {value: gtfs_mode(value) for value in vehicle['gtfs_route_type'].unique().to_list()}
    vehicle = vehicle.with_columns(pl.col('gtfs_route_type').replace_strict(modes, return_dtype=pl.String).alias('pt_mode'))
    walking = legs.filter((pl.col('vehicle') == 0) & (pl.col('from') != pl.col('to'))
                          & (pl.col('leg') != pl.col('first_leg')) & (pl.col('leg') != pl.col('last_leg')))
    walking = walking.group_by('trip_id').agg(
        ((pl.col('arrival') - pl.col('departure')).sum() / 60).alias('transfer_walk_min'),
        (pl.col('distance_m').max() / 1000).alias('longest_transfer_walk_km'))
    used_modes = vehicle.group_by('trip_id').agg(pl.col('pt_mode').unique().sort().str.join(' + ').alias('pt_modes'),
                                               pl.len().alias('boardings'))
    trips = trips.join(walking, on='trip_id', how='left', validate='1:1').join(
        used_modes, on='trip_id', how='left', validate='1:1').with_columns(
            pl.col('transfer_walk_min', 'longest_transfer_walk_km').fill_null(0.),
            pl.col('pt_modes').fill_null('not available'), pl.col('boardings').fill_null(0))
    trips = trips.with_columns(
        ((pl.col('origin_x') - pl.col('destination_x')).pow(2) +
         (pl.col('origin_y') - pl.col('destination_y')).pow(2)).sqrt().truediv(1000).alias('euclidean_distance_km'),
        (pl.col('pt_total_distance_km') - pl.col('pt_in_vehicle_distance_km')).clip(lower_bound=0).alias('walking_distance_km'),
        (pl.col('pt_total_travel_time_min') - pl.col('pt_in_vehicle_time_min') - pl.col('pt_access_egress_time_min')
         - pl.col('pt_transfer_time_min')).clip(lower_bound=0).alias('initial_waiting_min'),
        (pl.col('departure_time') / 3600).floor().cast(pl.Int64).alias('departure_hour'),
    ).with_columns((pl.col('euclidean_distance_km') * 1000 * walk_detour / walk_speed / 60).alias('direct_walk_estimate_min'))

    # Compare FLASH-only trips with successful journeys from the same FLASH engine.

    stats = []
    for cohort in ('flash_only', 'both'):
        subset = trips.filter(pl.col('cohort') == cohort)
        for metric in METRICS:
            values = subset[metric].to_numpy()
            values = values[np.isfinite(values)]
            stats.append({'cohort': cohort, 'metric': metric, 'n': len(values),
                          'mean': float(values.mean()) if len(values) else None,
                          'median': float(np.median(values)) if len(values) else None,
                          'p90': float(np.quantile(values, .9)) if len(values) else None,
                          'max': float(values.max()) if len(values) else None})
    pl.DataFrame(stats).write_csv(output / 'flash_only_statistics.csv')
    mode_stats = vehicle.group_by('cohort', 'pt_mode').agg(
        pl.col('trip_id').n_unique().alias('trips'), pl.len().alias('boardings'),
        (pl.col('distance_m').sum() / 1000).alias('passenger_km'),
        ((pl.col('arrival') - pl.col('departure')).sum() / 60).alias('in_vehicle_minutes')).sort('cohort', 'pt_mode')
    counts = trips.group_by('cohort').len().rename({'len': 'cohort_size'})
    mode_stats = mode_stats.join(counts, on='cohort', how='left', validate='m:1').with_columns(
        (100 * pl.col('trips') / pl.col('cohort_size')).alias('share_of_trips_percent'))
    mode_stats.write_csv(output / 'flash_only_pt_modes.csv')
    flags = trip_signals(trips)
    flags.write_csv(output / 'flash_only_signals.csv')
    trips.group_by('cohort', 'departure_hour').len().sort('cohort', 'departure_hour').write_csv(output / 'flash_only_departure_hours.csv')

    # Preserve reproducible random examples, with both projected and geographic coordinates.

    only = trips.filter(pl.col('cohort') == 'flash_only').sort('trip_id')
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    for endpoint in ('origin', 'destination'):
        lon, lat = transformer.transform(only[f'{endpoint}_x'].to_numpy(), only[f'{endpoint}_y'].to_numpy())
        only = only.with_columns(pl.Series(f'{endpoint}_longitude', lon), pl.Series(f'{endpoint}_latitude', lat))
    only.write_parquet(output / 'flash_only_trips.parquet')
    only.write_csv(output / 'flash_only_trips.csv')
    examples = only.sample(n=min(SAMPLE_SIZE, only.height), seed=seed, shuffle=True)
    examples.write_csv(output / SAMPLE_FILENAME)
    reasons = java_results(java_path, only.select('trip_id'))
    reasons.write_csv(output / 'flash_only_java_results.csv')
    draw_distributions(trips, output)
    metadata = {'trips': only.height, 'reference_trips': trips.height - only.height,
                'share_of_requests_percent': 100 * only.height / statuses.height if statuses.height else 0.,
                'sample_size': examples.height, 'sample_seed': seed,
                'walking_speed_m_per_s': walk_speed, 'walking_detour_factor': walk_detour,
                'java_has_per_trip_error_reason': False}
    (output / 'flash_only_summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def trip_signals(trips):
    signals = {
        'No transfers': pl.col('pt_transfers') == 0,
        'Three or more transfers': pl.col('pt_transfers') >= 3,
        'In-vehicle distance below 2 km': pl.col('pt_in_vehicle_distance_km') < 2,
        'Access + egress walking above 30 min': pl.col('pt_access_egress_time_min') > 30,
        'Walking transfer above 1 km': pl.col('longest_transfer_walk_km') > 1,
        'Journey above 120 min': pl.col('pt_total_travel_time_min') > 120,
        'PT slower than estimated direct walking': pl.col('pt_total_travel_time_min') > pl.col('direct_walk_estimate_min'),
        'Departure after 24:00': pl.col('departure_time') >= 86400,
        'Missing selected transit legs': pl.col('boardings') == 0,
    }
    rows = []
    for cohort in ('flash_only', 'both'):
        subset = trips.filter(pl.col('cohort') == cohort)
        for name, predicate in signals.items():
            count = subset.filter(predicate).height
            rows.append({'cohort': cohort, 'signal': name, 'trips': count,
                         'share_percent': 100 * count / subset.height if subset.height else 0.})
    return pl.DataFrame(rows)


def java_results(path, identifiers):
    raw = pl.read_csv(path, columns=['identifier', 'travel_time', 'route'],
                      schema_overrides={'identifier': pl.String, 'travel_time': pl.Float64, 'route': pl.String},
                      null_values=['NaN', 'nan', 'Infinity']).rename({'identifier': 'trip_id'})
    raw = raw.with_columns(pl.lit(True).alias('row_returned'))
    joined = identifiers.join(raw, on='trip_id', how='left', validate='1:1')
    status = pl.when(pl.col('row_returned').is_null()).then(pl.lit('No Java result row'))
    status = status.when(pl.col('travel_time').is_finite().fill_null(False)).then(
        pl.lit('Finite raw time; unavailable after Python processing (check other components/fare)'))
    status = status.when(pl.col('route').fill_null('').str.strip_chars().is_in(['', 'null', '{}', '[]'])).then(
        pl.lit('Java row returned without itinerary or finite travel time'))
    joined = joined.with_columns(status.otherwise(pl.lit('Incomplete/non-finite Java itinerary')).alias('observation'))
    return joined.group_by('observation').len().rename({'len': 'trips'}).sort('trips', descending=True)


def draw_distributions(trips, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, label, axis in zip(['pt_total_travel_time_min', 'pt_total_distance_km', 'pt_transfers'],
                                   ['Travel time (min)', 'Geometric distance (km)', 'Transfers'], axes.flat):
        all_values = trips[column].to_numpy()
        bins = np.histogram_bin_edges(all_values[np.isfinite(all_values)], bins=30) if len(all_values) else np.arange(2)
        if column == 'pt_transfers': bins = np.arange(-.5, max(float(np.max(all_values, initial=0)), 1) + 1.5)
        for cohort, name in [('flash_only', 'FLASH only'), ('both', 'Both routers (FLASH journey)')]:
            values = trips.filter(pl.col('cohort') == cohort)[column].to_numpy()
            if len(values): axis.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                                      histtype='step', linewidth=2, label=name)
        axis.set(xlabel=label, ylabel='Share of cohort (%)')
        if len(all_values): axis.legend(fontsize=8)
    axis = axes[1, 1]
    for cohort, name in [('flash_only', 'FLASH only'), ('both', 'Both routers')]:
        subset = trips.filter(pl.col('cohort') == cohort)
        if subset.height:
            hours = subset.group_by('departure_hour').len().sort('departure_hour')
            axis.plot(hours['departure_hour'], hours['len'] * 100 / subset.height, marker='.', label=name)
    axis.set(xlabel='Departure hour (hours beyond 24 retained)', ylabel='Share of cohort (%)')
    if trips.height: axis.legend(fontsize=8)
    save_figure(figure, output, 'flash_only_comparison.png')


def html_table(frame, labels=None):
    labels = labels or {}
    header = ''.join(f'<th>{html.escape(labels.get(col, col))}</th>' for col in frame.columns)
    rows = []
    for row in frame.iter_rows():
        cells = []
        for value in row:
            text = '—' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
            cells.append(f'<td>{html.escape(text)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '<div style="overflow-x:auto"><table border="1" cellpadding="7"><thead><tr>' + header + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def clock_time(seconds):
    if seconds is None: return '—'
    value = int(seconds)
    return f'{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}'


def render(output, metadata):
    output = Path(output)
    page = f'<section id="flash-only"><h2>Trips routed by FLASH but unavailable in Java</h2><p><strong>{metadata["trips"]:,} trips ({metadata["share_of_requests_percent"]:.2f}% of requests).</strong> Reference: {metadata["reference_trips"]:,} trips routed by both backends. All journey statistics below use FLASH, including for the reference group.</p>'
    reasons = pl.read_csv(output / 'flash_only_java_results.csv')
    page += '<h3>What Java returned</h3>' + html_table(reasons)
    page += '<p>The saved Java CSV does not provide a per-trip failure reason. An empty result does not prove there is no possible transit journey. Candidate explanations include walk-only fallback handling, different access-stop searches, network-link snapping, transfer walking rules and timetable coverage. These require targeted Java diagnostics to confirm; the indicators below are not causal classifications.</p>'
    if not metadata['trips']:
        return page + '<p>No FLASH-only trips in this comparison.</p></section>'
    stats = pl.read_csv(output / 'flash_only_statistics.csv')
    only = stats.filter(pl.col('cohort') == 'flash_only').select('metric', 'mean', 'median', 'p90', 'max')
    reference = stats.filter(pl.col('cohort') == 'both').select('metric', pl.col('mean').alias('both_mean'))
    comparison = only.join(reference, on='metric', validate='1:1').with_columns(pl.col('metric').replace_strict(METRICS))
    page += '<h3>Time, distance and transfer statistics</h3>' + html_table(comparison, {
        'metric': 'FLASH journey variable', 'mean': 'FLASH-only mean', 'median': 'Median', 'p90': '90th percentile',
        'max': 'Maximum', 'both_mean': 'Both-router mean'})
    page += '<p>Distances are geometric estimates. Initial waiting is derived as total time minus in-vehicle, access/egress and transfer time. Direct walking is an estimate from the original endpoint coordinates and FLASH walking factors, not an observed Java fallback.</p>'
    page += '<img src="flash_only_comparison.png" style="max-width:100%">'
    flags = pl.read_csv(output / 'flash_only_signals.csv')
    comparison = flags.filter(pl.col('cohort') == 'flash_only').drop('cohort').join(
        flags.filter(pl.col('cohort') == 'both').select('signal', pl.col('share_percent').alias('both_share_percent')),
        on='signal', validate='1:1')
    page += '<h3>Indicators to investigate</h3>' + html_table(comparison, {
        'signal': 'Indicator', 'trips': 'FLASH-only trips', 'share_percent': 'FLASH-only (%)', 'both_share_percent': 'Both-router (%)'})
    lookup = {row['signal']: row for row in comparison.iter_rows(named=True)}
    walking = lookup['PT slower than estimated direct walking']
    direct = lookup['No transfers']
    page += f'<p><strong>Observed pattern:</strong> {direct["share_percent"]:.1f}% of FLASH-only trips have no transfers. For {walking["share_percent"]:.1f}%, FLASH PT takes longer than the direct-walking estimate, compared with {walking["both_share_percent"]:.1f}% in the both-router group. The inspected Java PT wrapper discards nonzero walk-only fallback results. This makes walking-fallback handling a useful first hypothesis to investigate, but the saved results do not confirm that mechanism for individual trips.</p>'
    modes = pl.read_csv(output / 'flash_only_pt_modes.csv')
    page += '<h3>Public transport used</h3><p>A trip may use several modes, so trip percentages need not sum to 100. Kilometres below are sampled passenger-leg kilometres.</p>'
    page += html_table(modes.filter(pl.col('cohort') == 'flash_only').drop('cohort', 'cohort_size'))
    examples = pl.read_csv(output / SAMPLE_FILENAME, schema_overrides={'trip_id': pl.String})
    records = []
    for row in examples.iter_rows(named=True):
        records.append({'Trip ID': row['trip_id'], 'Departure': clock_time(row['departure_time']),
                        'Origin (latitude, longitude)': f"{row['origin_latitude']:.6f}, {row['origin_longitude']:.6f}",
                        'Destination (latitude, longitude)': f"{row['destination_latitude']:.6f}, {row['destination_longitude']:.6f}",
                        'Time (min)': row['pt_total_travel_time_min'], 'Distance (km)': row['pt_total_distance_km'],
                        'Transfers': int(row['pt_transfers']), 'PT modes': row['pt_modes']})
    page += f'<h3>Random origin–destination examples ({metadata["sample_size"]} trips)</h3><p>Uniform sample without replacement; seed {metadata["sample_seed"]}. Coordinates are WGS84, in latitude–longitude order. Departure times are seconds from the service-day start, displayed as HH:MM:SS; hours may exceed 24.</p>'
    page += html_table(pl.DataFrame(records))
    page += f'<p><a href="{SAMPLE_FILENAME}">Download these examples</a> (includes EPSG:2056 coordinates, geographic coordinates and all diagnostics). <a href="flash_only_trips.csv">Download all FLASH-only trips</a>.</p></section>'
    return page
