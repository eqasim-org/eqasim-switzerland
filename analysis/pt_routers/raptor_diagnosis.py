"""Summarize controlled SwissRailRaptor tests; never run the pipeline or reroute."""
from pathlib import Path
import argparse
import json

import polars as pl

from analysis.pt_routers.unmatched import html_table

VARIANTS = (
    'baseline', 'original_coordinates', 'access_1500', 'suppress_direct_walk',
    'original_coordinates_access_1500', 'access_1500_suppress_direct_walk',
    'original_coordinates_access_1500_suppress_direct_walk',
)
FOLLOWUP_VARIANTS = ('baseline', 'original_coordinates_access_1500_suppress_direct_walk',
                     'original_coordinates_access_3000', 'original_coordinates_access_3000_suppress_direct_walk')
LABELS = {
    'baseline': 'Current settings', 'original_coordinates': 'Original coordinates only',
    'access_1500': 'Larger access-stop search only',
    'suppress_direct_walk': 'Suppress direct-walk selection only',
    'original_coordinates_access_1500': 'Original coordinates + larger search',
    'access_1500_suppress_direct_walk': 'Larger search + suppress direct walking',
    'original_coordinates_access_1500_suppress_direct_walk': 'All three changes',
    'original_coordinates_access_3000': 'Original coordinates + 3 km search',
    'original_coordinates_access_3000_suppress_direct_walk': 'Original coordinates + 3 km search + suppress direct walking',
}


def summarize_rows(frame, variants=VARIANTS):
    """Require a complete experiment and a baseline reproducing the saved results."""
    if set(frame['variant'].unique()) != set(variants):
        raise ValueError('Incomplete diagnostic variants')
    identifiers = frame['trip_id'].n_unique()
    sizes = frame.group_by('variant').agg(pl.len().alias('rows'), pl.col('trip_id').n_unique().alias('ids'))
    if sizes.filter((pl.col('rows') != identifiers) | (pl.col('ids') != identifiers)).height:
        raise ValueError('Incomplete or duplicate requests within diagnostic variants')
    baseline = frame.filter(pl.col('variant') == 'baseline')
    if baseline.filter(pl.col('available') != pl.col('expected_java')).height:
        raise ValueError('Diagnostic baseline does not reproduce saved Java availability')
    unmatched = baseline.filter(pl.col('cohort') == 'flash_only')
    metadata = {'flash_only_trips_tested': unmatched.height,
                'matched_controls': baseline.filter(pl.col('cohort') == 'both_control').height,
                'long_5km_cases': unmatched.filter(pl.col('euclidean_distance_km') >= 5).height,
                'long_10km_cases': unmatched.filter(pl.col('euclidean_distance_km') >= 10).height,
                'zero_distance_cases': unmatched.filter(pl.col('euclidean_distance_km') == 0).height,
                'baseline_observations': unmatched.group_by('status').len().to_dicts(),
                'baseline_reproduced': True}
    rows = []
    for variant in variants:
        case = frame.filter(pl.col('variant') == variant)
        recovered = case.filter((pl.col('cohort') == 'flash_only') & pl.col('available'))
        rows.append({'variant': variant, 'search_radius_m': case['search_radius'][0],
                     'direct_walk_factor': case['direct_walk_factor'][0],
                     'transfer_walk_limit_m': case['transfer_walk_limit'][0],
                     'recovered_flash_only': recovered.height,
                     'recovered_5km': recovered.filter(pl.col('euclidean_distance_km') >= 5).height,
                     'recovered_10km': recovered.filter(pl.col('euclidean_distance_km') >= 10).height,
                     'controls_with_pt': case.filter((pl.col('cohort') == 'both_control') & pl.col('available')).height})
    flags = frame.filter(pl.col('cohort') == 'flash_only').select('trip_id', 'variant', 'available').pivot(
        on='variant', index='trip_id', values='available')
    details = unmatched.select('trip_id', 'euclidean_distance_km', pl.col('status').alias('baseline_status'),
                               'origin_snap_m', 'destination_snap_m', 'same_snapped_link').join(flags, on='trip_id', validate='1:1')
    return metadata, pl.DataFrame(rows), details


def summarize(output):
    output = Path(output)
    directory = output / 'java_diagnosis'
    frame = pl.read_csv(directory / 'counterfactuals.csv', null_values=['NaN', 'Infinity'], schema_overrides={'trip_id': pl.String})
    metadata, counts, details = summarize_rows(frame)
    counts.write_csv(directory / 'recovery_summary.csv')
    details.write_csv(directory / 'per_trip_recovery.csv')
    source = pl.read_parquet(output / 'flash_only_trips.parquet')
    long_cases = source.filter(pl.col('euclidean_distance_km') >= 5).select(
        'trip_id', 'origin_latitude', 'origin_longitude', 'destination_latitude', 'destination_longitude',
        'euclidean_distance_km', 'pt_total_travel_time_min', 'pt_modes', 'longest_transfer_walk_km')
    radius_case = frame.filter(pl.col('variant') == 'access_1500').select(
        'trip_id', pl.col('available').alias('java_pt_with_1500m_search'),
        pl.col('route_time_min').alias('java_time_with_1500m_search_min'))
    long_cases.join(radius_case, on='trip_id', validate='1:1').sort('euclidean_distance_km', descending=True).write_csv(directory / 'long_distance_cases.csv')
    if (directory / 'followup.csv').exists():
        followup = pl.read_csv(directory / 'followup.csv', null_values=['NaN', 'Infinity'], schema_overrides={'trip_id': pl.String})
        follow_meta, follow_counts, _ = summarize_rows(followup, FOLLOWUP_VARIANTS)
        follow_counts.write_csv(directory / 'followup_summary.csv')
        metadata['followup'] = follow_meta
    (directory / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    original = json.loads((output / 'summary.json').read_text())
    original['java_raptor_diagnosis'] = metadata
    from analysis.pt_routers.report import write_index
    write_index(output, original, original.get('warnings', []))
    return output / 'index.html'


def render(output, metadata):
    directory = Path(output) / 'java_diagnosis'
    counts = pl.read_csv(directory / 'recovery_summary.csv').with_columns(pl.col('variant').replace_strict(LABELS))
    page = '<section id="java-raptor-diagnosis"><h2>Controlled SwissRailRaptor parameter diagnosis</h2>'
    page += f'<p>Tested all {metadata["flash_only_trips_tested"]:,} FLASH-only requests and {metadata["matched_controls"]} successful control requests using the same Java runtime, network, schedule and Eqasim routing costs. The baseline reproduces the saved availability for every tested request. No pipeline, simulation or FLASH preprocessing was run.</p>'
    observations = ', '.join(f'{row["len"]:,} {row["status"]}' for row in metadata['baseline_observations'])
    page += f'<p>Baseline SwissRailRaptor returned: <strong>{observations}</strong> for the FLASH-only cohort. This is the raw result before the Java wrapper classifies it as unavailable.</p>'
    page += '<h3>Recovered PT routes</h3><p>Each row is an independent counterfactual; recovered counts overlap and must not be added. Direct-walk suppression uses a diagnostic factor of 1,000,000 rather than the configured factor of 3. The transfer walking limit remains unchanged in every test.</p>'
    page += html_table(counts, {'variant': 'Change from baseline', 'search_radius_m': 'Search radius (m)',
                              'direct_walk_factor': 'Direct-walk factor', 'transfer_walk_limit_m': 'Transfer limit (m)',
                              'recovered_flash_only': 'Recovered: all FLASH-only', 'recovered_5km': 'Recovered: OD ≥5 km',
                              'recovered_10km': 'Recovered: OD ≥10 km', 'controls_with_pt': 'Controls with PT'})
    access = counts.filter(pl.col('variant') == LABELS['access_1500']).row(0, named=True)
    page += f'<p><strong>Confirmed access-search effect:</strong> the larger search alone recovers {access["recovered_5km"]} of {metadata["long_5km_cases"]} cases with OD distance ≥5 km, and {access["recovered_10km"]} of {metadata["long_10km_cases"]} cases ≥10 km.</p>'
    page += '<p>The installed stop finder searches the initial radius and expands around the nearest stop only when fewer than two stop facilities are found. It does not expand again merely because the selected stops fail to yield a useful PT route. A nearby but unsuitable stop can therefore prevent consideration of a useful farther stop.</p>'
    page += '<p>The installed SwissRailRaptor compares PT with direct walking using generalized cost. With <code>intermodalLegOnlyHandling=forbid</code>, a selected walk-only alternative can become an empty result. Increasing the direct-walk factor can expose a PT alternative for mode-choice estimation; it does not prove that transit is preferable to walking. Keep the final mode comparison in the mode-choice model.</p>'
    page += '<h3>Long-distance cases</h3>'
    long_cases = pl.read_csv(directory / 'long_distance_cases.csv', schema_overrides={'trip_id': pl.String})
    page += html_table(long_cases.select('trip_id', 'euclidean_distance_km', 'pt_total_travel_time_min',
                                         'java_pt_with_1500m_search', 'java_time_with_1500m_search_min', 'pt_modes'),
                       {'trip_id': 'Trip', 'euclidean_distance_km': 'OD distance (km)',
                        'pt_total_travel_time_min': 'FLASH time (min)', 'java_pt_with_1500m_search': 'PT found with larger search',
                        'java_time_with_1500m_search_min': 'Java time (min)', 'pt_modes': 'FLASH modes'})
    if 'followup' in metadata:
        page += f'<h3>Follow-up on the remaining {metadata["followup"]["flash_only_trips_tested"]} cases</h3>'
        page += '<p>This smaller cohort was not recovered by all three changes at 1,500 m. It is tested separately with a 3 km endpoint search; denominators differ from the table above.</p>'
        follow = pl.read_csv(directory / 'followup_summary.csv').with_columns(pl.col('variant').replace_strict(LABELS))
        page += html_table(follow)
    page += '<h3>Recommended changes to evaluate</h3><p>Start with <code>transitRouter.searchRadius=1500</code> in the private mode-choice routing config, without increasing transfer-walk limits. Preserve the original PT endpoint coordinates using <code>FacilitiesUtils.wrapLinkAndCoord</code>, rather than replacing them with the nearest network link centre. If the objective is a transit-containing alternative conditional on PT, suppress direct-walk selection or explicitly choose a transit-containing journey, while retaining walking as a separate mode-choice alternative. Scope such changes to mode-choice routing instead of changing simulation-wide route choice implicitly.</p>'
    page += f'<p>{metadata["zero_distance_cases"]} FLASH-only requests have identical original origin and destination coordinates; recovery of a circular PT journey should not be treated as a necessary correction for these requests.</p>'
    page += '<p><a href="java_diagnosis/counterfactuals.csv">All counterfactual results</a> · <a href="java_diagnosis/per_trip_recovery.csv">Per-trip recovery flags</a> · <a href="java_diagnosis/long_distance_cases.csv">Long-distance cases with coordinates</a></p></section>'
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', help='Completed PT comparison output directory')
    print(summarize(parser.parse_args().output))
