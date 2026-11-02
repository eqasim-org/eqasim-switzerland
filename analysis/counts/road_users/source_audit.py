"""Read-only comparison of traced cross-border trips with their source caches."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


def audit_cross_border(output, original, renumbered, od, crossing_points):
    """Audit only records whose exported home, age and crossing identify the cache.

    This does not change OD assignment. Eligibility is recomputed using stored
    OD endpoints as a distance-policy diagnostic, not a rerouted counterfactual.
    """
    output = Path(output)
    if len(original) != len(renumbered) or not np.allclose(original.home_x, renumbered.home_x) or not np.allclose(original.home_y, renumbered.home_y) or not np.array_equal(original.age, renumbered.age):
        raise ValueError('Cannot establish the cached population ID correspondence')
    mapping = renumbered[['person_id', 'age', 'home_x', 'home_y']].copy()
    mapping['cross_border_person_id'] = original.person_id.to_numpy()
    mapping.person_id = mapping.person_id.astype(str)
    mapping = mapping.rename(columns={'person_id': 'person', 'age': 'cached_age', 'home_x': 'cached_home_x', 'home_y': 'cached_home_y'})
    profiles = pd.read_csv(output / 'corridor_trip_profiles.csv', dtype={'person': str})
    profiles = profiles[profiles.group.eq('Other cross-border') & profiles.cohort.notna()].copy()
    profiles = profiles.merge(mapping, on='person', how='left', validate='many_to_one')
    fields = ['cross_border_person_id', 'interview_place', 'interview_point_id', 'interview_geometry_point',
              'residence_x', 'residence_y', 'destination_x', 'destination_y', 'origin_is_projected',
              'origin_country_raw', 'destination_country_raw', 'label']
    profiles = profiles.merge(od[fields], on='cross_border_person_id', how='left', validate='many_to_one', suffixes=('', '_od'))
    features = dict(np.load(output / 'person_features.npz'))
    groups = {(place, label): frame for (place, label), frame in crossing_points.groupby(['interview_place', 'label'])}
    to_wgs = Transformer.from_crs(2056, 4326, always_xy=True)
    records = []
    for row in profiles.itertuples():
        candidates = groups.get((row.interview_place, 'road'))
        if candidates is None or pd.isna(row.residence_x) or pd.isna(row.destination_x_od):
            continue
        border_x = row.origin_x if row.origin_purpose == 'border' else row.destination_x if row.destination_purpose == 'border' else np.nan
        border_y = row.origin_y if row.origin_purpose == 'border' else row.destination_y if row.destination_purpose == 'border' else np.nan
        if not np.isfinite([border_x, border_y, row.cached_home_x, row.cached_home_y]).all():
            continue
        point = row.interview_geometry_point
        home_delta = np.hypot(features['home_x'][row.person_idx] - row.cached_home_x, features['home_y'][row.person_idx] - row.cached_home_y)
        crossing_delta = np.hypot(border_x - point.x, border_y - point.y)
        verified = home_delta <= 2 and crossing_delta <= 2 and row.age == row.cached_age
        start = shapely.Point(row.residence_x, row.residence_y)
        end = shapely.Point(row.destination_x_od, row.destination_y_od)
        distance = candidates.geometry.distance(start).to_numpy() + candidates.geometry.distance(end).to_numpy()
        best = int(distance.argmin())
        actual = start.distance(point) + end.distance(point)
        eligible = distance <= distance.min() * 1.2
        # Demonstrate the unit mismatch with the same endpoints converted back
        # to longitude/latitude while candidate geometries remain in metres.
        lon0, lat0 = to_wgs.transform(start.x, start.y)
        lon1, lat1 = to_wgs.transform(end.x, end.y)
        wrong = candidates.geometry.distance(shapely.Point(lon0, lat0)).to_numpy() + candidates.geometry.distance(shapely.Point(lon1, lat1)).to_numpy()
        records.append({'person': row.person, 'sequence': row.sequence, 'cohort': row.cohort,
            'cross_border_person_id': row.cross_border_person_id, 'interview_place': row.interview_place,
            'origin_country': row.origin_country_raw, 'destination_country': row.destination_country_raw,
            'source_label': row.label, 'origin_projected': bool(row.origin_is_projected),
            'export_cache_verified': bool(verified), 'home_delta_metres': float(home_delta),
            'crossing_delta_metres': float(crossing_delta), 'assigned_point': row.interview_point_id,
            'nearest_distance_point': candidates.iloc[best].border_crossing_point_id,
            'candidate_points': len(candidates), 'wrong_crs_eligible': int((wrong <= wrong.min() * 1.2).sum()),
            'metric_eligible': int(eligible.sum()), 'assigned_outside_metric_support': bool(actual > distance.min() * 1.2),
            'extra_geometric_distance_km': float((actual - distance.min()) / 1000)})
    result = pd.DataFrame(records)
    result.to_csv(output / 'cross_border_source_audit.csv', index=False)
    verified = result[result.export_cache_verified & ~result.origin_projected]
    summary = {'audited_trips': len(result), 'verified_nonprojected_trips': len(verified),
        'coordinate_system_evidence': 'Survey STARTORTORTLONGITUDE/LATITUDE and ZIELORTORTLONGITUDE/LATITUDE are read as degrees; read_2021_data.sample_point compares them directly with EPSG:2056 candidate geometries.',
        'eligibility_check': 'Stored OD endpoints in EPSG:2056, using the existing 1.2-times-minimum geometric-distance rule; not a network reroute.',
        'by_cohort': {}, 'limits': 'Source-cache evidence is included only after exported home, age and crossing agree. Stored OD endpoints may differ from the original raw survey coordinates within their destination zone.'}
    for cohort, frame in verified.groupby('cohort'):
        summary['by_cohort'][cohort] = {'trips': len(frame), 'unique_agents': frame.person.nunique(),
            'interview_groups': frame.interview_place.value_counts().to_dict(),
            'assigned_outside_metric_support': int(frame.assigned_outside_metric_support.sum()),
            'median_extra_geometric_km': float(frame.extra_geometric_distance_km.median()),
            'max_extra_geometric_km': float(frame.extra_geometric_distance_km.max())}
    # Pandas counts are converted explicitly so the audit is plain JSON.
    for record in summary['by_cohort'].values():
        record['unique_agents'] = int(record['unique_agents'])
    (output / 'cross_border_source_audit.json').write_text(json.dumps(summary, indent=2))
    return summary
