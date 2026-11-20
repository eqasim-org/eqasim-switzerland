"""Reviewed, mode-specific airport access points stored alongside source data."""

import json
import math

DEFAULT_FILES = {'GVA': 'airports/geneva/data_from_websites/airport_access_points.json',
                 'ZRH': 'airports/zurich/data_from_websites/airport_access_points.json'}


def read_access_points(sources):
    points, evidence = {}, {}
    for airport in ('GVA', 'ZRH'):
        path = sources[f'{airport}_access']
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or data.get('airport') != airport:
            raise ValueError(f'Invalid airport access-point metadata: {path}')
        if set(data.get('points', {})) != {'pt', 'road'}:
            raise ValueError(f'Airport access points require pt and road entries: {path}')
        for kind, point in data['points'].items():
            lon, lat = point.get('longitude'), point.get('latitude')
            if (isinstance(lon, bool) or isinstance(lat, bool) or not isinstance(lon, (int, float)) or
                    not isinstance(lat, (int, float)) or not math.isfinite(lon) or not math.isfinite(lat) or
                    not -180 <= lon <= 180 or not -90 <= lat <= 90):
                raise ValueError(f'Invalid {airport} {kind} longitude/latitude in {path}')
        points[airport] = data['points']
        evidence[airport] = data
    return points, evidence
