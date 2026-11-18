"""Export a standalone map of Geneva and Zurich airport access/egress trips."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from urllib.request import urlopen

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from synpp.pipeline import process_stages
import yaml

from synthesis.airports import trips as airport_trips
from data.airports.geneva.model import AIRPORTS, resolve
from data.airports._shared.common import file_digest

MODES = ('car', 'pt', 'bike', 'walk', 'car_passenger')
FIELDS = ('airport', 'direction', 'municipality', 'lat', 'lon', 'visitor',
          'access', 'mode', 'departure', 'distance_km', 'ovgk', 'weight', 'airport_access', 'location_source',
          'person_id', 'age', 'sex', 'employed')
TILES = 'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg'


def configure(context):
    context.stage('synthesis.airports.trips')
    context.stage('data.spatial.municipalities')
    context.config('analysis_path')
    context.config('airport_model_airports', default=AIRPORTS)


def execute(context):

    # Reuse the current airport movements and spatial layer without modifying them.

    frame = context.stage('synthesis.airports.trips')
    municipalities, _ = context.stage('data.spatial.municipalities')

    # Embed the data and map library in one file under the configured analysis folder.

    output = Path(context.config('analysis_path')) / 'airports/trips.html'
    return str(render(frame, municipalities, context.config('airport_model_airports'), output,
                      {'source': 'Synpp synthesis.airports.trips'}))


def prepare(frame, municipalities, airports, source):
    required = {'airport', 'direction', 'municipality_id', 'origin_x', 'origin_y',
                'destination_x', 'destination_y', 'is_visitor', 'access_mode', 'mode',
                'departure_time', 'euclidean_distance_m', 'endpoint_ovgk', 'sample_weight', 'airport_access_point', 'location_source',
                'person_id', 'age', 'sex', 'employed'}
    if not required.issubset(frame):
        raise ValueError(f'Airport map input missing columns: {sorted(required - set(frame.columns))}')
    if not frame.airport.isin(['GVA', 'ZRH']).all() or not frame.direction.isin(['to_airport', 'from_airport']).all():
        raise ValueError('Airport map requires known airport and direction labels')
    if (not frame['mode'].isin(MODES).all() or not frame.access_mode.isin(airport_trips.MODES).all()
            or frame.municipality_id.isna().any() or frame.is_visitor.isna().any()):
        raise ValueError('Airport map requires known municipalities, modes and visitor labels')
    residents = frame.loc[~frame.is_visitor]
    if residents.person_id.isna().any() or residents.person_id.duplicated().any():
        raise ValueError('Airport map requires unique assigned resident IDs')
    to = frame.direction.eq('to_airport')
    x = np.where(to, frame.origin_x, frame.destination_x).astype(float)
    y = np.where(to, frame.origin_y, frame.destination_y).astype(float)
    lon, lat = Transformer.from_crs(2056, 4326, always_xy=True).transform(x, y)
    if not np.isfinite(np.column_stack([lat, lon, frame.departure_time, frame.euclidean_distance_m, frame.sample_weight])).all():
        raise ValueError('Airport map cannot display missing coordinates, times, distances or weights')
    rows = [[str(a), int(d), int(m), round(float(la), 6), round(float(lo), 6), int(v),
             airport_trips.MODES.index(ac), MODES.index(mode), round(float(t)),
             round(float(km) / 1000, 3), str(q), round(float(w), 6), str(point), str(location),
             str(person) if pd.notna(person) else None, int(age) if pd.notna(age) else None,
             int(sex) if pd.notna(sex) else None, bool(employed) if pd.notna(employed) else None]
            for a, d, m, la, lo, v, ac, mode, t, km, q, w, point, location, person, age, sex, employed in zip(
                frame.airport, ~to, frame.municipality_id, lat, lon, frame.is_visitor,
                frame.access_mode, frame['mode'], frame.departure_time, frame.euclidean_distance_m,
                frame.endpoint_ovgk, frame.sample_weight, frame.airport_access_point, frame.location_source,
                frame.person_id, frame.age, frame.sex, frame.employed)]
    names = municipalities.drop_duplicates('municipality_id').set_index('municipality_id').municipality_name
    used = set(frame.municipality_id.astype(int))
    swiss = municipalities.loc[municipalities.municipality_id.gt(0)].to_crs(2056)
    outline = swiss.geometry.union_all().simplify(400, preserve_topology=True)
    outline = json.loads(gpd.GeoSeries([outline], crs=2056).to_crs(4326).to_json())
    summary = frame.attrs.get('summary', {})
    airport_points = {a: {kind: dict(point) for kind, point in v['points'].items()}
                      for a, v in summary.get('airport_access_points', {}).items()}
    # Keep markers aligned with the actual trip endpoints, including direct DataFrame inputs.
    for (airport, kind), group in frame.groupby(['airport', 'airport_access_point']):
        first = group.iloc[0]
        towards = first.direction == 'to_airport'
        ax = first.destination_x if towards else first.origin_x
        ay = first.destination_y if towards else first.origin_y
        alon, alat = Transformer.from_crs(2056, 4326, always_xy=True).transform(ax, ay)
        point = airport_points.setdefault(airport, {}).setdefault(kind, {})
        point.update(latitude=float(alat), longitude=float(alon))
        point.setdefault('name', 'PT interchange' if kind == 'pt' else 'Road parking access')
    return {'fields': FIELDS, 'trips': rows, 'municipalities': {str(m): str(names.get(m, m)) for m in used},
            'airports': airports,
            'airport_points': airport_points,
            'access_modes': airport_trips.MODES, 'simulation_modes': MODES,
            'outline': outline, 'summary': summary, 'source': source,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'tile_url': TILES}


def render(frame, municipalities, airports, output, source, assets=None):
    payload = prepare(frame, municipalities, airports, source)
    # The map library is embedded; only optional basemap tiles require the internet.
    if assets is None:
        assets = {}
        for ext in ('js', 'css'):
            with urlopen(f'https://unpkg.com/leaflet@1.9.4/dist/leaflet.{ext}', timeout=30) as response:
                assets[ext] = response.read().decode('utf-8')
    template = Path(__file__).with_name('trips.html.template').read_text()
    replacements = {'__LEAFLET_JS__': assets['js'], '__LEAFLET_CSS__': assets['css'],
                    '__DATA__': json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')}
    for key, value in replacements.items():
        template = template.replace(key, value)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding='utf-8')
    return output


def load_exports(configuration):
    """Read signed trip exports and a matching spatial cache; never execute stages."""
    config = configuration['config']
    folder = resolve(Path(config['data_path']), config.get('airport_trips_folder', 'airports/trips'))
    manifest = json.loads((folder / 'signature.json').read_text())
    frame = airport_trips.load_trips(folder, manifest['signature'])
    if frame is None:
        raise ValueError('Airport trip exports are corrupt or incomplete; run synthesis.airports.trips first')
    if frame.attrs['summary']['settings']['version'] != airport_trips.VERSION:
        raise ValueError('Airport exports predate the current trip model; run synthesis.airports.trips first')
    registry = process_stages([{'descriptor': 'data.spatial.municipalities', 'config': {}}], config)
    stage = next(s for s in registry.values() if s['wrapper'].name == 'data.spatial.municipalities')
    path = Path(configuration['working_directory']) / (stage['hash'] + '.p')
    with path.open('rb') as stream:
        municipalities, _ = pickle.load(stream)  # Trusted local Synpp result only.
    source = {'source': str(folder / 'trips.parquet'), 'sha256': file_digest(folder / 'trips.parquet'),
              'signature': manifest['signature'],
              'modified_utc': datetime.fromtimestamp((folder / 'trips.parquet').stat().st_mtime, timezone.utc).isoformat()}
    return frame, municipalities, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', nargs='?', default='config_dib_10.yml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    configuration = yaml.safe_load(Path(args.configuration).read_text())
    frame, municipalities, source = load_exports(configuration)
    output = args.output or Path(configuration['config']['analysis_path']) / 'airports/trips.html'
    result = render(frame, municipalities, configuration['config'].get('airport_model_airports', AIRPORTS), output, source)
    print(result)
    print('Mapped trips:', len(frame), frame.groupby('airport').size().to_dict())


if __name__ == '__main__':
    main()
