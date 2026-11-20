"""Build an interactive arrival-mode comparison for Geneva and Vaud municipalities.

Run with --config and --simulation-path pointing to an analysed simulation snapshot.
Survey inputs are read from the provenance of its existing mode-share analysis.
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
import yaml

MODES = ['car', 'pt', 'walk', 'bike', 'car_passenger']
COHORTS = ['microcensus', 'simulation_swiss', 'simulation_all']
TILES = 'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg'
LOG = logging.getLogger(__name__)


def assign_municipalities(frame, x, y, municipalities):
    """Assign each destination once; shared-boundary ties use the lowest BFS ID."""
    coordinates = frame[[x, y]].to_numpy(dtype=float)
    points = shapely.points(coordinates)
    pairs = shapely.STRtree(municipalities.geometry.array).query(points, predicate='intersects')
    ids = municipalities.municipality_id.to_numpy()
    order = np.lexsort((ids[pairs[1]], pairs[0]))
    point_indices, polygon_indices = pairs[:, order]
    first = np.r_[True, np.diff(point_indices) != 0] if len(order) else np.array([], dtype=bool)
    result = frame.iloc[point_indices[first]].copy()
    result['municipality_id'] = ids[polygon_indices[first]]
    return result


def empty_statistics(municipalities):
    return {int(identifier): {cohort: {'counts': np.zeros(len(MODES), dtype=int),
                                     'weights': np.zeros(len(MODES)), 'people': set()}
                             for cohort in COHORTS}
            for identifier in municipalities.municipality_id}


def accumulate(statistics, frame, cohort, person_column, weight_column=None):
    for identifier, group in frame.groupby('municipality_id'):
        record = statistics[int(identifier)][cohort]
        record['counts'] += group['mode'].value_counts().reindex(MODES, fill_value=0).to_numpy()
        weights = (group.groupby('mode')[weight_column].sum() if weight_column
                   else group['mode'].value_counts())
        record['weights'] += weights.reindex(MODES, fill_value=0).to_numpy()
        record['people'].update(group[person_column].unique())


def normalize_modes(values):
    return values.str.removesuffix('_loop').replace({'remote_walk': 'walk', 'transit_walk': 'walk'})


def prepare_survey(trips, persons, excluded):
    columns = ['person_id', 'person_weight', 'household_weight', 'weekend', 'canton_id']
    result = trips.merge(persons[columns], on='person_id', how='left', validate='many_to_one')
    result['mode'] = normalize_modes(result['mode'])
    valid = (~result.person_id.isin(excluded) & result.weekend.eq(False)
             & result.canton_id.between(1, 26) & result['mode'].isin(MODES)
             & np.isfinite(result.person_weight) & result.person_weight.gt(0)
             & result.household_weight.notna() & np.isfinite(result.crowfly_distance)
             & result.crowfly_distance.gt(0))
    # Origins and other trips by these people may be abroad; only destinations are selected.
    return result.loc[valid].copy()


def summarize(statistics):
    result = {}
    for identifier, cohorts in statistics.items():
        result[identifier] = {}
        for name, record in cohorts.items():
            total = float(record['weights'].sum())
            result[identifier][name] = {
                'persons': len(record['people']), 'trips': int(record['counts'].sum()),
                'counts': record['counts'].tolist(), 'weighted_counts': record['weights'].tolist(),
                'weight_sum': total,
                'shares': (record['weights'] / total).tolist() if total else [None] * len(MODES),
            }
    return result


def build_data(config_path, simulation_path):
    settings = yaml.safe_load(Path(config_path).read_text())
    config = settings['config']
    simulation = Path(simulation_path).resolve()
    output = simulation / 'mode_shares'
    provenance = json.loads((output / 'snapshot_mode_shares_sources.json').read_text())
    source = (Path(config['data_path']) / 'spatial/municipality/2023'
              / 'swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shp')
    municipalities = gpd.read_file(source, columns=['BFS_NUMMER', 'KANTONSNUM', 'NAME']).to_crs(2056)
    municipalities = municipalities.loc[municipalities.KANTONSNUM.isin([22, 25])].rename(
        columns={'BFS_NUMMER': 'municipality_id', 'NAME': 'name', 'KANTONSNUM': 'canton_id'})
    municipalities = municipalities[['municipality_id', 'name', 'canton_id', 'geometry']].copy()
    municipalities[['municipality_id', 'canton_id']] = municipalities[['municipality_id', 'canton_id']].astype(int)
    municipalities['canton'] = municipalities.canton_id.map({22: 'Vaud', 25: 'Geneva'})
    municipalities.geometry = shapely.force_2d(municipalities.geometry.array)
    municipalities = municipalities.sort_values('municipality_id').reset_index(drop=True)
    assert municipalities.municipality_id.is_unique
    statistics = empty_statistics(municipalities)
    cache = Path(settings['working_directory'])

    def cached(name):
        key = provenance['cached_survey_inputs'][name]
        with (cache / (key + '.p')).open('rb') as stream:
            return pickle.load(stream)

    trips, excluded, _, _ = cached('data.microcensus.trips')
    survey = prepare_survey(trips, cached('data.microcensus.persons'), excluded)
    survey = assign_municipalities(survey, 'destination_x', 'destination_y', municipalities)
    accumulate(statistics, survey, 'microcensus', 'person_id', 'person_weight')
    LOG.info('Assigned %s valid survey trips to %s municipalities', len(survey), len(municipalities))

    attributes = pd.read_csv(simulation / 'output_persons.csv.gz', sep=';', dtype={'person': str},
                             usecols=['person', 'subpopulation', 'cantonId'])
    assert attributes.person.is_unique
    columns = ['person', 'main_mode', 'end_x', 'end_y', 'euclidean_distance']
    for index, chunk in enumerate(pd.read_csv(provenance['trips_source'], sep=';', dtype={'person': str},
                                             usecols=columns, chunksize=250000), 1):
        chunk['mode'] = normalize_modes(chunk.main_mode)
        chunk = chunk.loc[np.isfinite(chunk.euclidean_distance) & chunk.euclidean_distance.gt(0)
                          & chunk['mode'].isin(MODES)]
        chunk = assign_municipalities(chunk, 'end_x', 'end_y', municipalities)
        chunk = chunk.merge(attributes, on='person', how='left', validate='many_to_one', indicator=True)
        assert chunk['_merge'].eq('both').all(), 'A trip person is missing from input attributes'
        chunk = chunk.loc[~chunk.subpopulation.eq('freight')]
        accumulate(statistics, chunk, 'simulation_all', 'person')
        accumulate(statistics, chunk.loc[chunk.cantonId.between(1, 26)], 'simulation_swiss', 'person')
        LOG.info('Processed simulation trip chunk %s', index)

    values = summarize(statistics)
    geojson = json.loads(municipalities.to_crs(4326).to_json(drop_id=True))
    for feature in geojson['features']:
        feature['properties']['statistics'] = values[feature['properties']['municipality_id']]
    payload = {'geojson': geojson, 'modes': MODES,
               'metadata': {'iteration': provenance['iteration'], 'municipal_boundary_year': 2023,
                            'boundary_source': str(source), 'trips_source': provenance['trips_source'],
                            'survey_sources': provenance['cached_survey_inputs'],
                            'simulation_sample_fraction': config['input_downsampling'],
                            'filters': 'Positive distance; five personal modes including normalized loop variants; freight excluded; origins unrestricted. Microcensus: valid weekday Swiss-resident trips, weighted by person_weight. Swiss simulation: home canton ID 1–26.',
                            'people_definition': 'Unique people with at least one retained trip ending in the municipality. These are sample counts, not population estimates.',
                            'boundary_ties': 'Shared-boundary destinations are assigned once to the lowest BFS municipality ID.'}}
    return payload, output


def write_outputs(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'municipality_arrivals_mode_shares.json').write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False))
    rows = []
    for feature in payload['geojson']['features']:
        properties = feature['properties']
        for cohort, record in properties['statistics'].items():
            for index, mode in enumerate(MODES):
                rows.append(dict(municipality_id=properties['municipality_id'], name=properties['name'],
                                 canton=properties['canton'], cohort=cohort, mode=mode,
                                 share=record['shares'][index], sample_trips=int(record['counts'][index]),
                                 weighted_trip_mass=record['weighted_counts'][index],
                                 cohort_sample_trips=record['trips'], cohort_sample_persons=record['persons']))
    pd.DataFrame(rows).to_csv(output / 'municipality_arrivals_mode_shares.csv', index=False)
    # Inline the map library so polygons, tables and bar charts work without tile/CDN access.
    assets = {}
    for name in ['leaflet.js', 'leaflet.css']:
        response = requests.get('https://unpkg.com/leaflet@1.9.4/dist/' + name, timeout=30)
        response.raise_for_status()
        assets[name] = response.text
    response = requests.get(TILES.format(z=13, x=4236, y=2907), timeout=20)
    response.raise_for_status()
    from io import BytesIO
    from PIL import Image
    Image.open(BytesIO(response.content)).verify()
    LOG.info('Verified a swisstopo tile (%s bytes)', len(response.content))
    template = Path(__file__).with_suffix('.html.template').read_text()
    text = (template.replace('__LEAFLET_CSS__', assets['leaflet.css'])
            .replace('__LEAFLET_JS__', assets['leaflet.js'].replace('</script>', '<\\/script>'))
            .replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/'))
            .replace('__TILES__', TILES))
    destination = output / 'municipality_arrivals_mode_shares.html'
    destination.write_text(text)
    LOG.info('Saved %s', destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_dib_10.yml')
    parser.add_argument('--simulation-path', required=True)
    arguments = parser.parse_args()
    payload, output = build_data(arguments.config, arguments.simulation_path)
    write_outputs(payload, output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
