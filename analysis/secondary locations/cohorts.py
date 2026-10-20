"""Comparable domestic secondary-trip cohorts and accessibility attributes."""
import gc
import numpy as np
import pandas as pd
import shapely

PURPOSES = ['shop', 'leisure', 'other', 'work_secondary', 'home_secondary', 'education_secondary']
PT_CLASSES = ['A', 'B', 'C', 'D', 'None']
LABELS = {'survey': 'Microcensus', 'rda': 'RDA', 'nn_distance': 'NN · distance input',
          'nn_no_distance': 'NN · learned selector'}


def accessibility(x, y, polygons):
    labels = np.full(len(x), 'None', dtype=object)
    for level in reversed(PT_CLASSES[:-1]):
        selected = polygons.loc[polygons.ovgk == level, 'geometry']
        if selected.empty:
            continue
        geometry = selected.union_all()
        shapely.prepare(geometry)
        labels[shapely.contains_xy(geometry, np.asarray(x), np.asarray(y))] = level
    return labels


def xy(frame):
    frame = frame.to_crs(2056)
    result = frame.drop(columns='geometry').copy()
    result['x'], result['y'] = frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()
    return pd.DataFrame(result)


def domestic(frame, border):
    geom = border.to_crs(2056).union_all()
    shapely.prepare(geom)
    return (shapely.contains_xy(geom, frame.origin_x.to_numpy(), frame.origin_y.to_numpy())
            & shapely.contains_xy(geom, frame.destination_x.to_numpy(), frame.destination_y.to_numpy()))


def decorate(frame, source):
    frame = frame.copy()
    frame['dataset'] = source
    for column in ['destination_id', 'mz_person_id']:
        if column in frame:
            frame[column] = frame[column].astype('string')
    frame['distance_km'] = np.hypot(frame.destination_x - frame.origin_x,
                                     frame.destination_y - frame.origin_y) / 1000
    frame['home_distance_km'] = np.hypot(frame.destination_x - frame.home_x,
                                          frame.destination_y - frame.home_y) / 1000
    frame['car_group'] = np.where(frame.car_available, 'Car access', 'No car access')
    frame['home_pt'] = frame.home_pt.fillna('None').astype(str)
    frame['home_pt_tier'] = frame.home_pt.map({'A': 'High (A/B)', 'B': 'High (A/B)',
                                               'C': 'Medium (C)', 'D': 'Low (D/none)', 'None': 'Low (D/none)'})
    frame['age_group'] = pd.cut(frame.age, [5, 17, 34, 54, 74, np.inf],
                                labels=['6–17', '18–34', '35–54', '55–74', '75+']).astype(str)
    frame['weight'] = pd.to_numeric(frame.weight)
    if not (np.isfinite(frame.weight) & (frame.weight > 0)).all():
        raise ValueError('Weights must be positive and finite')
    for column in ['dataset', 'purpose', 'car_group', 'home_pt', 'destination_pt',
                   'home_pt_tier', 'age_group', 'municipality_type', 'origin_purpose']:
        if column in frame:
            frame[column] = frame[column].astype('category')
    return frame


def population(cache):
    audit = {}
    columns = ['person_id', 'trip_id', 'trip_index', 'mz_person_id', 'preceding_purpose',
               'following_purpose', 'departure_time', 'mode']
    trips = cache.stage('synthesis.population.trips')[columns].copy()
    audit['all_trips'] = len(trips)
    border_ids = trips.loc[(trips.preceding_purpose == 'border') | (trips.following_purpose == 'border'), 'person_id'].unique()
    trips = trips.loc[trips.following_purpose.isin(PURPOSES) & ~trips.person_id.isin(border_ids)].copy()
    audit['secondary_trips_excluding_border_persons'] = len(trips)
    secondary = cache.stage('synthesis.population.spatial.secondary.locations')[0][['person_id', 'trip_index']].copy()
    # Secondary output trip_index denotes the following activity index.
    secondary = secondary.rename(columns={'trip_index': 'activity_index'})
    trips['activity_index'] = trips.trip_index + 1
    trips = trips.merge(secondary, on=['person_id', 'activity_index'], validate='one_to_one')
    audit['with_secondary_assignment'] = len(trips)
    locations = xy(cache.stage('synthesis.population.spatial.locations'))[
        ['person_id', 'activity_index', 'destination_id', 'x', 'y', 'ovgk']]
    origins = locations.rename(columns={'activity_index': 'trip_index', 'x': 'origin_x', 'y': 'origin_y'})
    trips = trips.merge(origins[['person_id', 'trip_index', 'origin_x', 'origin_y']],
                        on=['person_id', 'trip_index'], validate='many_to_one')
    destinations = locations.rename(columns={'x': 'destination_x', 'y': 'destination_y', 'ovgk': 'destination_pt'})
    trips = trips.merge(destinations, on=['person_id', 'activity_index'], validate='many_to_one')
    # First activity may be away from home. Use actual household home geometry.
    persons = cache.stage('synthesis.population.enriched')
    keep = ['person_id', 'household_id', 'age', 'sex', 'car_availability', 'driving_license', 'municipality_type']
    persons = persons[keep].copy()
    homes_hh = xy(cache.stage('synthesis.population.spatial.home.locations'))[['household_id', 'x', 'y']]
    persons = persons.merge(homes_hh, on='household_id', validate='many_to_one').rename(columns={'x': 'home_x', 'y': 'home_y'})
    polygons = cache.stage('data.spatial.ovgk')
    persons['home_pt'] = accessibility(persons.home_x, persons.home_y, polygons)
    persons['car_available'] = persons.pop('car_availability').eq(1)
    trips = trips.merge(persons, on='person_id', validate='many_to_one')
    trips = trips.rename(columns={'following_purpose': 'purpose', 'preceding_purpose': 'origin_purpose'})
    finite = np.isfinite(trips[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'home_x', 'home_y']]).all(axis=1)
    audit['invalid_coordinates'] = int((~finite).sum())
    trips = trips.loc[finite & (trips.age >= 6)].copy()
    inside = domestic(trips, cache.stage('data.spatial.swiss_border'))
    audit['excluded_non_domestic'] = int((~inside).sum())
    trips = trips.loc[inside].copy()
    audit['analysis_trips'] = len(trips)
    audit['analysis_persons'] = int(trips.person_id.nunique())
    trips['weight'] = 1.0
    del persons, locations, origins, destinations
    gc.collect()
    return decorate(trips, cache.kind), audit


def survey(cache):
    persons = cache.stage('data.microcensus.persons')
    years = sorted(int(y) for y in pd.to_datetime(persons['date'], errors='coerce').dt.year.dropna().unique()) if 'date' in persons else []
    persons = persons.loc[~persons.weekend, ['person_id', 'person_weight', 'age', 'sex', 'car_availability',
        'driving_license', 'municipality_type', 'home_x', 'home_y', 'work_x', 'work_y']].copy()
    persons['car_available'] = persons.pop('car_availability').ne(2)  # 0=always, 1=sometimes, 2=never
    persons = persons.rename(columns={'person_weight': 'weight'})
    trips = cache.stage('data.microcensus.trips')[0].copy()
    frame = trips.loc[trips.purpose.isin(PURPOSES)].merge(persons, on='person_id', validate='many_to_one')
    audit = {'weekday_secondary_trips_age_unrestricted': len(frame), 'survey_years': years}
    valid = np.isfinite(frame[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'home_x', 'home_y']]).all(axis=1)
    frame = frame.loc[valid & (frame.age >= 6)].copy()
    frame = frame.loc[domestic(frame, cache.stage('data.spatial.swiss_border'))].copy()
    polygons = cache.stage('data.spatial.ovgk')
    frame['home_pt'] = accessibility(frame.home_x, frame.home_y, polygons)
    frame['destination_pt'] = accessibility(frame.destination_x, frame.destination_y, polygons)
    audit.update(analysis_trips=len(frame), analysis_persons=int(frame.person_id.nunique()))
    return decorate(frame, 'survey'), audit
