"""Reusable, municipality-checked homes, hotels and dormitory locations."""

import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
from pathlib import Path
from data.airports._shared import accommodation
from data.airports._shared.common import file_digest
from data.airports._shared.accommodation import select_accommodation
from data.utils import coerce_boolean_series
from synthesis.airports import residents


def configure(context):
    context.stage('synthesis.population.enriched')
    context.stage('synthesis.population.activities')
    context.stage('data.microcensus.airports')
    context.stage('data.statent.statent')
    context.stage('data.spatial.municipalities')
    context.stage('data.spatial.ovgk')
    context.config('airport_resident_demographic_prior_strength', default=20.)
    context.config('airport_resident_max_nonhome_activities', default=1)


def execute(context):

    # Read post-matching residents and classify their actual assigned activity plans.

    municipalities, mapping = context.stage('data.spatial.municipalities')
    population = residents.activity_preferences(context.stage('synthesis.population.enriched'),
        context.stage('synthesis.population.activities'), context.config('airport_resident_max_nonhome_activities'))
    homes, hotels = prepare_locations(population,
        context.stage('data.statent.statent'), municipalities, mapping)

    # Check accessibility at each person's actual home and each accommodation point.

    areas = context.stage('data.spatial.ovgk')
    homes, hotels = (attach_ovgk(pool, areas) for pool in (homes, hotels))

    # Fit pooled demographic preferences once; no regional demographic models or exports.

    homes['weight'], report = residents.demographic_preferences(context.stage('data.microcensus.airports'),
        homes, context.config('airport_resident_demographic_prior_strength'))
    homes.attrs['demographic_model'] = report

    # Return individual residents and hotel/dormitory establishments, without changing parents.

    return homes, hotels


def validate(context):
    return file_digest(Path(accommodation.__file__)) + file_digest(Path(residents.__file__))


def prepare_locations(population, statent, municipalities, mapping):
    """Keep synthetic identities, valid Swiss homes and uniformly weighted accommodation."""
    required = {'person_id', 'age', 'sex', 'employed', 'home_x', 'home_y', 'home_municipality_id'}
    if not required.issubset(population):
        raise ValueError(f'Post-matching population lacks airport attributes: {sorted(required-set(population))}')
    if population.person_id.isna().any() or population.person_id.duplicated().any():
        raise ValueError('Synthetic resident person IDs must be complete and unique')
    selected = pd.Series(True, index=population.index)
    excluded = {}
    for flag in ('non_private_home_only_households', 'is_crossing_the_border',
                 'is_truck_driver', 'is_outside_of_switzerland'):
        if flag in population:
            mask = coerce_boolean_series(population[flag], name=flag)
            excluded[flag] = int(mask.sum())
            selected &= ~mask
    columns = [c for c in residents.ATTRIBUTES if c in population and c != 'resident_employment_code']
    pool_columns = [c for c in ('airport_preferred', 'has_home_activity') if c in population]
    homes = population.loc[selected, columns + pool_columns + ['home_x', 'home_y', 'home_municipality_id']].copy()
    homes = homes.rename(columns={'home_x':'x', 'home_y':'y', 'home_municipality_id':'municipality_id'})
    homes['resident_employment_code'] = homes.employed
    # Synthetic employment uses 1=employed, 2=unemployed, 3=inactive; never cast these to bool.
    homes['employed'] = pd.to_numeric(homes.employed, errors='coerce').map({0:False, 1:True, 2:False, 3:False}).astype('boolean')
    lookup = mapping[['deprecated_municipality_id','municipality_id']].drop_duplicates()
    if lookup.deprecated_municipality_id.duplicated().any():
        raise ValueError('Ambiguous home municipality ID mapping')
    ids = pd.to_numeric(homes.municipality_id, errors='coerce')
    remapped = ids.map(lookup.set_index('deprecated_municipality_id').municipality_id)
    current = set(municipalities.municipality_id)
    homes['municipality_id'] = remapped.fillna(ids.where(ids.isin(current)))
    homes['weight'] = 1.
    hotels = select_accommodation(statent, ['enterprise_id','municipality_id','x','y'])
    hotels['weight'] = 1.
    shapes = municipalities.loc[municipalities.municipality_id.gt(0)].to_crs(2056).dissolve(by='municipality_id').geometry
    shapely.prepare(shapes.to_numpy())
    output = []
    for frame in (homes, hotels):
        before = len(frame)
        valid = (np.isfinite(frame[['x','y']]).all(axis=1) & frame.x.gt(0) & frame.y.gt(0) & frame.municipality_id.isin(shapes.index))
        frame = frame.loc[valid].copy()
        if 'person_id' in frame:
            frame = frame.loc[pd.to_numeric(frame.age, errors='coerce').ge(0)]
        geometry = frame.municipality_id.map(shapes).to_numpy()
        frame = frame.loc[shapely.covers(geometry, shapely.points(frame.x.to_numpy(),frame.y.to_numpy()))].copy()
        frame['municipality_id'] = frame.municipality_id.astype(int)
        frame = frame.sort_values('person_id' if 'person_id' in frame else 'enterprise_id',kind='stable').reset_index(drop=True)
        frame.attrs['excluded_invalid_or_outside_municipality_locations'] = before-len(frame)
        if 'person_id' in frame:
            preferred = frame.get('airport_preferred', pd.Series(True, index=frame.index))
            frame.attrs.update(population_stage='synthesis.population.enriched', source_persons=len(population),
                               activity_stage='synthesis.population.activities', eligible_residents=len(frame),
                               preferred_residents=int(preferred.sum()), reserve_residents=int((~preferred).sum()),
                               max_nonhome_activities=population.attrs.get('airport_max_nonhome_activities'),
                               excluded_flags=excluded)
        output.append(frame)
    return tuple(output)


def attach_ovgk(pool, areas):
    """Use the best covering class for overlaps; uncovered points remain None."""
    result = pool.copy().reset_index(drop=True)
    ranks = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    polygons = areas[['ovgk', 'geometry']].to_crs(2056).copy()
    polygons['rank'] = polygons.ovgk.map(ranks).astype(float)
    polygons = polygons.loc[polygons['rank'].notna(), ['rank', 'geometry']]
    # Index local pieces, rather than the bounding boxes of nationwide multipolygons.
    polygons = polygons.explode(index_parts=False, ignore_index=True)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(result.x, result.y), crs=2056)
    joined = gpd.sjoin(points, polygons, how='left', predicate='intersects')
    best = joined.groupby(level=0)['rank'].min()
    result['ovgk'] = best.reindex(result.index).map({rank: label for label, rank in ranks.items()}).fillna('None')
    result.attrs = dict(pool.attrs)
    result.attrs['ovgk_counts'] = {str(k): int(v) for k, v in result.ovgk.value_counts().items()}
    return result
