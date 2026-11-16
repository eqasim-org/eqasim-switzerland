"""Reusable, municipality-checked STATPOP homes and STATENT hotel locations."""

import numpy as np
import pandas as pd
import shapely


def configure(context):
    context.stage('data.statpop.persons')
    context.stage('data.statent.statent')
    context.stage('data.spatial.municipalities')


def execute(context):

    # Prepare full-population location pools independently of trip sampling settings.

    municipalities, mapping = context.stage('data.spatial.municipalities')
    homes, hotels = prepare_locations(context.stage('data.statpop.persons'),
                                     context.stage('data.statent.statent'), municipalities, mapping)

    # Return person-weighted homes and uniformly weighted hotel establishments.

    return homes, hotels


def prepare_locations(persons, statent, municipalities, mapping):
    """Use full permanent residents (person weighted), and equally weighted hotels."""
    lookup = mapping[['deprecated_municipality_id', 'municipality_id']].drop_duplicates()
    if lookup.deprecated_municipality_id.duplicated().any():
        raise ValueError('Ambiguous home municipality ID mapping')
    lookup = lookup.set_index('deprecated_municipality_id').municipality_id
    selected = persons.population_type.eq(1) & persons.type_of_residence.eq(1)
    permanent_residents = int(selected.sum())
    if 'statpop_geocoding_indicator' in persons:
        selected &= persons.statpop_geocoding_indicator.eq(1)
    if 'collective_housing_resident' in persons:
        selected &= ~persons.collective_housing_resident.fillna(True)
    homes = persons.loc[selected, ['municipality_id', 'home_x', 'home_y']].rename(columns={'home_x': 'x', 'home_y': 'y'}).copy()
    homes['municipality_id'] = homes.municipality_id.map(lookup)
    missing_mapping = int(homes.municipality_id.isna().sum())
    homes = homes.groupby(['municipality_id', 'x', 'y']).size().rename('weight').reset_index()
    codes = statent.noga.astype(str).str.replace(r'\.0$', '', regex=True)
    hotels = statent.loc[codes.str.startswith('551'), ['enterprise_id', 'municipality_id', 'x', 'y']].drop_duplicates().copy()
    if hotels.enterprise_id.duplicated().any():
        raise ValueError('Hotel establishment assigned to multiple locations')
    hotels['weight'] = 1.
    shapes = municipalities.loc[municipalities.municipality_id.gt(0)].to_crs(2056).dissolve(by='municipality_id').geometry
    output = []
    for frame in (homes, hotels):
        before = len(frame)
        valid = (np.isfinite(frame[['x', 'y']]).all(axis=1) & frame.x.gt(0) & frame.y.gt(0) & frame.municipality_id.isin(shapes.index))
        frame = frame.loc[valid].copy()
        # Check coordinates against their municipality instead of trusting a reporting code.
        geometry = frame.municipality_id.map(shapes).to_numpy()
        inside = shapely.covers(geometry, shapely.points(frame.x.to_numpy(), frame.y.to_numpy()))
        frame = frame.loc[inside].copy()
        frame['municipality_id'] = frame.municipality_id.astype(int)
        order = ['municipality_id', 'x', 'y'] + (['enterprise_id'] if 'enterprise_id' in frame else [])
        frame = frame.sort_values(order).reset_index(drop=True)
        frame.attrs['excluded_invalid_or_outside_municipality_locations'] = before - len(frame)
        if 'enterprise_id' not in frame:
            frame.attrs.update(permanent_main_residents=permanent_residents,
                               eligible_geocoded_residents=int(selected.sum()),
                               residents_missing_municipality_mapping=missing_mapping,
                               sampleable_residents=int(frame.weight.sum()))
        output.append(frame)
    return tuple(output)
