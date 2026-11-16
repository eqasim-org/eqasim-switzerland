"""Full permanent main-residence population by current municipality."""

import numpy as np
import pandas as pd


def configure(context):
    context.stage('data.statpop.persons')
    context.stage('data.spatial.municipalities')


def execute(context):

    # Count source residents without household filtering or synthetic downsampling.

    persons = context.stage('data.statpop.persons')
    _, mapping = context.stage('data.spatial.municipalities')
    return aggregate_population(persons, mapping)


def aggregate_population(persons, mapping):
    selected = persons.loc[persons.population_type.eq(1) & persons.type_of_residence.eq(1),
                           ['person_id', 'municipality_id', 'home_x', 'home_y']].copy()
    if selected.person_id.duplicated().any():
        raise ValueError('STATPOP has duplicate permanent main-residence person IDs')
    source_count = len(selected)
    lookup = mapping[['deprecated_municipality_id', 'municipality_id']].drop_duplicates()
    if lookup.deprecated_municipality_id.duplicated().any():
        raise ValueError('Ambiguous municipality ID remapping')
    selected = selected.rename(columns={'municipality_id': 'deprecated_municipality_id'}).merge(
        lookup, how='left', on='deprecated_municipality_id', validate='many_to_one')
    missing = selected.municipality_id.isna()
    if missing.any():
        raise ValueError(f'{int(missing.sum())} STATPOP residents lack a current municipality mapping')
    selected['municipality_id'] = selected.municipality_id.astype(int)
    valid = np.isfinite(selected[['home_x', 'home_y']]).all(axis=1) & selected.home_x.gt(0) & selected.home_y.gt(0)
    centres = selected.loc[valid].groupby('municipality_id')[['home_x', 'home_y']].mean()
    counts = selected.groupby('municipality_id').size().rename('population').to_frame().join(centres).reset_index()
    counts.attrs['source_persons'] = source_count
    counts.attrs['definition'] = 'STATPOP permanent population, main residence; no synthetic household or sampling filters'
    return counts
