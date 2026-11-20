"""Pooled airport-survey demographic preferences and unique resident selection."""

import numpy as np
import pandas as pd

AGE_BINS = (0, 18, 35, 55, 75, np.inf)
AGE_LABELS = ('0-17', '18-34', '35-54', '55-74', '75+')
ATTRIBUTES = ('person_id', 'household_id', 'age', 'sex', 'employed', 'resident_employment_code',
              'is_student', 'driving_license', 'car_availability', 'statpop_person_id')


def demographic_labels(frame):
    """Employment here is already harmonized to nullable true/false, not STATPOP codes."""
    return pd.DataFrame({
        'age_group': pd.cut(pd.to_numeric(frame.age, errors='coerce'), AGE_BINS,
                            labels=AGE_LABELS, right=False).astype('string'),
        'sex': pd.to_numeric(frame.sex, errors='coerce').map({0: 'male', 1: 'female'}).astype('string'),
        'employed': frame.employed.map({True: 'employed', False: 'not_employed'}).astype('string'),
    }, index=frame.index)


def demographic_preferences(survey, population, strength=20.):
    """Rake coarse population cells to smoothed pooled demographic marginals."""
    if isinstance(strength, bool) or not isinstance(strength, (int,float)) or not np.isfinite(strength) or strength <= 0:
        raise ValueError('airport_resident_demographic_prior_strength must be finite and positive')
    required = {'person_id', 'survey_year', 'age', 'sex', 'employed'}
    if not required.issubset(survey):
        raise ValueError(f'Airport survey missing demographic columns: {sorted(required-set(survey))}')
    # A respondent can contribute several Swiss diary segments; count them only once.
    donors = survey.drop_duplicates(['survey_year', 'person_id']).copy()
    source, target = demographic_labels(donors), demographic_labels(population)
    targets = {}
    report = {'method': 'Rake coarse population cells to smoothed pooled age/sex/employment marginals',
        'survey_respondents': len(donors), 'survey_segments': len(survey),
        'by_survey': {str(k): int(v) for k,v in donors.groupby('survey_year').size().items()},
        'weighting': 'One unweighted contribution per respondent, pooled across 2015/2021; not airport demand weights',
        'prior_strength_per_feature': float(strength), 'features': {},
        'limitations': ['National preferences shared by both airports and all municipalities/directions.',
            'Missing survey answers are ignored per feature, not converted to non-employment.',
            'No joint survey model is estimated; population-cell correlations provide the joint support.',
            'Fixed municipal demand, structural zeros and finite candidate pools can change realized demographics.']}
    for feature in target:
        counts = target[feature].value_counts().sort_index()
        observed = source[feature].value_counts().sort_index()
        if counts.empty:
            report['features'][feature] = {'population_known': 0, 'survey_known': int(observed.sum()), 'categories': {}}
            continue
        prior = counts / counts.sum()
        supported = observed.reindex(counts.index, fill_value=0)
        probability = (supported + strength * prior) / (supported.sum() + strength)
        ratios = probability / prior
        targets[feature] = probability
        report['features'][feature] = {'population_known': int(counts.sum()),
            'survey_known': int(observed.sum()), 'unsupported_survey_responses': int(observed.sum()-supported.sum()),
            'categories': {str(k): {'survey_count': int(supported[k]), 'population_count': int(counts[k]),
                'population_probability': float(prior[k]), 'target_probability': float(probability[k]),
                'marginal_ratio': float(ratios[k])} for k in counts.index}}
    weights, diagnostics = rake_preferences(target, targets)
    report.update(diagnostics)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid resident demographic selection weights')
    return weights, report


def rake_preferences(labels, targets):
    """Fit three national marginals on small age×sex×employment cells, not individuals."""
    if labels.empty:
        return np.ones(0), {'population_cells': 0, 'raking_iterations': 0, 'raking_max_marginal_error': 0.}
    encoded = labels.fillna('unknown')
    fields = list(encoded)
    cells = encoded.groupby(fields, dropna=False).size().rename('count').reset_index()
    prior = cells['count'].to_numpy(float) / len(labels)
    probability = prior.copy()
    desired = {}
    for feature, target in targets.items():
        unknown = float(prior[cells[feature].eq('unknown')].sum())
        desired[feature] = target * (1 - unknown)
        if unknown:
            desired[feature] = pd.concat([desired[feature], pd.Series({'unknown': unknown})])
    error, iteration = 0., 0
    for iteration in range(1, 101):
        for feature, marginal in desired.items():
            current = pd.Series(probability).groupby(cells[feature]).sum()
            probability *= cells[feature].map(marginal / current).to_numpy(float)
            probability /= probability.sum()
        error = max((float((pd.Series(probability).groupby(cells[f]).sum()-m).abs().max())
                     for f,m in desired.items()), default=0.)
        if error < 1e-8:
            break
    cells['weight'] = probability / prior
    fitted_cells = [{**{f: str(row[f]) for f in fields}, 'population_count': int(row['count']),
                     'probability': float(probability[i]), 'selection_weight': float(row['weight'])}
                    for i, row in cells.iterrows()]
    lookup = cells.set_index(fields).weight
    weights = lookup.reindex(pd.MultiIndex.from_frame(encoded)).to_numpy(float)
    return weights, {'population_cells': len(cells), 'raking_iterations': iteration,
                     'raking_max_marginal_error': error, 'raking_converged': error < 1e-8,
                     'raking_cells': fitted_cells}


def select_residents(frame, population, rng):
    """Choose at most one airport movement per synthetic resident across both airports."""
    result = frame.copy()
    for column in ATTRIBUTES:
        dtype = population[column].convert_dtypes().dtype if column in population else None
        if dtype is not None and pd.api.types.is_string_dtype(dtype):
            nullable = 'string'
        elif dtype is not None and pd.api.types.is_float_dtype(dtype):
            nullable = 'Float64'
        elif (dtype is not None and pd.api.types.is_bool_dtype(dtype)) or column in ('employed','is_student') and dtype is None:
            nullable = 'boolean'
        else:
            nullable = 'Int64'
        result[column] = pd.Series(pd.NA, index=result.index, dtype=nullable)
    result['resident_selection_weight'] = np.nan
    result['endpoint_x'] = np.nan
    result['endpoint_y'] = np.nan
    result['endpoint_ovgk'] = pd.Series('None', index=result.index, dtype='str')
    result['location_source'] = pd.Series('', index=result.index, dtype='str')
    result['unresolved_reason'] = pd.Series('', index=result.index, dtype='str')
    residents = result.loc[~result.is_visitor]
    pools = {int(k): g for k,g in population.groupby('municipality_id', sort=False)}
    shortages = []
    for municipality, rows in residents.groupby('municipality_id', sort=True):
        pool = pools.get(int(municipality), population.iloc[:0])
        n = min(len(rows), len(pool))
        # Randomize trip slots so scarce agents do not preferentially serve one airport/direction.
        slots = rng.permutation(rows.index.to_numpy())
        if n:
            probability = pool.weight.to_numpy(float)
            probability = probability / probability.sum()
            selected = pool.iloc[rng.choice(len(pool), n, replace=False, p=probability)]
            chosen = slots[:n]
            for column in ATTRIBUTES:
                if column in selected:
                    result.loc[chosen,column] = selected[column].to_numpy()
            result.loc[chosen,['endpoint_x','endpoint_y']] = selected[['x','y']].to_numpy()
            result.loc[chosen,'endpoint_ovgk'] = selected.ovgk.astype(str).to_numpy()
            result.loc[chosen,'resident_selection_weight'] = selected.weight.to_numpy()
            result.loc[chosen,'location_source'] = 'home'
        if n < len(rows):
            result.loc[slots[n:],'unresolved_reason'] = 'insufficient_synthetic_residents_in_municipality'
            shortages.append({'municipality_id': int(municipality), 'requested': len(rows),
                              'available': len(pool), 'unassigned': len(rows)-n})
    chosen = result.loc[result.person_id.notna()]
    if chosen.person_id.duplicated().any():
        raise ValueError('Airport resident selection reused a synthetic person')
    labels = demographic_labels(chosen) if len(chosen) else pd.DataFrame(columns=['age_group','sex','employed'])
    report = {'population_stage': 'synthesis.population.sampled', 'without_replacement': True,
        'requested_resident_trips': len(residents), 'assigned_resident_trips': len(chosen),
        'unassigned_resident_trips': len(residents)-len(chosen), 'municipal_shortages': shortages,
        'realized_demographics': {f: {str(k): int(v) for k,v in labels[f].value_counts().items()} for f in labels},
        'preference_model': population.attrs.get('demographic_model', {})}
    return result, report
