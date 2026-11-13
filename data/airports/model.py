"""Calibrated allocation of fixed airport passenger totals to Swiss municipalities."""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import uuid

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import least_squares
from scipy.special import logsumexp

from data.airports.common import apportion, atomic_write, export_signature, file_digest

AIRPORTS = {'GVA': {'longitude': 6.109444, 'latitude': 46.238333},
            'ZRH': {'longitude': 8.548056, 'latitude': 47.458056}}
# Approximate airport reference points, configurable and sufficient for municipal gravity.
CANTON_GROUPS = {'CH-GE': 25, 'CH-VD': 22, 'CH-VS': 23}
MODEL_VERSION = 1
PARAMETER_NAMES = ('distance_offset_km', 'population_distance_decay',
                   'hotel_distance_decay', 'visitor_hotel_mixture')


def configure(context):
    context.config('data_path')
    context.stage('data.statpop.municipality_counts')
    context.stage('data.statent.statent')
    context.stage('data.spatial.municipalities')
    context.stage('data.spatial.cantons')
    context.config('airport_model_folder', default='airports/models')
    context.config('airport_model_overwrite', default=False, volatile=True)
    context.config('airport_model_historical_csv', default='airports/geneva/historical_swiss_2013.csv')
    context.config('airport_model_airports', default=AIRPORTS)
    context.config('airport_model_distance_offset_km', default=50.)
    context.config('geneva_airport_pdf', default='airports/geneva/geneva_airport.pdf')
    context.config('geneva_airport_survey_csv', default=None)


def execute(context):

    # Reuse full census counts and already located STATENT establishments.

    municipalities, _ = context.stage('data.spatial.municipalities')
    features = build_features(municipalities, context.stage('data.spatial.cantons'),
                              context.stage('data.statpop.municipality_counts'),
                              context.stage('data.statent.statent'))
    options, output = runtime_inputs(context)
    output.mkdir(parents=True, exist_ok=True)

    # Estimate only on explicit request; otherwise reuse the saved coefficients.

    if overwrite_parameters(context):
        sources, fit_options, _ = model_inputs(context)
        historical, targets, visitor_share, provenance = read_calibration(sources)
        parameters, diagnostics = calibrate(features, historical, targets, fit_options,
                                            swiss_visitor_share=visitor_share)
        diagnostics['provenance'] = provenance
        diagnostics['input_summary'] = dict(features.attrs)
        report = {'schema_version': 1, 'signature': model_signature(features, sources, fit_options),
                  'parameters': parameters, 'swiss_visitor_share_transferred_to_zurich': visitor_share,
                  'options': fit_options, 'diagnostics': diagnostics}
        validate_parameters(report)
        text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        atomic_write(output / 'calibration.json', lambda p: p.write_text(text, encoding='utf-8'))
        atomic_write(output / 'parameters.json', lambda p: p.write_text(text, encoding='utf-8'))
    else:
        report = read_parameters(output)

    # Apply fixed parameters to the current municipality inputs.

    parameters = report['parameters']
    visitor_share = report['swiss_visitor_share_transferred_to_zurich']
    diagnostics = deepcopy(report.get('diagnostics', {}))
    diagnostics['current_input_summary'] = dict(features.attrs)
    token = model_signature(features, {'parameters': output / 'parameters.json'}, options)
    result = AirportModel(features, parameters, visitor_share, options['airports'],
                          parameters['distance_offset_km'], token, diagnostics)

    # Export the current model and municipal inputs without rewriting the calibration.

    atomic_write(output / 'municipalities.parquet', lambda p: features.to_parquet(p, index=False))
    atomic_write(output / 'model.pkl', lambda p: p.write_bytes(pickle.dumps(result, protocol=4)))
    manifest = {'signature': token, 'sha256': file_digest(output / 'model.pkl')}
    atomic_write(output / 'signature.json', lambda p: p.write_text(json.dumps(manifest)+'\n', encoding='utf-8'))
    return result


def validate(context):
    if overwrite_parameters(context):
        sources, options, _ = model_inputs(context)
        # Explicit overwrite must also bypass Synpp's stage cache on every run.
        return model_signature(None, sources, options) + str(uuid.uuid4())
    options, output = runtime_inputs(context)
    read_parameters(output)
    # Parent changes update allocations, but never trigger coefficient estimation.
    exports = {name: output / name for name in ('model.pkl', 'municipalities.parquet', 'signature.json')}
    return model_signature(None, {'parameters': output / 'parameters.json'}, options) + export_signature(exports)


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def overwrite_parameters(context):
    value = context.config('airport_model_overwrite')
    if not isinstance(value, bool):
        raise ValueError('airport_model_overwrite must be a YAML boolean (true or false)')
    return value


def runtime_inputs(context):
    root = Path(context.config('data_path'))
    airports = context.config('airport_model_airports')
    for airport in ('GVA', 'ZRH'):
        longitude, latitude = airports[airport]['longitude'], airports[airport]['latitude']
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError('Invalid airport longitude/latitude')
    return {'airports': airports, 'version': MODEL_VERSION}, resolve(root, context.config('airport_model_folder'))


def model_inputs(context):
    options, output = runtime_inputs(context)
    root = Path(context.config('data_path'))
    old = resolve(root, context.config('airport_model_historical_csv'))
    old_meta = old.with_suffix('.json')
    metadata = json.loads(old_meta.read_text(encoding='utf-8'))
    survey = context.config('geneva_airport_survey_csv')
    survey = resolve(root, survey) if survey else root / 'airports/geneva/geneva_survey.csv'
    sources = {'historical': old, 'historical_metadata': old_meta,
               'historical_pdf': resolve(old.parent, metadata['pdf_file']),
               'survey': survey, 'survey_metadata': survey.with_suffix('.json'),
               'survey_pdf': resolve(root, context.config('geneva_airport_pdf'))}
    offset = float(context.config('airport_model_distance_offset_km'))
    if not np.isfinite(offset) or offset <= 0:
        raise ValueError('airport_model_distance_offset_km must be positive')
    options['distance_offset_km'] = offset
    return sources, options, output


def model_signature(features, sources, options):
    digest = hashlib.sha256(json.dumps(options, sort_keys=True, allow_nan=False).encode())
    for path in [*sources.values(), Path(__file__), Path(__file__).with_name("geneva.py"), Path(__file__).with_name("common.py")]:
        digest.update(file_digest(path).encode())
    if features is not None:
        digest.update(pd.util.hash_pandas_object(features, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def read_parameters(folder):
    path = folder / 'parameters.json'
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        validate_parameters(report)
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError(f'Cannot load airport model parameters from {path}: {error}. '
                         'Restore a valid parameter file or set airport_model_overwrite: true '
                         'to estimate and save parameters explicitly.') from error
    return report


def validate_parameters(report):
    if report['schema_version'] != 1:
        raise ValueError('Unsupported airport model parameter schema')
    values = [report['parameters'][name] for name in PARAMETER_NAMES]
    visitor = report['swiss_visitor_share_transferred_to_zurich']
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in [*values, visitor]) or
            not np.isfinite([*values, visitor]).all() or
            any(value <= 0 for value in values[:3]) or not 0 <= values[3] <= 1 or
            not 0 <= visitor <= 1):
        raise ValueError('Invalid airport model coefficients or Swiss visitor share')


def build_features(municipalities, cantons, population, statent):
    frame = municipalities.loc[municipalities.municipality_id.gt(0)].to_crs(2056).copy()
    frame = frame.dissolve(by=['municipality_id', 'municipality_name'], as_index=False)
    frame = frame.merge(population, on='municipality_id', how='left', validate='one_to_one')
    frame['population'] = frame.population.fillna(0).astype(int)
    codes = statent.noga.astype(str).str.replace(r'\.0$', '', regex=True)
    hotels = statent.loc[codes.str.startswith('551'), ['enterprise_id', 'municipality_id']].drop_duplicates()
    if hotels.enterprise_id.duplicated().any():
        raise ValueError('Hotel establishment assigned to multiple municipalities')
    counts = hotels.groupby('municipality_id').size().rename('hotels')
    frame = frame.merge(counts, on='municipality_id', how='left', validate='one_to_one')
    frame['hotels'] = frame.hotels.fillna(0).astype(int)
    # Water/uninhabited polygons without residents or hotels cannot generate access trips.
    inactive = int((frame.population.eq(0) & frame.hotels.eq(0)).sum())
    frame = frame.loc[frame.population.gt(0) | frame.hotels.gt(0)].reset_index(drop=True)
    # Choose an interior municipal point, improving it with the census mean where usable.
    points = frame.geometry.representative_point()
    valid = np.isfinite(frame[['home_x', 'home_y']]).all(axis=1)
    home = gpd.GeoSeries(gpd.points_from_xy(frame.home_x, frame.home_y), index=frame.index, crs=2056)
    use_home = valid & frame.geometry.covers(home)
    points.loc[use_home] = home.loc[use_home]
    frame['x'], frame['y'] = points.x, points.y
    areas = cantons.loc[cantons.canton_id.between(1, 26), ['canton_id', 'geometry']].to_crs(2056)
    point_frame = gpd.GeoDataFrame(frame[['municipality_id']], geometry=points, crs=2056)
    joined = gpd.sjoin(point_frame, areas, predicate='intersects', how='left')
    if joined.municipality_id.duplicated().any() or joined.canton_id.isna().any():
        raise ValueError('Swiss municipal representative point lacks an unambiguous canton: ' + str(joined.loc[joined.canton_id.isna() | joined.municipality_id.duplicated(), 'municipality_id'].tolist()))
    frame = frame.merge(joined[['municipality_id', 'canton_id']], on='municipality_id', validate='one_to_one')
    result = pd.DataFrame(frame[['municipality_id', 'municipality_name', 'canton_id', 'population', 'hotels', 'x', 'y']])
    result['municipality_id'] = result.municipality_id.astype(int)
    result['canton_id'] = result.canton_id.astype(int)
    result = result.sort_values('municipality_id').reset_index(drop=True)
    if result.population.sum() != population.population.sum():
        raise ValueError('Population mass lost while joining Swiss municipalities')
    result.attrs = {'population': int(result.population.sum()), 'hotels': int(result.hotels.sum()),
                    'municipalities': len(result), 'hotel_definition': 'Distinct STATENT NOGA 551 establishments; all 559 dormitories excluded',
                    'population_definition': population.attrs.get('definition', ''),
                    'population_centres_used': int(use_home.sum()), 'inactive_geometries_excluded': inactive,
                    'statent_years': sorted(pd.to_numeric(statent.statent_year, errors='coerce').dropna().astype(int).unique().tolist()) if 'statent_year' in statent else []}
    if result.hotels.sum() <= 0 or result.population.sum() <= 0:
        raise ValueError('Airport gravity requires positive population and hotel counts')
    return result


def read_calibration(sources):
    from data.airports.geneva import process_pdf, build_ods
    old_meta = json.loads(sources['historical_metadata'].read_text())
    if file_digest(sources['historical_pdf']) != old_meta['pdf_sha256']:
        raise ValueError('Historical Geneva PDF changed: review calibration CSV and metadata')
    old = pd.read_csv(sources['historical'])
    if (len(old) < 5 or old.canton_id.duplicated().any() or
            not np.isfinite(old.passengers_thousands).all() or (old.passengers_thousands <= 0).any()):
        raise ValueError('Historical calibration needs at least five distinct positive canton observations')
    observations = process_pdf(sources['survey_pdf'], sources['survey'])
    targets = {}
    for profile in ('outgoing', 'incoming_other'):
        rows = observations.loc[observations.table.eq(profile)].set_index('region_id')
        values = rows.loc[['CH-GE', 'CH-VD', 'CH-VS', 'CH-OTHER'], 'percent'].to_numpy(float)
        targets[profile] = values / values.sum()
    ods = build_ods(observations)
    swiss = ods.loc[ods.country.eq('CH')].groupby('profile').joint_probability.sum()
    visitor_share = float(swiss['incoming_other'] / swiss.sum())
    return old, targets, visitor_share, {'historical': old_meta,
        'latest_survey_year': observations.attrs['survey_year'],
        'limitations': ['Historical 2013 flows fitted with contemporary population and hotel covariates; current Swiss visitor fraction is assumed for the historical pooled fit.',
                       'Only six unambiguous historic Swiss cantons are used; other cantons are not treated as zero.',
                       'Euclidean access and hotel counts are proxies; there are no measured municipal ODs.',
                       'Zurich uses Geneva parameters and Swiss resident/visitor mixture without local calibration.']}


def distances(features, airport):
    x, y = Transformer.from_crs(4326, 2056, always_xy=True).transform(airport['longitude'], airport['latitude'])
    return np.hypot(features.x.to_numpy() - x, features.y.to_numpy() - y) / 1000.


def kernels(features, distance, parameters, offset):
    fitted_offset, beta_population, beta_hotel, _ = parameters
    log_distance = np.log(distance + fitted_offset)
    def component(mass, exponent, beta):
        positive = mass > 0
        if not positive.any():
            raise ValueError('No positive municipal mass for gravity component')
        log_weight = np.full(len(mass), -np.inf)
        log_weight[positive] = exponent * np.log(mass[positive]) - beta * log_distance[positive]
        return np.exp(log_weight - logsumexp(log_weight))
    return (component(features.population.to_numpy(float), 1., beta_population),
            component(features.hotels.to_numpy(float), 1., beta_hotel))


def grouped(values, labels, groups):
    result = np.array([values[labels == key].sum() for key in groups])
    if (result <= 0).any():
        raise ValueError('Calibration region has no model support')
    return result / result.sum()


def calibrate(features, historical, targets, options, swiss_visitor_share=.4):
    distance = distances(features, options['airports']['GVA'])
    cantons = features.canton_id.to_numpy()
    observed_old = historical.passengers_thousands.to_numpy(dtype=float, copy=True)
    observed_old /= observed_old.sum()
    def old_residual(parameters):
        population, hotels = kernels(features, distance, parameters, options['distance_offset_km'])
        mixture = swiss_visitor_share * parameters[3]
        weights = (1 - mixture) * population + mixture * hotels
        prediction = grouped(weights, cantons, historical.canton_id.to_numpy())
        # Weak anchors stabilise a four-parameter fit to six rounded regional flows.
        prior = (np.asarray(parameters) - np.array([options['distance_offset_km'], 2., 2., .4])) / np.array([200., 4., 4., .8])
        return np.r_[np.log(prediction) - np.log(observed_old), .04 * prior]
    bounds = ([1., .05, .05, .001], [500., 12., 12., .999])
    fits = [least_squares(old_residual, [offset, b, b, mix], bounds=bounds, max_nfev=1500)
            for offset in (20., 100., 300.) for b in (2., 5.) for mix in (.2, .7)]
    old_fit = min(fits, key=lambda f: np.sum(f.fun ** 2))
    old_parameters = old_fit.x
    labels = np.where(cantons == 25, 0, np.where(cantons == 22, 1, np.where(cantons == 23, 2, 3)))
    def latest_residual(free):
        parameters = np.r_[old_parameters[0], free]
        population, hotels = kernels(features, distance, parameters, options['distance_offset_km'])
        visitor = (1 - parameters[3]) * population + parameters[3] * hotels
        fitted = [grouped(population, labels, range(4)), grouped(visitor, labels, range(4))]
        error = np.concatenate([np.log(p) - np.log(targets[k]) for p, k in zip(fitted, ('outgoing', 'incoming_other'))])
        prior = (free - old_parameters[1:]) / np.array([3., 3., .8])
        return np.r_[error, .04 * prior]
    latest_fits = [least_squares(latest_residual, [old_parameters[1], old_parameters[2], mix],
                                bounds=(bounds[0][1:], bounds[1][1:]), max_nfev=1500) for mix in (.2, .6, .9)]
    latest = min(latest_fits, key=lambda f: np.sum(f.fun ** 2))
    if not old_fit.success or not latest.success:
        raise ValueError('Airport model calibration did not converge')
    parameters = np.r_[old_parameters[0], latest.x]
    population, hotels = kernels(features, distance, parameters, options['distance_offset_km'])
    visitor = (1-parameters[3])*population + parameters[3]*hotels
    historical_population, historical_hotels = kernels(features, distance, old_parameters, options['distance_offset_km'])
    historical_weights = ((1-swiss_visitor_share*old_parameters[3])*historical_population
                          + swiss_visitor_share*old_parameters[3]*historical_hotels)
    names = ['distance_offset_km', 'population_distance_decay', 'hotel_distance_decay', 'visitor_hotel_mixture']
    details = {'historical_parameters': dict(zip(names, old_parameters.tolist())),
               'historical_cantons': historical.canton_id.tolist(), 'historical_conditional_observed': observed_old.tolist(),
               'historical_conditional_fitted': grouped(historical_weights, cantons, historical.canton_id.to_numpy()).tolist(),
               'latest_free_parameters': names[1:], 'latest_fixed_parameters': [names[0]],
               'latest_observed': {k: v.tolist() for k, v in targets.items()},
               'latest_fitted': {'outgoing': grouped(population, labels, range(4)).tolist(),
                                 'incoming_other': grouped(visitor, labels, range(4)).tolist()},
               'historical_swiss_visitor_share_assumed': swiss_visitor_share,
               'weak_regularisation': .04, 'historical_cost': float(np.sum(old_fit.fun**2)),
               'latest_cost': float(np.sum(latest.fun**2)),
               'historical_parameters_at_bound': [names[i] for i, value in enumerate(old_parameters) if min(value-bounds[0][i], bounds[1][i]-value) < .001],
               'latest_parameters_at_bound': [names[i] for i, value in enumerate(parameters) if min(value-bounds[0][i], bounds[1][i]-value) < .001]}
    return dict(zip(names, parameters.tolist())), details


@dataclass
class AirportModel:
    municipalities: pd.DataFrame
    parameters: dict
    swiss_visitor_share: float
    airports: dict
    distance_offset_km: float
    signature: str
    diagnostics: dict

    def probabilities(self, airport, profile=None):
        if airport not in self.airports:
            raise ValueError(f'Airport coordinates missing: {airport}')
        parameters = [self.parameters[name] for name in ['distance_offset_km', 'population_distance_decay', 'hotel_distance_decay', 'visitor_hotel_mixture']]
        distance = distances(self.municipalities, self.airports[airport])
        population, hotels = kernels(self.municipalities, distance, parameters, self.distance_offset_km)
        mixture = 0. if profile == 'outgoing' else parameters[3]
        if profile is None:
            mixture *= self.swiss_visitor_share
        return (1-mixture)*population + mixture*hotels

    def allocate(self, regional, airport):
        """Disaggregate Swiss rows; keep foreign counts explicit and unresolved."""
        output = []
        features = self.municipalities
        for _, row in regional.iterrows():
            if row.country != 'CH':
                unresolved = row.to_frame().T
                unresolved['municipality_id'] = pd.NA
                unresolved['municipality_name'] = pd.NA
                unresolved['canton_id'] = pd.NA
                unresolved['spatial_resolution'] = 'unresolved_foreign_region'
                unresolved['municipality_probability'] = 1.
                output.append(unresolved)
                continue
            if row.region_id in CANTON_GROUPS:
                mask = features.canton_id.eq(CANTON_GROUPS[row.region_id]).to_numpy()
            elif row.region_id == 'CH-OTHER':
                mask = ~features.canton_id.isin(CANTON_GROUPS.values()).to_numpy()
            elif row.region_id == 'CH':
                mask = np.ones(len(features), dtype=bool)
            else:
                raise ValueError(f'Unknown Swiss airport region: {row.region_id}')
            weights = self.probabilities(airport, row.get('profile', None))
            weights = np.where(mask, weights, 0.)
            if weights.sum() <= 0:
                raise ValueError(f'No municipal support for {row.region_id}')
            weights /= weights.sum()
            selected = np.flatnonzero(weights > 0)
            allocation = weights[selected]
            frame = pd.DataFrame([row.to_dict()] * len(selected))
            for column in ['municipality_id', 'municipality_name', 'canton_id']:
                frame[column] = features.iloc[selected][column].to_numpy()
            frame['municipality_probability'] = allocation
            frame['spatial_resolution'] = 'municipality'
            frame['distance_km'] = distances(features.iloc[selected], self.airports[airport])
            codes = 'CH-MUN-' + frame.municipality_id.astype(str)
            frame['origin_region'] = codes if row.direction == 'to_airport' else airport
            frame['destination_region'] = airport if row.direction == 'to_airport' else codes
            for column in ['annual_trips', 'period_trips', 'daily_trips']:
                frame[column] = float(row[column]) * allocation
                frame[column+'_integer'] = apportion(int(row[column+'_integer']) * allocation)
            output.append(frame)
        result = pd.concat(output, ignore_index=True)
        for column in ['municipality_id', 'canton_id']:
            result[column] = result[column].astype('Int64')
        for column in ['annual_trips', 'period_trips', 'daily_trips']:
            result[column] = pd.to_numeric(result[column])
            result[column+'_integer'] = pd.to_numeric(result[column+'_integer']).astype('int64')
            if not np.isclose(result[column].sum(), regional[column].sum(), rtol=1e-12):
                raise ValueError('Municipality allocation changed airport passenger demand')
            if result[column+'_integer'].sum() != regional[column+'_integer'].sum():
                raise ValueError('Municipality allocation changed integer passenger demand')
        result['origin_municipality_id'] = result.municipality_id.where(result.direction.eq('to_airport'))
        result['destination_municipality_id'] = result.municipality_id.where(result.direction.eq('from_airport'))
        result['allocation_model_signature'] = self.signature
        # Use explicit extension dtypes so first-run and Parquet-reloaded results agree.
        result = result.convert_dtypes()
        for column in ['annual_trips', 'period_trips', 'daily_trips']:
            result[column] = result[column].astype(float)
            result[column+'_integer'] = result[column+'_integer'].astype('int64')
        return result
