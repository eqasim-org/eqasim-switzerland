"""A cross-fitted probabilistic mixture of local and hierarchical destinations.

The historical stage name is retained for pipeline compatibility. The gate is
fitted to destination likelihood and proper distribution scores, never to an
observed-distance threshold or an estimated winner's sign.
"""

from dataclasses import dataclass
import copy
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from .feature_encoding import fit_person_trip_matrix, transform_person_static_vector, transform_person_dynamic_vector
from .hierarchical_utils import SECONDARY_ACTIVITIES, ORIGIN_PURPOSE_REMAP, DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION
from .tools import MOBILITY_FEATURES, OriginContext, training_mobility, selector_origin_context
from .distribution_scores import ensemble_score_components, mixture_score

logger = logging.getLogger('synpp: expert_selector')
MODEL_NAME = 'expert_selector.pkl'


def configure(context):
    context.config('secondary_nn_use_target_distance', True)
    if context.config('secondary_nn_use_target_distance'):
        return
    for stage in ['data.microcensus.trips', 'data.microcensus.persons', 'data.constants',
                  'data.spatial.ovgk', 'synthesis.population.destinations',
                  'synthesis.population.spatial.secondary.detour_factors.factors']:
        context.stage(stage)
    for stage in ['mz_chains', 'h3', 'training_split', 'cross_fitted_experts',
                  'regional_model', 'subregional_model', 'local_model', 'short_range_model']:
        context.stage('synthesis.population.spatial.secondary_nn.' + stage)
    context.config('random_seed')
    context.config('threads', volatile=True)
    context.config('secondary_nn_short_trip_threshold_m', 1500.0)
    context.config('secondary_nn_short_query_radius_m', 200.0)
    context.config('secondary_nn_short_min_candidates', 6)
    context.config('secondary_nn_selector_draws', 10)
    context.config('secondary_nn_selector_context_radii_m', [1500.0, 5000.0, 20000.0])
    context.config('secondary_nn_selector_max_iter', 300)
    context.config('secondary_nn_selector_learning_rate', .01)
    context.config('secondary_nn_selector_weight_decay', .001)
    context.config('secondary_nn_selector_patience', 20)
    context.config('secondary_nn_selector_crps_weight', .1)
    context.config('secondary_nn_selector_energy_weight', .1)
    context.config('secondary_nn_distribution_scale_m', 1000.0)


def execute(context):

    # Use out-of-fold expert distributions for every gate development person.

    if context.config('secondary_nn_use_target_distance'):
        return None
    from .model_wrappers import HierarchicalLocationChoiceModel
    torch.set_num_threads(max(1, int(context.config('threads'))))
    frame = prepare_training_data(context)
    out_of_fold = pd.read_parquet(context.stage('synthesis.population.spatial.secondary_nn.cross_fitted_experts')[0])

    # Evaluate the final refitted experts on outer people untouched by all fitting.

    wrapper = HierarchicalLocationChoiceModel.build(context, optimize=False, load_selector=False)
    validation = frame.loc[frame.partition.eq('validation')]
    with context.progress(label='Outer expert validation', total=len(validation)) as progress:
        held_out = compare_experts(validation, wrapper, draws=int(context.config('secondary_nn_selector_draws')),
                                   seed=int(context.config('random_seed')),
                                   scale_m=float(context.config('secondary_nn_distribution_scale_m')), progress=progress)
    held_out['prediction_fold'] = -1
    comparisons = pd.concat([out_of_fold, held_out], ignore_index=True)
    comparisons.to_parquet(Path(context.path()) / 'expert_comparisons.parquet', index=False)

    # Fit and stop on development people before scoring the outer validation set.

    selector, audit = train_selector(
        frame, comparisons, selector_origin_context(context),
        seed=int(context.config('random_seed')), max_iter=int(context.config('secondary_nn_selector_max_iter')),
        learning_rate=float(context.config('secondary_nn_selector_learning_rate')),
        weight_decay=float(context.config('secondary_nn_selector_weight_decay')),
        patience=int(context.config('secondary_nn_selector_patience')),
        crps_weight=float(context.config('secondary_nn_selector_crps_weight')),
        energy_weight=float(context.config('secondary_nn_selector_energy_weight')))

    # Persist validation coverage, distribution diagnostics and the fitted mixture.

    selector.metrics['draws_per_expert'] = int(context.config('secondary_nn_selector_draws'))
    selector.origin_context.cache.clear()
    path = Path(context.path()) / MODEL_NAME
    joblib.dump(selector, path)
    audit.to_parquet(Path(context.path()) / 'selector_partitions.parquet', index=False)
    (Path(context.path()) / 'validation.json').write_text(json.dumps(selector.metrics, indent=2))
    validation_diagnostics(frame, comparisons, audit).to_csv(Path(context.path()) / 'validation_by_distance.csv', index=False)
    logger.info('Probabilistic mixture outer validation: %s', selector.metrics)
    ratio = selector.metrics['validation_mean_distance_ratio']
    if not .7 <= ratio <= 1.3:
        logger.warning('Outer validation mean-distance ratio %.3f is outside [0.7, 1.3]; inspect distribution diagnostics before using these models', ratio)
    return (str(path),)


def prepare_training_data(context):
    trips = context.stage('data.microcensus.trips')[0][[
        'person_id', 'trip_id', 'purpose', 'origin_purpose', 'origin_x', 'origin_y', 'destination_x', 'destination_y']]
    chains = context.stage('synthesis.population.spatial.secondary_nn.mz_chains')
    valid = context.stage('synthesis.population.spatial.secondary_nn.h3')[0]['microcensus_trips'][[
        'person_id', 'trip_id', 'destination_level_2']].rename(columns={'destination_level_2': 'observed_leaf'})
    persons = context.stage('data.microcensus.persons')[[
        'person_id', 'age', 'sex', 'employed', 'car_availability', 'income_class',
        'weekend', 'person_weight', 'home_x', 'home_y', 'work_x', 'work_y']].copy()
    persons = persons.loc[~persons.weekend].drop(columns='weekend')
    persons['car_availability'] = persons.car_availability != context.stage('data.constants').CAR_AVAILABILITY_NEVER
    frame = trips.merge(chains, on=['person_id', 'trip_id'], validate='one_to_one')
    frame = frame.merge(valid, on=['person_id', 'trip_id'], validate='one_to_one')
    frame = frame.merge(persons, on='person_id', validate='many_to_one')
    frame = frame.merge(context.stage('synthesis.population.spatial.secondary_nn.training_split'), on='person_id', validate='many_to_one')
    coordinates = frame[['origin_x', 'origin_y', 'destination_x', 'destination_y', 'home_x', 'home_y']]
    distance = frame.target_distance.to_numpy(dtype=float)
    eligible = (frame.purpose.isin(SECONDARY_ACTIVITIES) & np.isfinite(coordinates).all(axis=1)
                & np.isfinite(distance) & (distance >= DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION))
    frame = frame.loc[eligible].sort_values(['person_id', 'trip_id']).reset_index(drop=True)
    frame['has_work'] = np.isfinite(frame.work_x) & np.isfinite(frame.work_y)
    frame.loc[~frame.has_work, ['work_x', 'work_y']] = 0.
    frame[MOBILITY_FEATURES] = training_mobility(context, frame)
    return frame


def prediction_inputs(row):
    return dict(
        person_id=row.person_id, home_x=row.home_x, home_y=row.home_y,
        work_x=row.work_x, work_y=row.work_y, origin_x=row.origin_x, origin_y=row.origin_y,
        age=row.age, daily_longest_distance_from_home=row.daily_longest_distance_from_home,
        daily_crowfly_total=row.daily_crowfly_total, daily_longest_distance_from_work=row.daily_longest_distance_from_work,
        crowfly_consumed_before_trip=row.crowfly_consumed_before_trip, trip_position_class=row.trip_position_class,
        departure_time_normalized=row.departure_time_normalized, activity_duration_h=row.activity_duration_h,
        target_distance=0., activity_chain_vector=row.activity_chain, sex=row.sex, employed=row.employed,
        car_availability=row.car_availability, income_class=row.income_class,
        purpose=row.purpose, origin_purpose=row.origin_purpose, has_work=row.has_work,
        has_education=False, mobility_vector=np.asarray([getattr(row, c) for c in MOBILITY_FEATURES]),
        chain_state_vector=row.chain_state)


def compare_experts(frame, wrapper, *, draws, seed, scale_m=1000., progress=None):
    if draws < 2:
        raise ValueError('Expert comparison requires at least two destination draws')
    records = []
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            inputs = prediction_inputs(row)
            short_probability, hierarchy_probability = wrapper.expert_leaf_probabilities(row.observed_leaf, **inputs)
            record = dict(person_id=row.person_id, trip_id=row.trip_id,
                          short_leaf_probability=short_probability, hierarchy_leaf_probability=hierarchy_probability)
            predictions = []
            for expert_index, expert in enumerate(['short', 'hierarchy']):
                points = []
                for draw in range(draws):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, int(row.person_id), int(row.trip_id), expert_index, draw]))
                    _, point = wrapper.predict(**inputs, rng=rng, force_expert=expert)
                    points.append([point.x, point.y])
                points = np.asarray(points)
                distances = np.linalg.norm(points - [row.origin_x, row.origin_y], axis=1)
                record[expert + '_mean_distance_m'] = float(distances.mean())
                record[expert + '_short_share'] = float((distances <= 1500).mean())
                record[expert + '_long_share'] = float((distances > 20000).mean())
                predictions.append(points)
            record.update(ensemble_score_components(*predictions, [row.origin_x, row.origin_y],
                                                     [row.destination_x, row.destination_y], scale_m=scale_m))
            records.append(record)
            if progress is not None:
                progress.update()
    return pd.DataFrame(records)


def _feature_matrix(frame, context, static_scaler=None, dynamic_scaler=None, seed=None):
    def numeric(column):
        values = frame[column].to_numpy(dtype=float)
        return np.where(np.isfinite(values), np.maximum(values, 0.), 0.)
    chains = np.stack(frame.activity_chain.map(lambda v: np.asarray(v, dtype=float)[:len(SECONDARY_ACTIVITIES)]))
    matrix, _, _, static, dynamic, names = fit_person_trip_matrix(
        age=numeric('age'), sex=numeric('sex'), employed=numeric('employed'),
        car_availability=numeric('car_availability'), income_class=numeric('income_class'),
        daily_longest=numeric('daily_longest_distance_from_home'), daily_total=numeric('daily_crowfly_total'),
        daily_longest_work=numeric('daily_longest_distance_from_work'), activity_chain_matrix=chains,
        consumed_before=numeric('crowfly_consumed_before_trip'), trip_position=numeric('trip_position_class'),
        departure_time=numeric('departure_time_normalized'), activity_duration_h=numeric('activity_duration_h'),
        target_distance=np.zeros(len(frame)), purpose_series=frame.purpose, origin_purpose_series=frame.origin_purpose,
        purpose_categories=SECONDARY_ACTIVITIES, person_static_scaler=static_scaler,
        person_dynamic_scaler=dynamic_scaler, use_target_distance=False, random_state=seed,
        mobility_matrix=frame[MOBILITY_FEATURES].to_numpy(), chain_state_matrix=np.stack(frame.chain_state))
    origins = np.stack([context.vector(r.origin_x, r.origin_y, r.home_x, r.home_y, r.work_x, r.work_y, r.has_work, purpose=r.purpose)
                        for r in frame.itertuples(index=False)])
    return np.concatenate([matrix, origins], axis=1), static, dynamic, names + context.names


@dataclass
class ExpertSelector:
    network: object
    static_scaler: object
    dynamic_scaler: object
    feature_scaler: object
    origin_context: OriginContext
    feature_names: list
    metrics: dict

    def predict_probability(self, *, age, income_class, daily_longest, daily_total,
                            daily_longest_work, activity_chain_vector, sex, employed,
                            car_availability, consumed_before, trip_position, departure_time,
                            activity_duration_h, purpose, origin_purpose, mobility_vector,
                            origin_x, origin_y, home_x, home_y, work_x, work_y, has_work,
                            chain_state_vector):
        static = transform_person_static_vector(
            age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_vector,
            sex, employed, car_availability, self.static_scaler, mobility_vector=mobility_vector)
        identity = np.eye(len(SECONDARY_ACTIVITIES), dtype=np.float32)
        origin = ORIGIN_PURPOSE_REMAP.get(origin_purpose, origin_purpose)
        dynamic = transform_person_dynamic_vector(
            consumed_before, trip_position, departure_time, activity_duration_h, 0., identity[SECONDARY_ACTIVITIES.index(purpose)],
            identity[SECONDARY_ACTIVITIES.index(origin)] if origin in SECONDARY_ACTIVITIES else np.zeros(len(identity)),
            self.dynamic_scaler, chain_state_vector=chain_state_vector)
        context = self.origin_context.vector(origin_x, origin_y, home_x, home_y, work_x, work_y, has_work, purpose=purpose)
        x = np.concatenate([static, dynamic, context[None, :]], axis=1)
        x = self.feature_scaler.transform(np.log1p(np.maximum(x, 0.))).astype(np.float32)
        with torch.inference_mode():
            return float(torch.sigmoid(self.network(torch.from_numpy(x))).item())


def _network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1))


def mixture_objective(logits, statistics, crps_weight=.1, energy_weight=.1):
    log_short = torch.log(statistics['short_leaf_probability'].clamp_min(1e-30))
    log_hierarchy = torch.log(statistics['hierarchy_leaf_probability'].clamp_min(1e-30))
    likelihood = -torch.logsumexp(torch.stack([
        nn.functional.logsigmoid(logits) + log_short,
        nn.functional.logsigmoid(-logits) + log_hierarchy]), dim=0)
    probability = torch.sigmoid(logits)
    return (likelihood + crps_weight * mixture_score(probability, statistics, 'crps')
            + energy_weight * mixture_score(probability, statistics, 'energy'))


def train_selector(frame, comparisons, origin_context, *, seed, max_iter=300, learning_rate=.01,
                   weight_decay=.001, patience=20, crps_weight=.1, energy_weight=.1):
    if max_iter < 1 or learning_rate <= 0 or min(weight_decay, crps_weight, energy_weight) < 0 or patience < 1:
        raise ValueError('Invalid probabilistic selector controls')
    if len(frame) != len(comparisons):
        raise ValueError('Every eligible trip requires exactly one expert comparison')
    frame = frame.merge(comparisons, on=['person_id', 'trip_id'], how='left', validate='one_to_one', indicator=True)
    if not frame.pop('_merge').eq('both').all():
        raise ValueError('Expert comparisons do not cover every eligible trip')
    if frame.person_id.nunique() < 10:
        raise ValueError('Selector requires at least ten independent persons')
    if frame.groupby('person_id').partition.nunique().gt(1).any():
        raise ValueError('A selector person occurs in multiple partitions')
    indices = {name: np.flatnonzero(frame.partition.eq(name)) for name in ['train', 'stopping', 'validation']}
    if any(not len(rows) for rows in indices.values()):
        raise ValueError('Train, stopping and outer validation must all contain trips')
    development = frame.partition.ne('validation')
    if not frame.loc[development, 'prediction_fold'].eq(frame.loc[development, 'fold']).all():
        raise ValueError('Selector development predictions must come from the person\'s excluded fold')
    if not frame.loc[~development, 'prediction_fold'].eq(-1).all():
        raise ValueError('Outer validation must use the final development-only experts')
    fit = indices['train']
    _, static, dynamic, names = _feature_matrix(frame.iloc[fit], origin_context, seed=seed)
    x, _, _, _ = _feature_matrix(frame, origin_context, static, dynamic)
    scaler = StandardScaler().fit(np.log1p(np.maximum(x[fit], 0.)))
    x = torch.as_tensor(scaler.transform(np.log1p(np.maximum(x, 0.))), dtype=torch.float32)
    columns = ['short_leaf_probability', 'hierarchy_leaf_probability'] + [c for c in comparisons if c.startswith(('energy_', 'crps_'))]
    statistics = {name: torch.as_tensor(frame[name].to_numpy(dtype=np.float32)) for name in columns}
    if any(not frame[column].between(0., 1.).all() for column in ['short_leaf_probability', 'hierarchy_leaf_probability']):
        raise ValueError('Expert leaf probabilities must lie in [0, 1]')
    weight = torch.as_tensor(frame.person_weight.to_numpy(dtype=np.float32))
    if not torch.isfinite(weight).all() or (weight <= 0).any() or any(not torch.isfinite(v).all() for v in statistics.values()):
        raise ValueError('Finite scores and positive weights are required')
    def loss(network, rows):
        logits = network(x[rows]).squeeze(-1)
        values = mixture_objective(logits, {key: value[rows] for key, value in statistics.items()}, crps_weight, energy_weight)
        return (values * weight[rows]).sum() / weight[rows].sum()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = _network(x.shape[1])
        initial = copy.deepcopy(network.state_dict())
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best, best_epoch, checks = np.inf, 0, 0
        for epoch in range(max_iter):
            network.train()
            order = fit[torch.randperm(len(fit)).numpy()]
            for start in range(0, len(order), 512):
                optimizer.zero_grad(set_to_none=True)
                objective = loss(network, order[start:start + 512])
                objective.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
            network.eval()
            with torch.inference_mode():
                stopping = float(loss(network, indices['stopping']))
            if stopping < best - 1e-5:
                best, best_epoch, checks = stopping, epoch + 1, 0
            else:
                checks += 1
            if checks >= patience:
                break

        # Refit for the selected epoch count using all out-of-fold development rows.

        network.load_state_dict(initial)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        rows = np.flatnonzero(development)
        for epoch in range(best_epoch):
            order = rows[torch.randperm(len(rows)).numpy()]
            for start in range(0, len(order), 512):
                optimizer.zero_grad(set_to_none=True)
                objective = loss(network, order[start:start + 512])
                objective.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.)
                optimizer.step()
        network.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(network(x)).squeeze(-1).numpy()
            outer_loss = float(loss(network, indices['validation']))
    validation = frame.iloc[indices['validation']]
    g = probability[indices['validation']]
    average = lambda values: float(np.average(values, weights=validation.person_weight))
    expected_distance = g * validation.short_mean_distance_m + (1 - g) * validation.hierarchy_mean_distance_m
    observed = np.hypot(validation.destination_x - validation.origin_x, validation.destination_y - validation.origin_y)
    metrics = dict(validation_rows=len(validation), validation_persons=int(validation.person_id.nunique()),
                   development_rows=int(development.sum()), development_persons=int(frame.loc[development, 'person_id'].nunique()),
                   refit_epochs=int(best_epoch), outer_mixture_objective=outer_loss,
                   validation_short_expert_probability=average(g),
                   validation_mean_distance_m=average(expected_distance), validation_observed_mean_distance_m=average(observed),
                   validation_mean_distance_ratio=average(expected_distance) / max(average(observed), 1.),
                   validation_predicted_long_share=average(g * validation.short_long_share + (1 - g) * validation.hierarchy_long_share),
                   validation_observed_long_share=average(observed > 20000),
                   validation_unsupported_leaf_share=average((validation.short_leaf_probability + validation.hierarchy_leaf_probability) == 0),
                   gate_uses_target_distance=False, experts_exclude_outer_validation=True,
                   development_predictions_are_cross_fitted=True, probabilistic_routing=True)
    audit = frame[['person_id', 'trip_id', 'partition', 'fold', 'prediction_fold']].copy()
    audit['short_probability'] = probability
    return ExpertSelector(network, static, dynamic, scaler, origin_context, names, metrics), audit


def validation_diagnostics(frame, comparisons, audit):
    data = frame.merge(comparisons, on=['person_id', 'trip_id'], validate='one_to_one')
    data = data.merge(audit[['person_id', 'trip_id', 'short_probability']].rename(columns={'short_probability': 'gate_probability'}),
                      on=['person_id', 'trip_id'], validate='one_to_one')
    data = data.loc[data.partition.eq('validation')].copy()
    observed = np.hypot(data.destination_x - data.origin_x, data.destination_y - data.origin_y)
    data['observed_m'] = observed
    data['distance_bin'] = pd.cut(observed, [0, 1500, 5000, 20000, np.inf])
    rows = []
    for label, group in data.groupby('distance_bin', observed=True):
        average = lambda value: float(np.average(value, weights=group.person_weight))
        rows.append(dict(distance_bin=str(label), trips=len(group), persons=group.person_id.nunique(),
                         short_expert_probability=average(group.gate_probability), observed_mean_m=average(group.observed_m),
                         predicted_mean_m=average(group.gate_probability * group.short_mean_distance_m
                                                  + (1 - group.gate_probability) * group.hierarchy_mean_distance_m)))
    return pd.DataFrame(rows)
