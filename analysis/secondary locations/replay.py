"""Observed-origin replay of the saved NN models; no model fitting."""
import gc
import logging
import numpy as np
import pandas as pd
import torch
from cohorts import decorate, accessibility

LOG = logging.getLogger(__name__)


def select_cohort(survey, regions, max_people, seed):
    # Sample persons uniformly from the union of city origin cohorts, preserving
    # all eligible trips of selected persons; selection is shared by both NNs.
    local = np.zeros(len(survey), dtype=bool)
    for region in regions.values():
        x0, y0, x1, y1 = region['bounds']
        local |= survey.origin_x.between(x0, x1) & survey.origin_y.between(y0, y1)
    candidates = np.sort(survey.loc[local, 'person_id'].unique())
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=min(len(candidates), max_people), replace=False)
    return survey.loc[survey.person_id.isin(selected)].sort_values(['person_id', 'trip_id']).copy()


def prepare_inputs(source, cohort):
    """One reference diary/attribute realization for both fitted models.

    In particular, independently imputed survey incomes differ across run caches.
    Holding IDs fixed alone would therefore not isolate model differences.
    """
    from synthesis.population.spatial.secondary_nn import mz_chains
    from synthesis.population.spatial.secondary_nn.chain_state import CHAIN_STATE_FEATURES
    from synthesis.population.spatial.secondary_nn.tools import MOBILITY_FEATURES, mobility_features
    name = 'synthesis.population.spatial.secondary_nn.mz_chains'
    chains = source.stage(name) if source.candidates(name) else mz_chains.execute(source)
    if ('chain_state' not in chains or
            any(np.asarray(state).shape != (len(CHAIN_STATE_FEATURES),) for state in chains.chain_state)):
        # Reconstruct missing diagnostic features locally; never rewrite an old run cache.
        chains = mz_chains.execute(source)
    survey_persons = source.stage('data.microcensus.persons')
    mobility = mobility_features(survey_persons, source.stage('data.constants'), survey=True,
                                 ovgk=source.stage('data.spatial.ovgk'))
    persons = survey_persons[[
        'person_id', 'age', 'sex', 'employed', 'income_class', 'car_availability', 'home_x', 'home_y', 'work_x', 'work_y']].copy()
    persons[MOBILITY_FEATURES] = mobility.to_numpy()
    persons['car_availability'] = persons.car_availability.ne(2).astype(int)
    frame = cohort[['person_id', 'trip_id', 'origin_x', 'origin_y', 'purpose', 'origin_purpose']].merge(
        chains, on=['person_id', 'trip_id'], validate='one_to_one').merge(persons, on='person_id', validate='many_to_one')
    if len(frame) != len(cohort):
        raise ValueError('Replay could not match every survey cohort trip')
    return frame


def replay(cache, cohort, inputs, draws, seed, threads, shared_detour=None):
    from synthesis.population.spatial.secondary_nn.model_wrappers import HierarchicalLocationChoiceModel
    from synthesis.population.spatial.secondary_nn.tools import MOBILITY_FEATURES
    from synthesis.population.spatial.secondary_nn.locations import DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION

    # Retain dependency objects only for this model's reconstruction.
    class Context:
        def stage(self, name):
            if name.endswith('detour_factors.factors') and shared_detour is not None:
                if cache.resolve(name) == shared_detour[0]:
                    # Record this run's identical dependency even when sharing RAM.
                    path = cache.directory / (shared_detour[0] + '.p')
                    stat = path.stat()
                    cache.used[name] = dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, completed=cache.meta[shared_detour[0]]['updated'])
                    return shared_detour[1]
            return cache.stage(name, retain=True)
        def config(self, name):
            return cache.config(name)
    context = Context()
    torch.set_num_threads(threads)
    LOG.info('Loading saved %s models for %d survey persons', cache.kind, cohort.person_id.nunique())
    wrapper = HierarchicalLocationChoiceModel.build(context, optimize=False,
        accessibility_weight=float(cache.config('secondary_nn_no_car_no_pt_weight', 1.)))
    wrapper.assert_ready_for_prediction()
    # Inputs are prepared once from the reference survey realization. Each trip
    # starts at its observed origin; this is not a sequential rollout.
    frame = inputs
    records = []
    with torch.inference_mode():
        for i, row in enumerate(frame.itertuples(index=False)):
            work = np.isfinite(row.work_x) and np.isfinite(row.work_y)
            for draw in range(draws):
                # Stable common seeds across NN variants, independent of ordering.
                rng = np.random.RandomState(np.random.SeedSequence([seed, int(row.person_id), int(row.trip_id), draw]).generate_state(1)[0])
                distance = float(row.target_distance)
                if distance < DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION:
                    x, y = row.origin_x, row.origin_y
                    short = True
                else:
                    xw, yw = (row.work_x, row.work_y) if work else (0., 0.)
                    mobility = np.asarray([getattr(row, column) for column in MOBILITY_FEATURES], dtype=np.float32)
                    _, geom = wrapper.predict(person_id=int(row.person_id), home_x=row.home_x, home_y=row.home_y,
                        work_x=xw, work_y=yw, origin_x=row.origin_x, origin_y=row.origin_y,
                        age=row.age, daily_longest_distance_from_home=row.daily_longest_distance_from_home,
                        daily_crowfly_total=row.daily_crowfly_total, daily_longest_distance_from_work=row.daily_longest_distance_from_work,
                        crowfly_consumed_before_trip=row.crowfly_consumed_before_trip, trip_position_class=row.trip_position_class,
                        departure_time_normalized=row.departure_time_normalized, activity_duration_h=row.activity_duration_h,
                        target_distance=distance, activity_chain_vector=row.activity_chain, sex=row.sex, employed=row.employed,
                        car_availability=row.car_availability, income_class=row.income_class, purpose=row.purpose,
                        origin_purpose=row.origin_purpose, has_work=work, has_education=False, rng=rng, mobility_vector=mobility,
                        chain_state_vector=row.chain_state)
                    x, y = geom.x, geom.y
                    short = wrapper.last_selected_expert == "short"
                probability = 1.0 if distance < DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION else wrapper.last_short_probability
                records.append((row.person_id, row.trip_id, draw, x, y, short, probability))
            if (i + 1) % 250 == 0:
                LOG.info('%s replay: %d/%d survey trips', cache.kind, i + 1, len(frame))
    points = pd.DataFrame(records, columns=['person_id', 'trip_id', 'draw', 'prediction_x', 'prediction_y', 'selected_short', 'short_probability'])
    frame = cohort.drop(columns=['dataset']).merge(points, on=['person_id', 'trip_id'], validate='one_to_many')
    frame['destination_x'], frame['destination_y'] = frame.pop('prediction_x'), frame.pop('prediction_y')
    frame['weight'] /= draws
    frame['destination_pt'] = accessibility(frame.destination_x, frame.destination_y, context.stage('data.spatial.ovgk'))
    result = decorate(frame, cache.kind)
    detour = (cache.resolve('synthesis.population.spatial.secondary.detour_factors.factors'), wrapper.detour_factor)
    del wrapper, context
    cache.keep.clear()
    gc.collect()
    return result, detour
