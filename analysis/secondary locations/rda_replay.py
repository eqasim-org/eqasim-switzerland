"""Observed-origin single-leg RDA replay using the production solver unchanged."""
import copy
import logging
from types import SimpleNamespace
import geopandas as gpd
import numpy as np
import pandas as pd
from cohorts import accessibility, decorate

LOG = logging.getLogger(__name__)


def replay_rda(cache, cohort, inputs, draws, seed):
    from synthesis.population.spatial.secondary import locations_rda

    # Load observed travel times and private copies of the calibrated RDA distributions.

    raw = cache.stage('data.microcensus.trips')[0]
    times = raw[['person_id','trip_id','departure_time','arrival_time','mode']].copy()
    times['travel_time'] = times.arrival_time - times.departure_time
    frame = inputs.merge(times[['person_id','trip_id','travel_time','mode']],
                         on=['person_id','trip_id'],validate='one_to_one').reset_index(drop=True)
    if len(frame) != len(cohort) or not np.isfinite(frame.travel_time).all() or (frame.travel_time < 0).any():
        raise ValueError('RDA replay needs a finite nonnegative travel time for every common survey trip')
    destinations = locations_rda.prepare_destinations(SimpleNamespace(stage=cache.stage))
    distributions = copy.deepcopy(cache.stage('synthesis.population.spatial.secondary.distance_distributions'))
    locations_rda.resample_distributions(distributions, dict(car=.8,car_passenger=1.,pt=1.,bike=0.,walk=0.,
        walk_loop=0.,bike_loop=0.,car_loop=.8,pt_loop=1.,car_passenger_loop=1.,remote_walk=0.))
    detour = cache.stage('synthesis.population.spatial.secondary.detour_factors.factors') if cache.config('consider_detour_factor_in_sl',True) else None
    can_drive = frame.car_availability.astype(bool) & frame.driving_license.astype(bool)
    records = []

    # Reset each trip to its observed origin; do not expose its observed destination.

    for draw in range(draws):
        for car in [False,True]:
            selected = frame.loc[can_drive.eq(car)]
            if selected.empty:
                continue
            identifiers = selected.index.to_numpy()
            trips = pd.DataFrame(dict(person_id=identifiers,trip_index=0,preceding_purpose='home',
                following_purpose=selected.purpose.to_numpy(),mode=selected['mode'].to_numpy(),
                travel_time=selected.travel_time.to_numpy(),activity_duration=selected.activity_duration_h.to_numpy()*3600))
            primary = gpd.GeoDataFrame(dict(person_id=identifiers),
                geometry=gpd.points_from_xy(selected.origin_x,selected.origin_y),crs=2056)
            primary['home'] = primary.geometry
            primary['work'] = primary.geometry
            primary['education'] = primary.geometry
            data = dict(distance_distributions=distributions,detour_factor=detour)
            context = SimpleNamespace(data=lambda name:copy.deepcopy(destinations) if name=='destinations' else data[name],
                                      progress=SimpleNamespace(update=lambda:None))
            rng_seed = int(np.random.SeedSequence([seed,draw,int(car),7241]).generate_state(1)[0])
            locations, convergence = locations_rda.process(context,(trips,primary,rng_seed,'EPSG:2056',car))
            if len(locations)!=len(selected) or not locations.person_id.is_unique:
                raise ValueError('RDA replay did not return exactly one destination per survey trip')
            points = locations.sort_values('person_id')
            result = frame.loc[points.person_id.to_numpy(),['person_id','trip_id']].copy()
            result['draw'],result['prediction_x'],result['prediction_y'] = draw,points.geometry.x.to_numpy(),points.geometry.y.to_numpy()
            result['rda_converged'] = convergence.valid.to_numpy()
            records.append(result)
            LOG.info('RDA replay draw %s, can_drive=%s: %s trips, %.1f%% converged',draw,car,len(points),100*convergence.valid.mean())

    # Keep the common replay schema; RDA has no short-expert selector.

    points = pd.concat(records,ignore_index=True)
    result = cohort.drop(columns='dataset').merge(points,on=['person_id','trip_id'],validate='one_to_many')
    result['destination_x'],result['destination_y'] = result.pop('prediction_x'),result.pop('prediction_y')
    result['weight'] /= draws
    result['selected_short'],result['short_probability'] = np.nan,np.nan
    result['destination_pt'] = accessibility(result.destination_x,result.destination_y,cache.stage('data.spatial.ovgk'))
    return decorate(result,'rda')
