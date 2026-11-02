from pathlib import Path
import json
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import pytest
import yaml

from cache_io import RunCache
from metrics import mean_ci, standardized_contrast, activity_spaces, paired_replay_summary
from figures import density_grid
from replay import select_cohort


def test_cache_uses_dependencies_not_newest_unrelated_artifact(tmp_path):
    root = 'synthesis.population.spatial.locations__root'
    old, new = 'stage__old', 'stage__new'
    metadata = {root: {'updated': 10, 'dependencies': {old: 1}, 'config': {'locations_algorithm': 'rda'}},
                old: {'updated': 1, 'dependencies': {}}, new: {'updated': 100, 'dependencies': {}}}
    for key, value in [(root, None), (old, 1), (new, 2)]:
        (tmp_path / (key + '.p')).write_bytes(pickle.dumps(value))
    (tmp_path / 'pipeline.json').write_text(json.dumps(metadata))
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump({'working_directory': str(tmp_path), 'config': {}}))
    cache = RunCache(path)
    assert cache.stage('stage') == 1
    assert cache.kind == 'rda'


def test_weighted_person_cluster_mean_accounts_for_repeated_trips():
    frame = pd.DataFrame({'person_id': [1,1,2], 'distance_km': [0.,0.,9.], 'weight': [1.,1.,2.]})
    mean, se, low, high = mean_ci(frame)
    assert mean == pytest.approx(4.5)
    assert se > 0 and low < mean < high


def test_standardization_removes_composition_difference():
    rows = []
    person = 0
    for car, counts in [('No car access', [9,1]), ('Car access', [1,9])]:
        for purpose, count, distance in zip(['shop','leisure'], counts, [1.,10.]):
            for _ in range(count):
                person += 1
                rows.append(dict(person_id=person, car_group=car, purpose=purpose, distance_km=distance, weight=1.))
    f = pd.DataFrame(rows)
    # Unadjusted car means differ solely because of purpose mix.
    assert f.groupby('car_group').distance_km.mean().nunique() == 2
    table = standardized_contrast({'survey': f, 'rda': f.copy()}, 'car_group', 'No car access', 'Car access', ['purpose'], min_people=1)
    np.testing.assert_allclose(table.difference_km, 0)
    np.testing.assert_allclose(table.retained_weight_share, 1)


def test_map_denominator_includes_out_of_frame_destinations():
    frame = pd.DataFrame({'person_id':[1,2], 'origin_x':[500,500], 'origin_y':[500,500],
                          'destination_x':[500,5000], 'destination_y':[500,5000], 'weight':[1.,3.]})
    grid, _, _, n, visible = density_grid(frame, [0,0,1000,1000], 500)
    assert n == 2 and visible == pytest.approx(.25)
    assert grid.sum() == pytest.approx(250)


def test_activity_space_counts_person_once_and_detects_revisits():
    frame = pd.DataFrame({'person_id':[1,1,1], 'weight':[2.,2.,2.], 'destination_x':[2600000,2600000,2601000],
                          'destination_y':[1200000]*3, 'distance_km':[1.,2.,3.], 'home_distance_km':[0.,0.,1.]})
    result = activity_spaces(frame).iloc[0]
    assert result.weight == 2
    assert result.unique_cells_250m == 2
    assert result.revisit_share_250m == pytest.approx(1/3)
    assert result.radius_gyration_km == pytest.approx(np.sqrt(2/9))


def test_common_survey_cohort_is_reproducible_and_person_based():
    frame = pd.DataFrame({'person_id':[1,1,2,2,3], 'trip_id':[1,2,1,2,1],
                          'origin_x':[1]*5,'origin_y':[1]*5})
    regions = {'test': {'bounds':[0,0,10,10]}}
    a, b = select_cohort(frame,regions,2,42), select_cohort(frame,regions,2,42)
    pd.testing.assert_frame_equal(a,b)
    assert a.person_id.nunique() == 2
    for person in a.person_id.unique():
        assert (a.person_id==person).sum() == (frame.person_id==person).sum()


def test_replay_draws_averaged_before_paired_difference():
    observed = pd.DataFrame({'person_id':[1,2], 'trip_id':[1,1], 'distance_km':[1.,1.], 'weight':[1.,3.]})
    predictions = pd.DataFrame({'person_id':[1,1,2,2]*2, 'trip_id':[1]*8,
        'dataset':['nn_distance']*4+['nn_no_distance']*4,
        'distance_km':[1.,3.,1.,3.,2.,4.,3.,5.], 'home_distance_km':[1.]*8})
    summary,_ = paired_replay_summary(observed,predictions)
    paired = summary.loc[summary.dataset=='nn_no_distance_minus_nn_distance'].iloc[0]
    assert paired.signed_distance_difference_km == pytest.approx(1.75)


def test_selector_diagnostic_excludes_stationary_bypass_and_collapses_draws():
    from metrics import selector_diagnostics
    observed = pd.DataFrame({'person_id':[1,2], 'trip_id':[1,1], 'crowfly_distance':[0.,1000.]})
    predictions = pd.DataFrame({'person_id':[1,1,2,2], 'trip_id':[1]*4,
        'dataset':['nn_no_distance']*4,'selected_short':[True]*4,
        'weight':[.5]*4,'car_group':['No car access']*4})
    table,trips = selector_diagnostics(observed,predictions)
    assert len(trips) == 1 and trips.iloc[0].person_id == 2
    assert trips.iloc[0].weight == pytest.approx(1)
    assert table.iloc[0].row_share == pytest.approx(1)


def test_accessibility_prefers_highest_overlapping_class():
    import geopandas as gpd
    from shapely.geometry import box
    from cohorts import accessibility
    polygons = gpd.GeoDataFrame({'ovgk':['A','B']}, geometry=[box(0,0,2,2),box(0,0,3,3)],crs=2056)
    result = accessibility([1.,2.5,4.],[1.,2.5,4.],polygons)
    assert list(result) == ['A','B','None']


def test_replay_prepares_common_income_and_chain_inputs_once(monkeypatch):
    from replay import prepare_inputs
    from data.constants import Constants
    import geopandas as gpd
    from shapely.geometry import box
    cohort = pd.DataFrame({'person_id':[1], 'trip_id':[1], 'origin_x':[1.], 'origin_y':[2.],
                          'purpose':['shop'], 'origin_purpose':['home']})
    chains = pd.DataFrame({'person_id':[1],'trip_id':[1],'activity_chain':[np.ones(6)],'target_distance':[100.], 'chain_state':[np.zeros(17)]})
    persons = pd.DataFrame({'person_id':[1],'age':[30],'sex':[0],'employed':[1], 'income_class':[3],
                           'car_availability':[0], 'home_x':[1.],'home_y':[2.],'work_x':[3.],'work_y':[4.]})
    persons['bike_availability'] = 0
    persons['subscriptions_ga'] = True
    ovgk = gpd.GeoDataFrame({'ovgk': ['A']}, geometry=[box(0, 0, 10, 10)], crs=2056)
    class Reference:
        def candidates(self,name): return [name]
        def stage(self,name):
            return {'data.microcensus.persons': persons, 'data.constants': Constants(),
                    'data.spatial.ovgk': ovgk}.get(name, chains)
    inputs = prepare_inputs(Reference(),cohort)
    assert inputs.iloc[0].income_class == 3
    assert inputs.iloc[0].car_availability == 1
    assert inputs.iloc[0].target_distance == 100

    assert inputs.iloc[0].pt_ga == 1
    assert inputs.iloc[0].bike_available == 1
    assert inputs.iloc[0].home_ovgk == 0

    # Legacy RDA diaries must be rebuilt without changing the cached source.

    from synthesis.population.spatial.secondary_nn import mz_chains
    from synthesis.population.spatial.secondary_nn.chain_state import CHAIN_STATE_FEATURES
    current = chains.copy()
    current['chain_state'] = [np.zeros(len(CHAIN_STATE_FEATURES))]
    chains['chain_state'] = [np.zeros(len(CHAIN_STATE_FEATURES)-1)]
    rebuilt = []
    monkeypatch.setattr(mz_chains, 'execute', lambda source: (rebuilt.append(True) or current))
    refreshed = prepare_inputs(Reference(), cohort)
    assert rebuilt == [True]
    assert len(refreshed.iloc[0].chain_state) == len(CHAIN_STATE_FEATURES)
    assert len(chains.iloc[0].chain_state) == len(CHAIN_STATE_FEATURES)-1


def test_cache_uses_latest_secondary_assignment_and_its_dependencies(tmp_path):
    full = 'synthesis.population.spatial.locations__old'
    old_secondary = 'synthesis.population.spatial.secondary.locations__old'
    new_secondary = 'synthesis.population.spatial.secondary.locations__new'
    old, new = 'stage__old', 'stage__new'
    metadata = {
        full: {'updated': 10, 'dependencies': {old_secondary: 1, old: 1}, 'config': {}},
        old_secondary: {'updated': 9, 'dependencies': {old: 1}, 'config': {}},
        new_secondary: {'updated': 20, 'dependencies': {new: 1},
                        'config': {'locations_algorithm': 'nn', 'secondary_nn_use_target_distance': False}},
        old: {'updated': 1, 'dependencies': {}}, new: {'updated': 19, 'dependencies': {}},
    }
    for key in metadata:
        (tmp_path / (key + '.p')).write_bytes(pickle.dumps(key))
    (tmp_path / 'pipeline.json').write_text(json.dumps(metadata))
    config = tmp_path / 'config.yml'
    config.write_text(yaml.safe_dump({'working_directory': str(tmp_path), 'config': {}}))
    cache = RunCache(config)
    assert cache.root == new_secondary
    assert cache.resolve('stage') == new
    assert cache.kind == 'nn_no_distance'


def test_reconstructs_latest_secondary_origins_and_primary_origins():
    import geopandas as gpd
    from shapely.geometry import Point
    from cohorts import assignment_locations
    secondary = gpd.GeoDataFrame({'person_id': [1, 1, 1], 'trip_index': [1, 2, 4],
                                 'destination_id': ['new1', 'new2', 'new4']},
                                geometry=[Point(100, 0), Point(200, 0), Point(400, 0)], crs=2056)
    work = gpd.GeoDataFrame({'person_id': [1], 'destination_id': ['work']},
                           geometry=[Point(300, 0)], crs=2056)
    education = gpd.GeoDataFrame({'person_id': [], 'destination_id': []}, geometry=[], crs=2056)
    persons = pd.DataFrame({'person_id': [1], 'home_x': [0.], 'home_y': [0.]})
    trips = pd.DataFrame({'person_id': [1, 1, 1], 'trip_index': [0, 1, 3],
                          'preceding_purpose': ['home', 'shop', 'work']})
    class Cache:
        def stage(self, name):
            assert name == 'synthesis.population.spatial.primary.locations'
            return work, education
    locations = assignment_locations(Cache(), trips, secondary, persons).set_index('activity_index')
    assert locations.x.to_dict() == {0: 0., 1: 100., 2: 200., 3: 300., 4: 400.}
    assert locations.loc[1, 'destination_id'] == 'new1'


def test_cache_selects_production_expert_before_deeper_crossfit_versions(tmp_path):
    root = 'synthesis.population.spatial.secondary.locations__root'
    final, fold = 'model__final', 'model__fold'
    selector, crossfit = 'selector__stage', 'crossfit__stage'
    metadata = {root: {'updated': 10, 'dependencies': {final: 1, selector: 1}, 'config': {}},
                final: {'updated': 1, 'dependencies': {}},
                selector: {'updated': 2, 'dependencies': {crossfit: 1}},
                crossfit: {'updated': 3, 'dependencies': {fold: 1}},
                fold: {'updated': 9, 'dependencies': {}}}
    for key in metadata:
        (tmp_path/(key+'.p')).write_bytes(pickle.dumps(key))
    (tmp_path/'pipeline.json').write_text(json.dumps(metadata))
    config=tmp_path/'config.yml'
    config.write_text(yaml.safe_dump({'working_directory':str(tmp_path),'config':{}}))
    cache=RunCache(config)
    assert cache.resolve('model')==final


def test_selector_diagnostics_keep_probabilities_and_actual_draws_separate():
    from metrics import selector_diagnostics
    observed=pd.DataFrame(dict(person_id=[1],trip_id=[1],crowfly_distance=[30000.]))
    predictions=pd.DataFrame(dict(person_id=[1,1,1],trip_id=[1,1,1],dataset=['nn_no_distance']*3,
        selected_short=[True,False,False],short_probability=[.2]*3,weight=[1/3]*3,
        car_group=['Car access']*3))
    table,trips=selector_diagnostics(observed,predictions)
    assert trips.iloc[0].selected_short==pytest.approx(.2)
    assert table.loc[table.selected_short,'row_share'].iloc[0]==pytest.approx(1/3)


def test_replay_fingerprint_hashes_chain_arrays_by_content_without_mutating_inputs():
    import importlib.util
    spec = importlib.util.spec_from_file_location('secondary_report_run', Path(__file__).with_name('run.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(dict(person_id=[1, 2], activity_chain=[np.array([1, 2]), np.array([2, 1])],
                              chain_state=[np.array([100., 2.]), np.array([200., 3.])]))
    fingerprint = module.replay_input_fingerprint(frame)
    copied = frame.copy()
    copied['chain_state'] = copied.chain_state.map(np.copy)
    assert module.replay_input_fingerprint(copied) == fingerprint
    copied.loc[0, 'chain_state'][0] += 1.
    assert module.replay_input_fingerprint(copied) != fingerprint
    assert frame.loc[0, 'chain_state'][0] == 100.
    assert isinstance(frame.loc[0, 'activity_chain'], np.ndarray)


def test_short_range_diagnosis_uses_saved_representative_points():
    import geopandas as gpd
    from shapely.geometry import Point
    from diagnose_short_range import diagnose
    from synthesis.population.spatial.secondary_nn.h3 import H3_LEVEL_NAMES

    # Geometry centroids are deliberately far away; inference uses the saved point column.

    leaves = gpd.GeoDataFrame(dict(
        h3_index=['root/a/1', 'root/a/2', 'root/b/3', 'root/b/4'], num_statent=[1]*4),
        geometry=[Point(10000+i*100, 0) for i in range(4)], crs=2056)
    leaves['centroid'] = gpd.GeoSeries([Point(-1000,0), Point(1000,0), Point(0,1000), Point(0,-1000)], crs=2056)
    labels = pd.DataFrame({'person_id':[1], 'trip_id':[0],
                           'destination_'+H3_LEVEL_NAMES[1]:['root/a']})
    class Cache:
        root = 'frozen'
        def config(self, key, default=None):
            return default
        def stage(self, name):
            return {'microcensus_trips':labels}, {H3_LEVEL_NAMES[-1]:leaves}, None
    observed = pd.DataFrame(dict(person_id=[1], trip_id=[0], origin_x=[0.], origin_y=[0.], crowfly_distance=[100.]))
    result = diagnose(Cache(), observed)
    assert result['median_origin_candidate_cells'] == 4
    assert result['median_observed_branch_candidate_cells'] == 2
    assert result['median_fraction_origin_candidates_outside_training_branch'] == .5
    assert result['share_trips_with_origin_candidates_outside_training_branch'] == 1.


def test_sampling_policy_comes_from_the_completed_assignment_not_edited_yaml(tmp_path):
    root='synthesis.population.spatial.secondary.locations__root'
    linked='synthesis.population.spatial.secondary_nn.locations__linked'
    unrelated='synthesis.population.spatial.secondary_nn.locations__unrelated'
    metadata={root:dict(updated=10,dependencies={linked:1},config={}),
        linked:dict(updated=5,dependencies={},config={}),
        unrelated:dict(updated=20,dependencies={},config={'secondary_nn_short_sampling':'distance'})}
    for key in metadata:
        (tmp_path/(key+'.p')).write_bytes(pickle.dumps(None))
    path=tmp_path/'config.yml'
    path.write_text(yaml.safe_dump(dict(working_directory=str(tmp_path),config={'secondary_nn_short_sampling':'distance'})))
    (tmp_path/'pipeline.json').write_text(json.dumps(metadata))
    assert RunCache(path).config('secondary_nn_short_sampling')=='neural'
    metadata[linked]['config']['secondary_nn_short_sampling']='distance'
    metadata[linked]['config']['secondary_nn_short_distance_sigma_min_m']=15.
    (tmp_path/'pipeline.json').write_text(json.dumps(metadata))
    assert RunCache(path).config('secondary_nn_short_sampling')=='distance'
    assert RunCache(path).config('secondary_nn_short_distance_sigma_min_m')==15.


def test_historical_assignment_can_be_selected_for_reproducible_diagnosis(tmp_path):
    old='synthesis.population.spatial.secondary.locations__old'
    new='synthesis.population.spatial.secondary.locations__new'
    metadata={old:dict(updated=1,dependencies={},config={'secondary_nn_short_sampling':'neural'}),
              new:dict(updated=2,dependencies={},config={'secondary_nn_short_sampling':'distance'})}
    for key in metadata:
        (tmp_path/(key+'.p')).write_bytes(pickle.dumps(None))
    (tmp_path/'pipeline.json').write_text(json.dumps(metadata))
    path=tmp_path/'config.yml'
    path.write_text(yaml.safe_dump(dict(working_directory=str(tmp_path),config={})))
    assert RunCache(path).root==new
    assert RunCache(path,root=old).config('secondary_nn_short_sampling')=='neural'
    with pytest.raises(ValueError,match='not available'):
        RunCache(path,root='missing')


def test_short_loss_provenance_uses_completed_checkpoint_history(tmp_path):
    root='synthesis.population.spatial.secondary.locations__root'
    stage='synthesis.population.spatial.secondary_nn.short_range_model'
    short=stage+'__trained'
    checkpoint=tmp_path/'short.pt'
    checkpoint.write_bytes(b'checkpoint')
    (tmp_path/'loss_history.json').write_text(json.dumps(dict(distance_loss_kind='relative',
        distance_loss_weight=.07,relative_distance_underestimation_factor=2.,best_epoch=21)))
    metadata={root:dict(updated=2,dependencies={short:1},config={}),
              short:dict(updated=1,dependencies={},config={})}
    (tmp_path/(root+'.p')).write_bytes(pickle.dumps(None))
    (tmp_path/(short+'.p')).write_bytes(pickle.dumps((str(checkpoint),)))
    (tmp_path/'pipeline.json').write_text(json.dumps(metadata))
    config=tmp_path/'config.yml'
    config.write_text(yaml.safe_dump(dict(working_directory=str(tmp_path),config={})))
    cache=RunCache(config)
    cache.stage(stage)
    recorded=cache.provenance()['artifacts'][stage]['training_settings']
    assert recorded['relative_distance_underestimation_factor']==2.
    assert recorded['distance_loss_kind']=='relative' and recorded['distance_loss_weight']==.07
