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


def test_replay_prepares_common_income_and_chain_inputs_once():
    from replay import prepare_inputs
    from data.constants import Constants
    import geopandas as gpd
    from shapely.geometry import box
    cohort = pd.DataFrame({'person_id':[1], 'trip_id':[1], 'origin_x':[1.], 'origin_y':[2.],
                          'purpose':['shop'], 'origin_purpose':['home']})
    chains = pd.DataFrame({'person_id':[1],'trip_id':[1],'activity_chain':[np.ones(6)],'target_distance':[100.]})
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
    assert inputs.iloc[0].home_ovgk_a == 1
