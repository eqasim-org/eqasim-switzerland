"""Distance conditioning must keep pooled OD directions, support and all people."""

import numpy as np
import pandas as pd

from synthesis.population.spatial.primary.education.secondary import (
    assign_secondary_od_locations, sample_secondary_zones)


def test_short_and_long_programme_targets_share_one_od_row():
    people = pd.DataFrame({"home_x":np.zeros(2000), "home_y":0.})
    # Both targets use the same mostly long-distance pooled secondary OD row.
    targets = np.r_[np.full(1000,2000.),np.full(1000,20000.)]
    zones, valid, sigma = sample_secondary_zones(people, [[2000.,0.],[20000.,0.]],
        [10,70], [np.array([0]),np.array([1])], [.1,.9], targets,np.random.RandomState(0))
    assert valid.all() and np.isfinite(sigma).all()
    assert (zones[:1000] == 10).mean() > .99
    assert (zones[1000:] == 70).mean() > .99


def test_equal_distance_zones_keep_od_direction_without_site_count_bonus():
    people = pd.DataFrame({"home_x":np.zeros(8000),"home_y":0.})
    # One direction has three duplicate site locations, the other only one.
    zones, _, _ = sample_secondary_zones(people,
        [[1000.,0.]]*3+[[0.,1000.]], [10,70], [np.array([0,1,2]),np.array([3])],
        [.8,.2], np.full(len(people),1000.),np.random.RandomState(0))
    assert abs((zones == 10).mean() - .8) < .02


def test_missing_ss_evidence_keeps_prior_and_zero_routes_stay_zero():
    people = pd.DataFrame({"home_x":np.zeros(8000),"home_y":0.})
    zones, valid, sigma = sample_secondary_zones(people, [[1000.,0.],[20000.,0.]],
        [10,70],[np.array([0]),np.array([1])],[.2,.8],np.full(len(people),np.nan),np.random.RandomState(0))
    assert not valid.any() and np.isnan(sigma).all()
    assert abs((zones == 70).mean() - .8) < .02
    class ZeroDraw:
        def random_sample(self,n):return np.zeros(n)
    zones, _, _ = sample_secondary_zones(people.iloc[:1],[[0.,0.],[1000000.,0.]],
        [10,70],[np.array([0]),np.array([1])],[0.,1.],[0.],ZeroDraw())
    # Even an exactly-zero random draw cannot select an unsupported near school.
    assert zones.tolist() == [70]


def test_wrapper_preserves_scope_and_quarter_membership():
    people = pd.DataFrame({"person_id":range(1,8),"age":[16,16,14,20,16,16,16],
        "is_student":[1,1,1,1,1,1,0],"current_education":[7,6,7,None,7,7,7],
        "education_destination_status":["domestic"]*4+["abroad","unknown","domestic"],
        "home_zone_id":10,"home_x":0.,"home_y":0.,"commute_home_distance":2000.})
    locations = pd.DataFrame({"person_id":range(1,8),"destination_id":"unchanged",
        "x":1.,"y":2.,"education_assignment_status":"original",
        "education_secondary_target_distance_m":[2000.,20000.,2000.,np.nan,2000.,2000.,2000.]})
    sites = pd.DataFrame({"enterprise_id":["short_gym","long_gym","short_voc","long_voc"],
        "noga":["853102","853102","853200","853200"],
        "x":[2000.,20000.,2000.,20000.],"y":0.,"number_employees":10.,
        "zone_id":[30,70,30,70],"zone_municipality_id":[20,70,20,70]})
    zones = pd.DataFrame({"zone_id":[10,20,30,70],
        "zone_level":["municipality","municipality","quarter","municipality"]})
    pdf = np.array([[0.,0.,.1,.9]]*4)
    before = locations.copy(deep=True)
    result = assign_secondary_od_locations(people,locations,sites,zones,pdf,0).set_index("person_id")
    assert result.loc[1,"destination_id"] == "short_gym"
    assert result.loc[1,"education_secondary_destination_zone_id"] == 30
    assert result.loc[2,"destination_id"] == "long_voc"
    assert result.loc[2,"education_secondary_destination_zone_id"] == 70
    assert result.loc[[1,2],"education_secondary_zone_distance_status"].eq("ss_age_programme_distance").all()
    assert result.loc[3:,"destination_id"].eq("unchanged").all()
    assert result.index.tolist() == list(range(1,8))
    pd.testing.assert_frame_equal(locations,before)
    repeat=assign_secondary_od_locations(people,locations,sites,zones,pdf,0).set_index("person_id")
    pd.testing.assert_frame_equal(result,repeat)
    # The MZ/reference distance remains 2 km, but without an SS target the
    # original mostly long-distance OD row must govern destination-zone choice.
    no_target=assign_secondary_od_locations(people,
        locations.drop(columns="education_secondary_target_distance_m"),sites,zones,pdf,0).set_index("person_id")
    assert no_target.loc[1,"destination_id"] == "long_gym"
    assert no_target.loc[1,"education_secondary_zone_distance_status"] == "unconditioned_missing_ss_target"
