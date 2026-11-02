"""Usual commutes must be independent of attendance on the diary day."""

import numpy as np
import pandas as pd
import pytest

from data.microcensus.persons import reported_education_location
from synthesis.population.matching.commute import donor_commutes, population_commutes, write_diagnostics


def test_reported_school_and_diary_fallback_do_not_change_attendance(tmp_path):
    persons = pd.DataFrame({"person_id":[1,2,3,4,5], "age":[20]*5,
        "employment_status":[2,3,2,1,2], "employed":[0,1,0,1,0],
        "person_weight":[1.,2.,1.,1.,1.], "commute_distance":[-1.,6000.,-1.,9000.,-1.],
        "education_commute_distance":[4000.,2000.,np.nan,50000.,np.nan]})
    diaries = pd.DataFrame({"person_id":[2,3], "commute_home_distance":[1000.,3000.]})
    before = persons.copy(deep=True)
    result = donor_commutes(persons, diaries, seed=17).set_index("person_id")
    # Respondent 1 reports a school but does not attend today. Work is longer
    # than school for respondent 2; respondent 3 needs the diary-place fallback.
    assert result.loc[[1,2,3,4],"commute_distance"].tolist() == [4000.,6000.,3000.,9000.]
    assert result.loc[4,"education_distance_source"] == "not_student"
    assert result.loc[5,"education_distance_source"] == "imputed_mz_age_employment"
    assert result.loc[5,"commute_distance_education"] in [3000.,4000.]
    assert result.index[result.education_diary_attendance].tolist() == [2,3]
    pd.testing.assert_frame_equal(persons, before)
    shuffled = donor_commutes(persons.sample(frac=1.,random_state=1), diaries,seed=17).set_index("person_id")
    pd.testing.assert_frame_equal(result.sort_index(),shuffled.sort_index())
    write_diagnostics(result.reset_index(), result.reset_index(),tmp_path)
    diagnostics = pd.read_csv(tmp_path/"mz_matching_commute_diagnostics.csv")
    assert diagnostics.education_visitors.sum() == 2


def test_unknown_is_not_zero_and_every_student_requires_an_anchor():
    people = pd.DataFrame({"person_id":[1,2,3],"age":[20]*3,"employed":[0,1,0],"is_student":[1,1,0]})
    work = pd.DataFrame({"person_id":[2],"work_location_type":["remote"],"commute_distance":[0.]})
    education = pd.DataFrame({"person_id":[1,2],"commute_distance":[10000.,2000.]})
    result = population_commutes(people,work,education)
    assert result.commute_distance.tolist() == [10000.,2000.,-1.]
    assert result.loc[1,"commute_distance_work"] == 0.
    with pytest.raises(ValueError,match="no education distance"):
        population_commutes(people,work,education.iloc[:1])


def test_school_coordinates_use_wgs84_and_missing_codes_stay_missing():
    raw = pd.DataFrame({"AU_X":[6.566,-99,0,6.566],"AU_Y":[46.519,-99,0,46.519],
                        "AU_LND":[8100,8100,8100,8100],"AU_QAL":[1,1,1,-99]})
    result = reported_education_location(raw)
    assert 2500000 < result.loc[0,"education_x"] < 2600000
    assert 1100000 < result.loc[0,"education_y"] < 1200000
    assert result.loc[1:,["education_x","education_y"]].isna().all().all()


def test_imputation_never_uses_other_imputations_as_reference():
    people = pd.DataFrame({"person_id":range(1,6), "age":[20,20,20,50,80],
        "employment_status":[2]*5,"employed":[0,0,1,0,1],"person_weight":1.,
        "commute_distance":-1.,"education_commute_distance":[1234.,np.nan,np.nan,np.nan,np.nan]})
    diary = pd.DataFrame({"person_id":pd.Series(dtype=int),"commute_home_distance":pd.Series(dtype=float)})
    result = donor_commutes(people,diary)
    assert result.commute_distance.eq(1234.).all()
    assert result.education_distance_source.tolist() == ["reported_education_place",
        "imputed_mz_age_employment","imputed_mz_age","imputed_mz_all_students","imputed_mz_all_students"]
    assert not result.education_diary_attendance.any()


def test_overseas_schools_are_preserved_but_not_drawn_as_domestic_peers():
    people = pd.DataFrame({"person_id":[1,2,3],"age":[20]*3,"employment_status":[2]*3,
        "employed":0,"person_weight":1.,"commute_distance":-1.,
        "education_commute_distance":[2000.,8000000.,np.nan],
        "education_country_code":[8100,8408,-99]})
    diary = pd.DataFrame({"person_id":pd.Series(dtype=int),"commute_home_distance":pd.Series(dtype=float)})
    result = donor_commutes(people,diary)
    assert result.commute_distance_education.tolist() == [2000.,8000000.,2000.]
