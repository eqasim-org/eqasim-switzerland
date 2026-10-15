"""Student-only departure geography must not alter labels or the nonstudent denominator."""

import unittest

import numpy as np
import pandas as pd

from data.structural_survey.student_training import prepare_student_training
from synthesis.population.models.education import assign_education


class StudentOriginTrainingTest(unittest.TestCase):
    def test_departure_fallback_and_foreign_geography_keep_every_respondent(self):
        survey = pd.DataFrame({'is_student':[1,1,1,0], 'weight':[2.,3.,4.,5.],
            'home_municipality_id':1, 'home_quarter_id':np.nan, 'home_zone_id':10,
            'home_zone_level':'municipality', 'district_id':100, 'canton_id':1,
            'municipality_type':pd.Categorical(['rural']*4),
            'education_departure_status':['domestic','unresolved','abroad','domestic'],
            'education_departure_municipality_id':[2,np.nan,np.nan,2],
            'education_departure_quarter_id':np.nan, 'education_departure_zone_id':[20,np.nan,np.nan,20],
            'education_departure_zone_level':['municipality',None,None,'municipality']}, index=[0,0,1,1])
        original = survey.copy(deep=True)
        lookup = pd.DataFrame({'municipality_id':[1,2], 'district_id':[100,200],
            'canton_id':[1,2], 'municipality_type':['rural','urban']})
        result = prepare_student_training(survey, lookup)
        pd.testing.assert_frame_equal(survey, original)
        self.assertEqual(result.is_student.tolist(), [1,1,1,0])
        self.assertEqual(result.weight.sum(), 14.)
        self.assertEqual(result.canton_id.tolist(), [2,1,-1,1])
        self.assertEqual(result.district_id.tolist(), [200,100,-1,100])
        self.assertEqual(result.municipality_type.tolist(), ['urban','rural','outside_switzerland','rural'])
        self.assertEqual(result.student_origin_source.tolist(),
                         ['school_departure','residence_fallback','abroad','residence'])

    def test_all_tertiary_programmes_collapse_before_imputation(self):
        survey = pd.DataFrame({'age':22, 'sex':0, 'employed':3, 'canton_id':1,
            'weight':1., 'is_student':1, 'current_education':[9,10,11,12,13],
            'education_country_code':8100, 'education_municipality':1})
        people = pd.DataFrame({'age':[22]*20, 'sex':0, 'employed':3, 'canton_id':1,
            'is_student':[1]*19+[0]})
        result = assign_education(people, survey, seed=0)
        self.assertTrue(result.loc[result.is_student.eq(1), 'study_level'].eq('tertiary').all())
        self.assertTrue(result.loc[result.is_student.eq(1), 'current_education'].isna().all())
        self.assertEqual(result.loc[19, 'study_level'], 'none')


if __name__ == '__main__':
    unittest.main()
