"""A usual education anchor must exist without an MZ donor or education activity."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data.statent.education_institutions import build_registry, pooled_compatibility
from synthesis.population.spatial.primary.education.balancing import build_balancing
from synthesis.population.spatial.primary.education.locations import execute
from synthesis.population.spatial.primary.education.secondary_distances import fit_distance_model


class EducationBeforeMatchingTest(unittest.TestCase):
    def test_every_student_has_an_anchor_before_diary_selection(self):
        people = pd.DataFrame({'person_id':range(1,8), 'age':[5,10,16,20,30,35,25],
            'is_student':[1]*6+[0], 'home_zone_id':1, 'home_x':2500000., 'home_y':1100000.,
            'current_education':pd.array([None,None,6,None,None,None,1],dtype='Int64'),
            'study_level':['below_survey_age_coverage']*2+['vocational_secondary']+['tertiary']*3+['none'],
            'education_destination_status':['unknown']*2+['domestic']*3+['abroad','not_applicable']})
        sites = pd.DataFrame({'enterprise_id':['k','p','s','u','f','h'],
            'statent_enterprise_id':['K','P','S','U','F','H'], 'statent_local_id':['1','2','3','4','5','6'],
            'noga':['851000','852001','853200','854201','854202','854203'],
            'enterprise_noga':['851000','852001','853200','854201','854202','854203'],
            'x':[2500100.,2500200.,2501000.,2502000.,2503000.,2501200.], 'y':1100000.,
            'zone_id':[1,1,2,3,3,2], 'zone_municipality_id':[1,1,2,3,3,2],
            'number_employees':10., 'number_employees_fte':10., 'statent_year':2023,
            'education_type':['kindergarten','primary','secondary','tertiary','tertiary','tertiary']})
        registry = build_registry(sites)
        rules = pd.DataFrame(columns=['statent_local_id','study_level','regular_eligible'])
        compatible = pooled_compatibility(registry, rules)
        zones = pd.DataFrame({'zone_id':[1,2,3], 'zone_level':'municipality'})
        pdf = np.array([[0.,.5,.5]]*3)
        targets = {'institution_type_targets':pd.DataFrame({'study_level':'tertiary',
            'institution_type':['UH','FH_PH','higher_vocational'], 'enrolment_share':[.5,.25,.25]}),
            'campus_targets':pd.DataFrame({'study_level':['tertiary'],
            'campus_id':[registry['establishments'].set_index('enterprise_id').loc['u','campus_id']],
            'institution_type':['UH'], 'institution_enrolled_students':[100.],
            'bfs_level_enrolment':[200.], 'campus_share':[1.]})}
        balancing = build_balancing(people, None, sites, registry, compatible, zones, pdf, targets)
        mz = pd.DataFrame({'person_id':[101,102,103], 'age':[10,16,22], 'workday':True,
            'employment_status':2, 'person_weight':1.})
        commute = {'education':pd.DataFrame({'person_id':[101,102,103],
                                            'commute_home_distance':[200.,1000.,2000.]})}
        survey = pd.DataFrame({'age':[16], 'current_education':[6], 'is_student':[1],
            'education_departure_status':['domestic'], 'education_zone_id':[2],
            'education_zone_level':['municipality'], 'education_country_code':[8100],
            'crowfly_distance_to_school':[1.], 'weight':[1.]})
        model = fit_distance_model(survey, minimum_effective_n=1.)
        class Constants:
            census = 'statpop'
        stages = {'synthesis.population.sampled':people, 'data.constants':Constants(),
            'data.statent.statent':sites, 'data.statent.education_institutions':registry,
            'data.microcensus.persons':mz, 'data.microcensus.commute':commute,
            'data.spatial.zones':zones, 'data.od.education_matrix':(pdf,pdf),
            'data.od.tertiary_education_matrix':(pdf,pdf),
            'synthesis.population.spatial.primary.education.secondary_distances':model,
            'synthesis.population.spatial.primary.education.balancing':balancing}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'eligibility.csv'
            rules.to_csv(path,index=False)
            settings = {'random_seed':0, 'education_site_eligibility':str(path), 'specific_day_scenario':'workday'}
            class Context:
                def stage(self,name):
                    # Enriched population and activities are intentionally absent:
                    # requesting either would recreate the dependency cycle.
                    return stages[name]
                def config(self,name): return settings[name]
                def progress(self,iterable,**kwargs): return iterable
            result = execute(Context()).set_index('person_id')
        self.assertEqual(set(result.index), set(range(1,7)))
        self.assertFalse(result.destination_id.isna().any())
        self.assertTrue(np.isfinite(result.commute_distance).all())
        self.assertEqual(result.loc[3,'education_secondary_od_status'], 'pooled_secondary_od')
        self.assertTrue(result.loc[[4,5],'education_od_status'].eq('balanced_tertiary_od').all())
        self.assertEqual(result.loc[6,'education_assignment_status'], 'legacy_abroad_proxy')


if __name__ == '__main__':
    unittest.main()
