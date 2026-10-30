"""Descriptive dorm evidence, ordinary prediction and home-only invariants."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd

from data.statpop.households import classify_non_private_residences
from synthesis.population.models.non_private import execute_without_home_only, dorm_candidate_mask
from synthesis.population.models.education import assign_education
from synthesis.population.spatial.primary.education.balancing import build_balancing
from synthesis.population.spatial.primary.education.tertiary import assign_od_locations
from synthesis.population.activities import execute as build_activities
from matsim.scenario.population import PersonWriter


class NonPrivateResidentsTest(unittest.TestCase):
    def test_age_distance_and_private_boundaries(self):
        people = pd.DataFrame({"person_id":range(6), "age":[16,17,18,38,39,25],
            "home_x":[1000,1250,1250.1,1000,1000,1000], "home_y":[1000]*6,
            "is_synthetic_household":[True]*5+[False]})
        sites = pd.DataFrame({"EGID":[1],"GKODE":[1000.],"GKODN":[1000.],"evidence":["GWR_class_1130"]})
        assigned=classify_non_private_residences(people,sites)
        self.assertEqual(assigned.non_private_dorm_candidate_households.tolist(),[True,True,False,True,False,False])
        self.assertEqual(assigned.non_private_home_only_households.tolist(),[False,False,False,False,True,False])
        self.assertNotIn("non_private_education_stage", assigned)
        self.assertFalse(people.columns.str.startswith('non_private_').any())

    def test_home_only_never_enters_prediction_or_validation(self):
        source=pd.DataFrame({"person_id":[7,8,9],"age":[20,80,50],
            "non_private_home_only_households":[False,True,False]})
        class Context:
            def stage(self,name):return source
        seen=[]
        def model(context):
            frame=context.stage('population').copy()
            seen.extend(frame.person_id)
            frame['predicted']=1
            return frame.iloc[::-1]
        result=execute_without_home_only(Context(),'population',model,{'predicted':0})
        self.assertEqual(seen,[7,9])
        self.assertEqual(result.person_id.tolist(),[7,8,9])
        self.assertEqual(result.predicted.tolist(),[1,0,1])
        self.assertEqual(result.loc[1,'non_private_model_status'],'home_only_not_modelled')
        self.assertNotIn('predicted',source)
        self.assertEqual(dorm_candidate_mask(pd.DataFrame({'non_private_dorm_candidate_households':['False','True']})).tolist(),[False,True])

    def test_accommodation_flag_does_not_change_education_predictions(self):
        # The same demographics and seed must produce identical study/country
        # draws, whether a person is near a dorm candidate or in a private home.
        survey = pd.DataFrame({'age':[17]*4+[22]*4, 'employed':[3]*8,
            'job_position':70,
            'sex':[0]*8, 'canton_id':[1]*8, 'is_student':[1]*8,
            'weight':[1.]*8, 'current_education':[5,11,12,13]*2,
            'education_country_code':[8100,250,8100,8100]*2,
            'education_municipality':[1]*8})
        persons = pd.DataFrame({'age':[17,22]*20, 'employed':[3]*40,
            'job_position':70,
            'sex':[0]*40, 'canton_id':[1]*40, 'is_student':[1]*40,
            'non_private_dorm_candidate_households':[True]*40})
        private = persons.assign(non_private_dorm_candidate_households=False)
        flagged = assign_education(persons, survey, seed=0)
        ordinary = assign_education(private, survey, seed=0)
        pd.testing.assert_frame_equal(flagged.drop(columns='non_private_dorm_candidate_households'),
                                      ordinary.drop(columns='non_private_dorm_candidate_households'))
        self.assertTrue(flagged.is_student.eq(1).all())
        self.assertTrue(flagged.education_destination_status.eq('abroad').any())

    def test_od_zone_ids_accept_spatial_object_and_nullable_inputs(self):
        # Spatial zone imputation produces object IDs; nullable integer inputs
        # also produce object arrays when missing values are present. Neither
        # may require coercing identifiers into float columns during assignment.
        for dtype in ['object', 'Int64']:
            with self.subTest(dtype=dtype):
                persons = pd.DataFrame({'person_id':[1,2,3,4], 'age':[20]*4,
                    'is_student':[1]*4, 'current_education':[11]*4,
                    'education_destination_status':['domestic']*4,
                    'home_zone_id':pd.Series([1,pd.NA,1,1], dtype=dtype),
                    'home_x':[1000.]*4, 'home_y':[1000.]*4,
                    'commute_home_distance':[1000.]*4,
                    'non_private_dorm_candidate_households':[True,True,False,False]})
                locations = pd.DataFrame({'person_id':[1,2,3,4], 'destination_id':['old']*4,
                    'x':[0.]*4, 'y':[0.]*4, 'education_assignment_status':['old']*4},
                    index=[10,30,50,90])
                statent = pd.DataFrame({'enterprise_id':['a'], 'x':[2000.], 'y':[1000.],
                    'zone_id':[2], 'zone_municipality_id':[2],
                    'number_employees':[10.], 'number_employees_fte':[10.]})
                compatibility = pd.DataFrame({'enterprise_id':['a'], 'campus_id':['A'],
                    'study_level':['tertiary']})
                zones = pd.DataFrame({'zone_id':[1,2], 'zone_level':['municipality']*2})
                result = assign_od_locations(persons, locations, statent, compatibility,
                    zones, np.array([[0.,1.],[0.,1.]]), [1,2,3], 0,
                    {'models':{}})
                self.assertEqual(result.loc[10,'education_od_origin_zone_id'], 1)
                self.assertTrue(pd.isna(result.loc[30,'education_od_origin_zone_id']))
                self.assertEqual(result.loc[50,'education_od_origin_zone_id'], 1)
                self.assertTrue(result.loc[[10,50],'education_od_destination_zone_id'].eq(2).all())
                self.assertTrue(pd.isna(result.loc[30,'education_od_destination_zone_id']))
                self.assertEqual(result.loc[10,'education_od_status'], 'pooled_tertiary_od')
                self.assertEqual(result.loc[30,'education_od_status'], 'compatible_location_proxy_missing_origin')
                self.assertEqual(result.loc[50,'education_od_status'], 'pooled_tertiary_od')
                self.assertEqual(result.loc[90,'destination_id'], 'old')
                self.assertTrue(pd.isna(result.loc[90,'education_od_destination_zone_id']))

    def test_dorm_candidates_share_the_ordinary_od_and_campus_margins(self):
        persons = pd.DataFrame({'person_id':[1,2,3], 'age':20, 'is_student':1,
            'current_education':11, 'education_destination_status':'domestic',
            'home_zone_id':1, 'home_x':1000., 'home_y':1000.,
            'commute_home_distance':1000.,
            'non_private_dorm_candidate_households':[True,True,False]})
        sites = pd.DataFrame({'enterprise_id':['a','b'], 'x':[1000.,2000.],
            'y':1000., 'zone_id':[2,3], 'zone_municipality_id':[2,3],
            'number_employees':10., 'number_employees_fte':10.})
        compatibility = pd.DataFrame({'enterprise_id':['a','b'], 'campus_id':['A','B'],
            'study_level':'tertiary'})
        registry = {'establishments':compatibility.assign(institution_type='UH')}
        zones = pd.DataFrame({'zone_id':[1,2,3], 'zone_level':'municipality'})
        pdf = np.array([[0.,.5,.5]] * 3)
        targets = {'institution_type_targets':pd.DataFrame({'study_level':['tertiary'],
            'institution_type':['UH'], 'enrolment_share':[1.]}),
            'campus_targets':pd.DataFrame({'study_level':'tertiary', 'campus_id':['A','B'],
            'institution_type':'UH', 'institution_enrolled_students':[1.,2.],
            'bfs_level_enrolment':3., 'campus_share':1.})}
        flagged = build_balancing(persons, [1,2,3], sites, registry, compatibility, zones, pdf, targets)
        private = persons.assign(non_private_dorm_candidate_households=False)
        ordinary = build_balancing(private, [1,2,3], sites, registry, compatibility, zones, pdf, targets)
        pd.testing.assert_frame_equal(flagged['diagnostics'], ordinary['diagnostics'])
        self.assertTrue(flagged['diagnostics'].converged.all())
        self.assertEqual(flagged['models']['tertiary']['integer_flows'].sum(), 3)
        locations = pd.DataFrame({'person_id':[1,2,3], 'destination_id':'old',
            'x':0., 'y':0., 'education_assignment_status':'old'})
        placed = assign_od_locations(persons, locations, sites, compatibility, zones, pdf, [1,2,3], 0, flagged)
        unflagged = assign_od_locations(private, locations, sites, compatibility, zones, pdf, [1,2,3], 0, ordinary)
        pd.testing.assert_frame_equal(placed, unflagged)
        self.assertEqual(placed.destination_id.value_counts().to_dict(), {'b':2, 'a':1})
        self.assertTrue(placed.education_od_status.eq('balanced_tertiary_od').all())

    def test_home_only_activity_without_any_donor_trips(self):
        persons=pd.DataFrame({'person_id':[10,11], 'non_private_home_only_households':[True,False],
                             'is_truck_driver':[False,False],'is_outside_of_switzerland':[False,False]})
        trips=pd.DataFrame(columns=['person_id','trip_index','departure_time','arrival_time','preceding_purpose','following_purpose','mode'])
        class Context:
            def stage(self,name):return persons if name=='synthesis.population.enriched' else trips
        activities=build_activities(Context())
        home=activities.loc[activities.person_id.eq(10)]
        self.assertEqual(len(home),1)
        self.assertEqual(home.purpose.iloc[0],'home')
        self.assertEqual(home.duration.iloc[0],30*3600)

    def test_home_only_export_attributes_and_car_availability(self):
        # The unmodelled vehicle placeholder must be accepted by MATSim's binary
        # car-availability mapping, and explicit scenario flags must survive XML.
        attributes = {}
        class Writer:
            def add_attribute(self, name, datatype, value):
                attributes[name] = (value, datatype)
            def true_false(self, value): return 'true' if value else 'false'
            def yes_no(self, value): return 'yes' if value else 'no'
            def location(self, *args): return {}
            def __getattr__(self, name): return lambda *args, **kwargs: None
        person = SimpleNamespace(person_id=8, age=80, employed=0,
            driving_license=0, sex=0, home_x=2500000, home_y=1100000,
            car_availability=0, pt_subscription=0, bike_availability=0,
            non_private_dorm_candidate_households=False,
            non_private_home_only_households=True, is_synthetic_household=True,
            non_private_model_status='home_only_not_modelled')
        PersonWriter(person).write(Writer())
        self.assertEqual(attributes['non_private_home_only_households'], ('true', 'java.lang.Boolean'))
        self.assertEqual(attributes['non_private_dorm_candidate_households'], ('false', 'java.lang.Boolean'))
        self.assertEqual(attributes['non_private_model_status'], ('home_only_not_modelled', 'java.lang.String'))
        self.assertEqual(attributes['carAvail'][0], 'never')


if __name__=='__main__':unittest.main()
