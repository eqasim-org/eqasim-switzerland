"""Protect annual denominators, pooled row identity and student-stock scope."""
import unittest

import pandas as pd

from analysis.education_student_stock import population_stock, survey_stock, compare_degree_totals, imputation_audit


class EducationStudentStockTest(unittest.TestCase):
    def survey(self):
        # Deliberately repeated indices, as in raw.execute's concat of years.
        return pd.DataFrame({'age':[22,27,22,27], 'survey_year':[2022,2022,2023,2023],
            'weight':[100.,20.,120.,30.], 'is_student':[1,1,1,0], 'in_education':[1,1,1,0],
            'current_education':[11,13,11,13], 'employed':[3,1,3,1],
            'education_country_code':[8100]*4, 'education_municipality':[1]*4},index=[0,1,0,1])

    def test_raw_year_indices_do_not_duplicate_records_or_hide_employed_doctorate(self):
        parsed, table=survey_stock(self.survey())
        self.assertEqual(len(parsed),4)
        self.assertEqual(table.observations.sum(),4)
        self.assertEqual(table.weighted_persons.sum(),270.)
        self.assertEqual(table.loc[table.study_level.eq('doctorate_habilitation')&table.is_student.eq(0),'observations'].sum(),1)

    def test_domestic_status_preserves_unknown_and_explicit_foreign_country(self):
        survey=self.survey().reset_index(drop=True)
        survey['education_country_code']=[8100,250,-9,-9]
        survey['education_municipality']=[1,1,-9,1]
        parsed,_=survey_stock(survey)
        self.assertEqual(parsed.education_destination_status.tolist(),['domestic','abroad','unknown','domestic'])

    def test_stock_includes_nonattendance_and_separates_recovered_residents(self):
        persons=pd.DataFrame({'person_id':[1,2,3,4], 'age':[22,23,24,80],
            'study_level':['bachelor','bachelor','master','none'],
            'education_destination_status':['domestic','domestic','abroad','not_applicable'],
            'employed':[3,3,1,3], 'is_synthetic_household':[False,True,False,True],
            'non_private_dorm_candidate_households':[False,True,False,False],
            'non_private_home_only_households':[False,False,False,True]})
        activities=pd.DataFrame({'person_id':[2,2,3], 'purpose':['education','education','home']})
        _,stock=population_stock(persons,activities,.5)
        bachelor=stock.loc[stock.study_level.eq('bachelor')]
        self.assertEqual(bachelor.agents.sum(),2)
        self.assertEqual(bachelor.daily_agents.sum(),1)
        self.assertEqual(bachelor.estimated_persons.sum(),4)
        self.assertEqual(set(bachelor.population_cohort),{'private','recovered_dorm_candidate'})

    def test_annual_pooled_comparison_averages_years_and_excludes_nonstudent_report(self):
        _,table=survey_stock(self.survey())
        stock=pd.DataFrame({'study_level':['bachelor'], 'education_destination_status':['domestic'],
            'population_cohort':['private'], 'estimated_persons':[125.], 'estimated_daily_visitors':[50.]})
        reference=pd.DataFrame({'academic_year':['2023/24']*3,'institution_type':['UH']*3,
            'bfs_institution_code':[1]*3,'study_level':['bachelor','master','doctorate_habilitation'],
            'students':[130.,40.,35.]})
        comparison=compare_degree_totals(stock,table,reference,'2023/24',2023).set_index('study_level')
        self.assertEqual(comparison.loc['bachelor','survey_domestic_average_year'],110.)
        self.assertEqual(comparison.loc['bachelor','survey_domestic_reference_year'],120.)
        self.assertEqual(comparison.loc['doctorate_habilitation','survey_domestic_average_year'],10.)
        self.assertEqual(comparison.loc['doctorate_habilitation','survey_domestic_reference_year'],0.)

    def test_codebook_bridge_adds_diplomas_to_corresponding_level_only(self):
        _,table=survey_stock(self.survey())
        stock=pd.DataFrame({'study_level':['bachelor'], 'education_destination_status':['domestic'],
            'population_cohort':['private'], 'estimated_persons':[125.], 'estimated_daily_visitors':[50.]})
        reference=pd.DataFrame({'academic_year':['2023/24']*6,
            'institution_type':['UH','UH','UH','FH_PH','UH','UH'],
            'bfs_institution_code':[1]*6,
            'study_level':['bachelor','master','doctorate_habilitation','legacy_degree','legacy_degree','continuing_education_and_other'],
            'students':[130.,40.,35.,5.,7.,100.]})
        result=compare_degree_totals(stock,table,reference,'2023/24',2023).set_index('study_level')
        self.assertEqual(result.loc['bachelor','bfs_enrolment'],130.)
        self.assertEqual(result.loc['bachelor','bfs_codebook_aligned_enrolment'],135.)
        self.assertEqual(result.loc['master','bfs_codebook_aligned_enrolment'],47.)
        self.assertEqual(result.loc['doctorate_habilitation','bfs_codebook_aligned_enrolment'],35.)

    def test_unresolved_imputation_outcome_retains_saved_student(self):
        survey=pd.DataFrame({'age':[22,22], 'sex':[0,0], 'canton_id':[1,1], 'employed':[3,3],
            'job_position':70,
            'weight':[1.,1.], 'is_student':[1,1], 'current_education':[1,1],
            'survey_year':[2022,2023], 'education_country_code':[8100,8100], 'education_municipality':[1,1]})
        persons=pd.DataFrame({'age':[22], 'sex':[0], 'canton_id':[1], 'employed':[3],
            'job_position':70,
            'is_student':[1], 'current_education':pd.Series([pd.NA],dtype='Int64'),
            'education_destination_status':['domestic'], 'population_cohort':['private']})
        result=imputation_audit(persons,survey,.5,2023,20.)
        outcome=result.loc[result.scenario.eq('pooled_current')&result.current_education.eq(0)
                           &result.education_destination_status.eq('domestic')]
        self.assertEqual(outcome.actual_agents.iloc[0],1)
        self.assertEqual(outcome.expected_agents.iloc[0],1.)


if __name__=='__main__':unittest.main()
