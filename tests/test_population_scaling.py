"""Reference-date and household-scope regressions for population scaling."""
import unittest
import contextlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
from data.statpop.projections import population, households
from synthesis.population.scaled import _build_problem, process, execute


class PopulationScalingTest(unittest.TestCase):
    def population_extract(self):
        records = [(c, sex, nationality, f'{age} Jahre' if age < 100 else '100 Jahre oder mehr', 2025, 1)
                   for c in population.CANTON_TO_ID for sex in ['Mann', 'Frau']
                   for nationality in ['Schweiz', 'Ausland'] for age in range(101)]
        return pd.DataFrame(records, columns=['Kanton', 'Geschlecht', 'Staatsangehörigkeit (Kategorie)',
                                              'Alter', 'Jahr', population.STOCK_COLUMN])

    def test_year_end_ages_and_oldest_open_class_preserve_total(self):
        source = self.population_extract()
        # An additional January measure must never be used as the stock.
        source['Bevölkerungsstand am 1. Januar'] = 999
        result = population.prepare_controls(source, 2025, [6, 15, 18, 24, 45, 65, 80])
        totals = result.groupby('age_class').weight.sum()
        self.assertEqual(result.weight.sum(), len(source))
        self.assertEqual(totals.loc[0], 26 * 4 * 6)
        self.assertEqual(totals.loc[7], 26 * 4 * 21)
        self.assertEqual(len(result), 26 * 4 * 8)

    def test_january_only_and_missing_or_duplicate_cells_are_rejected(self):
        source = self.population_extract()
        with self.assertRaisesRegex(ValueError, 'year-end'):
            population.prepare_controls(source.rename(columns={population.STOCK_COLUMN:'Bevölkerungsstand am 1. Januar'}), 2025, [18])
        with self.assertRaisesRegex(ValueError, 'all 26'):
            population.prepare_controls(source.iloc[1:], 2025, [18])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            population.prepare_controls(pd.concat([source, source.iloc[:1]]), 2025, [18])
        with self.assertRaisesRegex(ValueError, 'not present'):
            population.prepare_controls(source, 2023, [18])

    def household_extract(self):
        return pd.DataFrame({'SCENARIO':'AM-00-2025','CANTON':[str(i) for i in range(1,27)],
            'SIZE_HOUSEHOLD':'_T','NUMBER_HOUSEHOLD_PERSON':'MEN','FREQ':'A',
            'TIME_HORIZ':'P','TIME_PERIOD':2025,'OBS_VALUE':100})

    def test_household_measure_vintage_and_total_selection(self):
        source = self.household_extract()
        source = pd.concat([source, source.assign(NUMBER_HOUSEHOLD_PERSON='PER',OBS_VALUE=1000),
            source.iloc[:1].assign(CANTON='_T',OBS_VALUE=2600),
            source.assign(SCENARIO='AM-00-2020',OBS_VALUE=1)],ignore_index=True)
        result = households.prepare_controls(source, 2025)
        self.assertEqual(result.weight.sum(),2600)
        self.assertEqual(len(result),26)
        with self.assertRaisesRegex(ValueError,'one private-household'):
            households.prepare_controls(self.household_extract().assign(SCENARIO='AM-00-2020'),2025)

    def problem(self, recovered=True):
        # Two private singletons need one copy each; one recovered person needs
        # two copies. The private-household margin must not suppress that person.
        persons=pd.DataFrame({'person_id':[1,2,3],'household_id':[1,2,3],
            'statpop_household_id':[101,102,103],'canton_id':1,'sex':0,'nationality':0,
            'age_class':[0,0,1],'is_synthetic_household':['False','False','True']})
        if not recovered:persons=persons.iloc[:2].drop(columns='is_synthetic_household')
        h=pd.DataFrame({'canton_id':[1],'weight':[2]})
        p=pd.DataFrame({'canton_id':1,'sex':0,'nationality':0,'age_class':[0,1],'weight':[2,2]})
        if not recovered:p=p.iloc[:1]
        return _build_problem(persons,h,p)

    def context(self):
        return SimpleNamespace(config=lambda key:42,progress=SimpleNamespace(update=lambda:None))

    def test_private_margin_excludes_recovered_but_integerization_retains_them(self):
        problem=self.problem()
        self.assertEqual(problem.group_controls[0][1].tolist(),[True,True,False])
        result,converged=process(self.context(),problem)
        self.assertTrue(converged)
        self.assertEqual(result.statpop_household_id.value_counts().to_dict(),{103:2,101:1,102:1})
        self.assertTrue(result.household_id.is_unique)

    def test_private_only_legacy_input_remains_supported(self):
        result,converged=process(self.context(),self.problem(False))
        self.assertTrue(converged)
        self.assertEqual(set(result.statpop_household_id),{101,102})

    def test_overlapping_group_controls_do_not_duplicate_households(self):
        problem=self.problem()
        problem.group_controls.append(problem.group_controls[0])
        result,converged=process(self.context(),problem)
        self.assertTrue(converged)
        self.assertEqual(len(result),4)

    def test_stage_ids_do_not_depend_on_worker_completion_order(self):
        source = self.problem().df.drop(columns='expansion_factor')
        second = source.assign(canton_id=2, person_id=source.person_id+3,
            household_id=source.household_id+3, statpop_household_id=source.statpop_household_id+3)
        source = pd.concat([source, second], ignore_index=True)
        h = pd.DataFrame({'canton_id':[1,2], 'weight':[2,2]})
        p = pd.DataFrame({'canton_id':[1,1,2,2], 'sex':0, 'nationality':0,
                          'age_class':[0,1,0,1], 'weight':2})
        class Progress:
            def __call__(self, items=None, **kwargs):
                return items if items is not None else contextlib.nullcontext()
            def update(self):
                pass
        class Context:
            progress = Progress()
            def __init__(self, reverse):
                self.reverse = reverse
            def config(self, key):
                return {'enable_scaling':True,'scaling_year':2025,'random_seed':42,'threads':2}[key]
            def stage(self, key):
                return {'synthesis.population.models.caravailability':source,
                    'data.statpop.projections.households':(h,2025),
                    'data.statpop.projections.population':(p,2025)}[key]
            def parallel(self, processes):
                return contextlib.nullcontext(self)
            def imap_unordered(self, function, problems):
                for problem in problems[::-1] if self.reverse else problems:
                    yield function(self, problem)
        first, second = execute(Context(False)), execute(Context(True))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first),8)
        self.assertEqual(first.loc[~first.is_synthetic_household,'household_id'].nunique(),4)
        self.assertEqual(first.is_synthetic_household.sum(),4)
        self.assertTrue(first.person_id.is_unique)


if __name__ == '__main__':
    unittest.main()
