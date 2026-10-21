"""Configured seeds control imputation and stochastic RDA decisions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_income_imputation_uses_configured_seed_for_model_and_draws(monkeypatch):
    from data.microcensus import income
    from sklearn.base import BaseEstimator, ClassifierMixin
    seeds = []
    class Classifier(ClassifierMixin, BaseEstimator):
        def __init__(self, **kwargs):
            seeds.append(kwargs['random_seed'])
        def fit(self, frame, labels, **kwargs):
            self.classes_ = np.array([0, 1])
            return self
        def predict_proba(self, frame):
            return np.tile([.5, .5], (len(frame), 1))
    monkeypatch.setattr(income, 'CatBoostClassifier', Classifier)
    columns = ['household_size', 'number_of_cars', 'number_of_bikes', 'N_adults',
               'N_children_under_18', 'N_children_under_6', 'is_swiss',
               'population_density', 'ovgk', 'municipality_type', 'canton_id',
               'employment_status']
    frame = pd.DataFrame({column: [1] * 66 for column in columns})
    frame['income_class'] = [0, 1] + [-1] * 64
    first = income.impute(frame.copy(), random_seed=17)
    np.random.seed(999)
    repeated = income.impute(frame.copy(), random_seed=17)
    changed = income.impute(frame.copy(), random_seed=18)
    pd.testing.assert_frame_equal(first, repeated)
    assert not first.income_class.equals(changed.income_class)
    assert seeds == [17, 17, 18]


def test_rda_detour_decisions_use_the_supplied_random_stream():
    from synthesis.population.spatial.secondary.rda import DiscretizationErrorObjective
    detour = SimpleNamespace(get_detour_factor=lambda *args: np.array([2.1]))
    problem = {'origin': None, 'destination': None}
    coordinates = np.array([[0., 0.], [1000., 0.]])
    def decisions(seed):
        objective = DiscretizationErrorObjective(
            {'detour_factor': 2.0}, np.random.RandomState(seed), detour)
        return [objective.check_detour_factor(problem, coordinates)[0] for _ in range(100)]
    first = decisions(17)
    np.random.seed(999)
    assert decisions(17) == first
    assert decisions(18) != first
