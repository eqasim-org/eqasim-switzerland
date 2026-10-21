"""Purpose masks must match real facility pools at every hierarchy level."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import torch
from shapely.geometry import Point

from synthesis.population.spatial.secondary_nn.hierarchical_utils import SECONDARY_ACTIVITIES
from synthesis.population.spatial.secondary_nn.location_helpers import _prepare_destination_level2_index
from synthesis.population.spatial.secondary_nn.model_wrappers import HierarchicalLocationChoiceModel
from synthesis.population.spatial.secondary_nn.tools import destination_offer_masks
from synthesis.population.spatial.secondary_nn.choice_model import predict_choice_proba


def pools_and_model():
    destinations = gpd.GeoDataFrame(dict(
        destination_id=['work', 'park', 'work2', 'school'], number_employees=1,
        ovgk='B', offers_work=[True, False, True, False],
        offers_education=[False, False, False, True], offers_shop=False,
        offers_leisure=[False, True, False, False], offers_other=[False, True, False, False],
        offers_work_secondary=False, offers_education_secondary=False,
        offers_home_secondary=[True, False, False, False]),
        geometry=[Point(i, 0) for i in range(4)], crs=2056)
    levels = pd.DataFrame(dict(destination_id=destinations.destination_id,
        level_1=['a/one', 'a/one', 'a/two', 'b/one'],
        level_2=['a/one/work', 'a/one/park', 'a/two/work', 'b/one/school']))
    context = SimpleNamespace(stage=lambda _: destinations)
    pools = _prepare_destination_level2_index(context, {'destinations': levels}, None)
    model = HierarchicalLocationChoiceModel.__new__(HierarchicalLocationChoiceModel)
    (model.destination_index, model.destination_fallback, model.car_available_probabilities,
     model.no_car_available_probabilities, model.destination_index_xy,
     model.destination_fallback_xy) = pools
    model._purpose_categories = SECONDARY_ACTIVITIES
    model.wrappers = dict(
        coarse=SimpleNamespace(all_h3=['a', 'b']),
        medium=SimpleNamespace(level1_candidate_attributes_by_level0={
            'a': dict(children=['a/one', 'a/two'], x=np.zeros(3)),
            'b': dict(children=['b/one'], x=np.zeros(3))}),
        detailed=SimpleNamespace(level2_candidate_attributes_by_level1={
            ('a', 'a/one'): dict(children=['a/one/work', 'a/one/park'], x=np.zeros(3)),
            ('a', 'a/two'): dict(children=['a/two/work'], x=np.zeros(3)),
            ('b', 'b/one'): dict(children=['b/one/school'], x=np.zeros(3))}),
        short=SimpleNamespace(all_h3=levels.level_2.tolist()))
    tree = {'a': {'a/one': ['a/one/work', 'a/one/park'], 'a/two': ['a/two/work']},
            'b': {'b/one': ['b/one/school']}}
    model._build_destination_support_masks(tree)
    return model, destinations, levels


def test_work_and_education_ignore_the_secondary_offer_flags():
    model, _, _ = pools_and_model()
    assert set(model.destination_index['work']) == {'a/one/work', 'a/two/work'}
    assert set(model.destination_index['education']) == {'b/one/school'}
    assert model.destination_index['work'] is model.destination_index['work_secondary']
    assert 'a/one/park' not in model.destination_index['work_secondary']


def test_masks_propagate_actual_eligibility_and_exclude_padding():
    model, _, _ = pools_and_model()
    assert model.c.mask['other'].tolist() == [True, False]
    assert model.m.mask['a']['other'].tolist() == [True, False, False]
    assert model.d.mask[('a', 'a/one')]['other'].tolist() == [False, True, False]
    assert model.s.mask['work'].tolist() == [True, False, True, False]
    assert model.s.mask['home'].tolist() == [True, False, False, False]
    assert model.c.mask['work_secondary'].tolist() == model.c.mask['work'].tolist()


@pytest.mark.parametrize('purpose', SECONDARY_ACTIVITIES)
def test_supported_cells_always_draw_a_facility_of_the_requested_purpose(purpose):
    model, destinations, levels = pools_and_model()
    masks = destination_offer_masks(destinations)
    base = purpose.removesuffix('_secondary')
    eligible = set(destinations.loc[masks[base], 'destination_id'])
    for leaf in model.s.all_h3:
        if model.s.mask[base][model.s.all_h3.index(leaf)]:
            middle = levels.loc[levels.level_2 == leaf, 'level_1'].iloc[0]
            for _ in range(10):
                facility, _ = model.sample_company_in_l2(purpose, middle, leaf, np.random.default_rng(3))
                assert facility in eligible


def test_invalid_leaf_raises_instead_of_widening_or_switching_purpose():
    model, _, _ = pools_and_model()
    with pytest.raises(RuntimeError, match='mask/pool mismatch'):
        model.sample_company_in_l2('work_secondary', 'a/one', 'a/one/park', np.random.default_rng(3))


class Utilities:
    _device = torch.device('cpu')
    def __call__(self, *args):
        return torch.tensor([[0., 1e20, 1e20]])


def test_hard_masks_have_zero_probability_even_with_extreme_utilities():
    result = predict_choice_proba(Utilities(), [], [], [], [], [True, False, False])
    assert result.tolist() == [[1., 0., 0.]]


def test_empty_masks_raise_instead_of_producing_a_uniform_distribution():
    with pytest.raises(ValueError, match='no eligible candidates'):
        predict_choice_proba(Utilities(), [], [], [], [], [False, False, False])
