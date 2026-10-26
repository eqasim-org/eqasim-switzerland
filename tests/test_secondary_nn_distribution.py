"""Distribution calibration, information boundaries and seeded mixture routing."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

from synthesis.population.spatial.secondary_nn.distribution_scores import (
    discrete_crps, projected_energy_score, ensemble_score_components, mixture_score,
)
from synthesis.population.spatial.secondary_nn.chain_state import (
    CHAIN_STATE_FEATURES, chain_state_vector, survey_chain_states,
)
from synthesis.population.spatial.secondary_nn.model_wrappers import HierarchicalLocationChoiceModel
from synthesis.population.spatial.secondary_nn import short_trip_classifier as selector
from synthesis.population.spatial.secondary_nn import feature_encoding as encoding
from synthesis.population.spatial.secondary_nn.tools import OriginContext, MOBILITY_FEATURES


def test_exact_discrete_crps_matches_pairwise_formula_and_gradients():
    probabilities = torch.tensor([[.1, .3, .6]], requires_grad=True)
    values = torch.tensor([[20000., 1000., 5000.]])
    target = torch.tensor([2500.])
    expected = (probabilities * (values - target[:, None]).abs()).sum(1)
    expected -= .5 * (probabilities[:, :, None] * probabilities[:, None, :]
                       * (values[:, :, None] - values[:, None, :]).abs()).sum((1, 2))
    actual = discrete_crps(probabilities, values, target)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(probabilities.grad).all()


def test_fixed_scale_score_does_not_prefer_underprediction_to_equal_overshoot():
    p = torch.ones((1, 1))
    over = discrete_crps(p, torch.tensor([[20000.]]), torch.tensor([1000.]))
    under = discrete_crps(p, torch.tensor([[1000.]]), torch.tensor([20000.]))
    torch.testing.assert_close(over, under)
    assert over.item() == 19000


def test_expected_crps_preserves_occasional_long_trips():
    # Identical people make local trips 80% of the time and 20 km trips 20%.
    candidates = torch.tensor([[0., 20000.]])
    scores = []
    for local_share in np.linspace(0, 1, 11):
        p = torch.tensor([[local_share, 1-local_share]], dtype=torch.float32)
        score = .8 * discrete_crps(p, candidates, torch.tensor([0.]))
        score += .2 * discrete_crps(p, candidates, torch.tensor([20000.]))
        scores.append(score.item())
    assert np.argmin(scores) == 8
    assert scores[8] < scores[-1]


def test_projected_energy_tracks_direction_and_keeps_gradients():
    p = torch.tensor([[.5, .5]], requires_grad=True)
    xy = torch.tensor([[[1000., 0.], [-1000., 0.]]])
    target = torch.tensor([[1000., 0.]])
    value = projected_energy_score(p, xy, target, projections=32)
    assert value.item() == pytest.approx(500., rel=.01)
    value.sum().backward()
    assert torch.isfinite(p.grad).all()


def test_remaining_chain_features_use_known_anchors_and_no_future_secondary_coordinates():
    trips = pd.DataFrame(dict(person_id=[1]*3, purpose=['leisure', 'shop', 'home'],
        origin_x=[5000., 6000., 7000.], origin_y=[0.]*3,
        destination_x=[6000., 7000., 0.], destination_y=[0.]*3,
        daily_crowfly_total=[20000.]*3, crowfly_consumed_before_trip=[1000., 2000., 3000.]))
    observed = survey_chain_states(trips)
    changed = trips.copy()
    changed.loc[:1, 'destination_x'] = 1000000.
    for a, b in zip(observed, survey_chain_states(changed)):
        np.testing.assert_array_equal(a, b)
    synthetic = chain_state_vector(20000, 1000, ['leisure', 'shop', 'home'], 5000, 0,
                                    primary_locations={'home': (0, 0)})
    np.testing.assert_array_equal(observed[0], synthetic)
    state = dict(zip(CHAIN_STATE_FEATURES, synthetic))
    assert state['remaining_trip_count'] == 3
    assert state['remaining_distance_budget_m'] == 19000
    assert state['next_anchor_distance_m'] == 5000
    assert state['trips_to_next_anchor'] == 3


def test_seeded_routing_samples_probabilities_and_ignores_target_distance():
    class Gate:
        def predict_probability(self, **features):
            assert 'target_distance' not in features
            return .8
    wrapper = SimpleNamespace(use_target_distance=False, s=object(), short_trip_classifier=Gate())
    first, second = np.random.default_rng(12), np.random.default_rng(12)
    a = [HierarchicalLocationChoiceModel._select_short_model(wrapper, target_distance=100., rng=first) for _ in range(2000)]
    b = [HierarchicalLocationChoiceModel._select_short_model(wrapper, target_distance=50000., rng=second) for _ in range(2000)]
    assert a == b
    assert np.mean(a) == pytest.approx(.8, abs=.03)
    assert not all(a)


def mixture_fixture(n=250):
    frame = pd.DataFrame(dict(person_id=np.arange(n), trip_id=0, age=30., sex=0., employed=1.,
        car_availability=1., income_class=3., daily_longest_distance_from_home=30000.,
        daily_crowfly_total=50000., daily_longest_distance_from_work=0.,
        crowfly_consumed_before_trip=0., trip_position_class=0., departure_time_normalized=.5,
        activity_duration_h=2., target_distance=999., purpose='leisure', origin_purpose='home',
        origin_x=0., origin_y=0., home_x=0., home_y=0., work_x=0., work_y=0., has_work=False,
        destination_x=np.where(np.arange(n)%5==0, 20000., 0.), destination_y=0., person_weight=1.))
    frame['partition'] = np.repeat(['train','stopping','validation'], [150,50,50])
    frame['fold'] = np.where(frame.partition.eq('validation'), -1, np.arange(n)%3)
    frame['chain_state'] = [np.zeros(len(CHAIN_STATE_FEATURES))]*n
    frame['activity_chain'] = [np.ones(6)]*n
    frame[MOBILITY_FEATURES] = 0.
    rows = []
    for row in frame.itertuples():
        stats = ensemble_score_components([[0.,0.]]*10, [[20000.,0.]]*10, [0.,0.], [row.destination_x,0.])
        stats.update(person_id=row.person_id, trip_id=0, short_leaf_probability=float(row.destination_x==0),
                     hierarchy_leaf_probability=float(row.destination_x>0), prediction_fold=row.fold,
                     short_mean_distance_m=0., hierarchy_mean_distance_m=20000.,
                     short_long_share=0., hierarchy_long_share=0.)
        rows.append(stats)
    import geopandas as gpd
    from shapely.geometry import box
    cells = gpd.GeoDataFrame({key:[1.] for key in encoding.STATIC_CANDIDATE_FEATURES}, geometry=[box(0,0,100,100)], crs=2056)
    cells['centroid'] = cells.geometry.representative_point()
    return frame, pd.DataFrame(rows), cells


def test_fitted_mixture_recovers_long_trip_frequency_without_distance_input():
    torch.set_num_threads(2)
    frame, comparisons, cells = mixture_fixture()
    fitted, audit = selector.train_selector(frame, comparisons, OriginContext(cells), seed=5,
                                            max_iter=100, patience=15)
    assert fitted.metrics['validation_short_expert_probability'] == pytest.approx(.8, abs=.05)
    assert fitted.metrics['validation_mean_distance_ratio'] == pytest.approx(1., abs=.25)
    assert audit.short_probability.between(0, 1).all()
    assert 'target_distance' not in fitted.feature_names


def test_outer_validation_targets_cannot_change_the_fitted_gate():
    torch.set_num_threads(2)
    frame, comparisons, cells = mixture_fixture()
    first, _ = selector.train_selector(frame, comparisons, OriginContext(cells), seed=6, max_iter=8, min_epochs=0)
    other = comparisons.copy()
    validation = frame.partition.eq('validation')
    other.loc[validation, 'short_leaf_probability'] = .999
    other.loc[validation, 'hierarchy_leaf_probability'] = .001
    other.loc[validation, 'energy_short_target'] = 999.
    second, _ = selector.train_selector(frame, other, OriginContext(cells), seed=6, max_iter=8, min_epochs=0)
    for a,b in zip(first.network.parameters(), second.network.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_observed_leaf_in_an_unsupported_branch_has_zero_likelihood(monkeypatch):
    from synthesis.population.spatial.secondary_nn import model_wrappers as wrappers
    identity=SimpleNamespace(transform=lambda x:x)
    c=SimpleNamespace(all_h3=['blocked','valid'],scaled_static_features_torch=torch.zeros(1,2,1),
        _build_candidate_dynamic=lambda **kwargs:np.zeros((2,4)),candidate_dynamic_scaler=identity,
        mask={'leisure':torch.tensor([False,True])},model=object())
    s=SimpleNamespace(predict_level2_short=lambda *args,**kwargs:(['valid/mid/leaf'],np.array([1.])))
    wrapper=SimpleNamespace(c=c,m=object(),d=object(),s=s,
        _get_cached_person_static=lambda *args:np.zeros((1,encoding.N_PERSON_STATIC)),
        _build_person_dynamic=lambda *args,**kwargs:np.zeros((1,1)),
        detour_factor=SimpleNamespace(get_cell_indices=lambda *args:np.array([0])),
        _coarse_detour_indices=np.array([0,0]),_add_cached_detour_feature=lambda x,*args:x,
        short_trip_threshold_m=1500.,short_query_radius_m=200.,short_min_candidates=6,
        short_destination_support_masks={'leisure':np.array([True])})
    monkeypatch.setattr(wrappers,'predict_choice_proba',lambda *args:torch.tensor([[0.,1.]]))
    inputs={key:0. for key in ['person_id','age','income_class','daily_longest_distance_from_home',
        'daily_crowfly_total','daily_longest_distance_from_work','sex','employed','car_availability',
        'crowfly_consumed_before_trip','trip_position_class','departure_time_normalized','activity_duration_h',
        'home_x','home_y','work_x','work_y','origin_x','origin_y']}
    inputs.update(activity_chain_vector=np.zeros(6),mobility_vector=np.zeros(len(MOBILITY_FEATURES)),
                  chain_state_vector=np.zeros(len(CHAIN_STATE_FEATURES)),has_work=False,purpose='leisure',origin_purpose='home')
    short,hierarchy=HierarchicalLocationChoiceModel.expert_leaf_probabilities(wrapper,'blocked/mid/leaf',**inputs)
    assert short==hierarchy==0.


@pytest.mark.parametrize("use_distance", [True, False])
def test_remaining_plan_and_distance_features_match_batch_and_online_encoding(use_distance):
    # Vary both trip distances and chain states so scaling cannot hide an omitted input.

    n = 8
    values = np.arange(n, dtype=float)
    state = values[:, None] + np.arange(len(CHAIN_STATE_FEATURES))[None, :]
    target = 1000. * (values + 1)
    full, _, _, static_scaler, dynamic_scaler, names = encoding.fit_person_trip_matrix(
        age=20 + values, sex=values % 2, employed=np.ones(n), car_availability=values % 2,
        income_class=values, daily_longest=10000 + values, daily_total=20000 + values,
        daily_longest_work=5000 + values, activity_chain_matrix=np.ones((n, 6)),
        consumed_before=values * 100, trip_position=values / n,
        departure_time=values / n, activity_duration_h=values,
        target_distance=target, purpose_series=pd.Series(['leisure'] * n),
        origin_purpose_series=pd.Series(['leisure'] * n), purpose_categories=['leisure'],
        use_target_distance=use_distance, random_state=1, chain_state_matrix=state)
    assert all(field in names for field in CHAIN_STATE_FEATURES)
    assert ('target_distance' in names) == use_distance
    for i in range(n):
        online = encoding.transform_person_trip_vector(
            age=20 + values[i], income_class=values[i], daily_longest=10000 + values[i],
            daily_total=20000 + values[i], daily_longest_work=5000 + values[i],
            activity_chain_vector=np.ones(6), sex=values[i] % 2, employed=1,
            car_availability=values[i] % 2, consumed_before=values[i] * 100,
            trip_position=values[i] / n, departure_time=values[i] / n,
            activity_duration_h=values[i], target_distance=target[i],
            purpose_hot=[1], origin_purpose_hot=[1], static_scaler=static_scaler,
            dynamic_scaler=dynamic_scaler, chain_state_vector=state[i])
        np.testing.assert_allclose(online[0], full[i], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match='remaining-chain'):
        encoding.transform_person_dynamic_vector(0, 0, 0, 1, 1000, [1], [1], dynamic_scaler)


class ConstantChoiceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3))
        self._device = torch.device('cpu')
        self.dropout_rate = 0.

    def forward(self, person_static, person_dynamic, candidate_static, candidate_dynamic):
        return self.logits.expand(len(person_static), -1)


def hybrid_training_inputs():
    n = 10
    return dict(person_static_x=np.zeros((n, 1)), person_dynamic_x=np.zeros((n, 1)),
        candidate_static_x=np.zeros((1, 3, 1)), candidate_dynamic_x=np.zeros((n, 3, 1)),
        y=np.zeros(n, dtype=int), valid_mask=np.tile([True, True, False], (n, 1)),
        distance_candidates=np.tile([50., 300., 9999.], (n, 1)),
        distance_targets=np.full(n, 100.),
        candidate_xy=np.tile([[50., 0.], [300., 0.], [9999., 0.]], (n, 1, 1)),
        observed_xy=np.tile([100., 0.], (n, 1)),
        distance_loss_kind='crps_energy', distance_loss_weight=.1,
        epochs=1, lr=0., num_threads=2, n_val_folds=2, val_every=1, grad_clip=None,
        validation_groups=np.arange(n), rotate_validation=False,
        weights=np.arange(n, dtype=np.float32)+1)


@pytest.mark.parametrize('refit', [False, True])
def test_auxiliary_relative_loss_matches_objective_and_gradients_in_all_fitting_paths(tmp_path, refit):
    import json
    from synthesis.population.spatial.secondary_nn.choice_model import train_choice_model
    from synthesis.population.spatial.secondary_nn.distribution_scores import relative_mean_distance_loss

    model = ConstantChoiceModel()
    inputs = hybrid_training_inputs()
    train_choice_model(model, **inputs, refit_full=refit, path=str(tmp_path),
                       auxiliary_relative_distance_loss_weight=.02)

    # Compute the objective independently with masked probabilities and a known radial error.

    logits = torch.zeros(3, requires_grad=True)
    masked = logits.masked_fill(torch.tensor([False, False, True]), -1e9)
    probabilities = torch.softmax(masked, dim=0)[None, :]
    distances = torch.tensor([[50., 300., 9999.]])
    target = torch.tensor([100.])
    crps = discrete_crps(probabilities, distances, target)
    energy = projected_energy_score(probabilities, torch.tensor([[[50.,0.],[300.,0.],[9999.,0.]]]),
                                    torch.tensor([[100.,0.]]))
    auxiliary = relative_mean_distance_loss(probabilities, distances, target)
    assert auxiliary.item() == pytest.approx(.5 * ((175.-100.)/200.)**2)
    expected = -torch.log_softmax(masked, dim=0)[0] + .1*(crps+energy).sum()/1000 + .02*auxiliary.sum()
    expected.backward()
    history = json.loads((tmp_path/'loss_history.json').read_text())
    assert history['train_losses'][0] == pytest.approx(expected.item(), abs=1e-6)
    assert history['val_losses'][0] == pytest.approx(expected.item(), abs=1e-6)
    assert history['auxiliary_relative_distance_loss_weight'] == .02
    assert history['refit_full'] == refit
    torch.testing.assert_close(model.logits.grad, logits.grad, atol=1e-6, rtol=1e-5)
    assert model.logits.grad[2].item() == 0.


@pytest.mark.parametrize('weight', [-.01, float('nan'), float('inf')])
def test_auxiliary_loss_rejects_invalid_weights(weight):
    from synthesis.population.spatial.secondary_nn.choice_model import train_choice_model
    with pytest.raises(ValueError, match='finite and nonnegative'):
        train_choice_model(ConstantChoiceModel(), **hybrid_training_inputs(),
                           auxiliary_relative_distance_loss_weight=weight)


def test_auxiliary_loss_requires_valid_origin_distance_supervision():
    from synthesis.population.spatial.secondary_nn.choice_model import train_choice_model
    inputs = hybrid_training_inputs()
    inputs['distance_candidates'] = None
    with pytest.raises(ValueError, match='requires origin-distance tensors'):
        train_choice_model(ConstantChoiceModel(), **inputs,
                           auxiliary_relative_distance_loss_weight=.02)


def test_auxiliary_relative_loss_supplies_correct_radius_gradient():
    from synthesis.population.spatial.secondary_nn.distribution_scores import relative_mean_distance_loss
    logits = torch.zeros((1, 2), requires_grad=True)
    probabilities = torch.softmax(logits, dim=1)
    loss = relative_mean_distance_loss(probabilities, torch.tensor([[50., 300.]]), torch.tensor([100.]))
    loss.sum().backward()
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]
    assert relative_mean_distance_loss(torch.tensor([[.8, .2]]), torch.tensor([[50., 300.]]), torch.tensor([100.])).item() == 0.


def test_short_range_checkpoint_uses_its_stage_cache(tmp_path):
    from synthesis.population.spatial.secondary_nn import short_range_model
    shared = tmp_path/'shared'
    stage = tmp_path/'stage'
    shared.mkdir()
    stage.mkdir()
    (shared/short_range_model.MODEL_NAME).write_text('baseline')
    (stage/short_range_model.MODEL_NAME).write_text('trial')
    context = SimpleNamespace(working_directory=str(shared), path=lambda:str(stage),
                              config=lambda key:False)
    result = short_range_model.execute(context)
    assert result == (str(stage/short_range_model.MODEL_NAME),)
    assert (shared/short_range_model.MODEL_NAME).read_text() == 'baseline'


def test_selector_uses_independent_folds_and_minimum_exploration(tmp_path):
    import json
    torch.set_num_threads(2)
    frame, comparisons, cells = mixture_fixture()
    fitted, audit = selector.train_selector(frame, comparisons, OriginContext(cells), seed=9,
        max_iter=8, patience=1, min_epochs=3, folds=5, learning_rate=1e-15, path=tmp_path)
    assert fitted.metrics['validation_mode']=='independent_kfold'
    assert fitted.metrics['validation_folds']==5
    assert fitted.metrics['epochs_completed']==4
    assert fitted.metrics['refit_epochs']==1
    outer=set(frame.loc[frame.partition.eq('validation'),'person_id'])
    folds=json.loads((tmp_path/'cross_validation_folds.json').read_text())
    held_out=[]
    for fold in folds:
        assert set(fold['train_people']).isdisjoint(fold['validation_people'])
        assert outer.isdisjoint(fold['train_people'])
        assert outer.isdisjoint(fold['validation_people'])
        held_out.extend(fold['validation_people'])
    assert sorted(held_out)==sorted(frame.loc[frame.partition.ne('validation'),'person_id'].unique())
    assert audit.loc[audit.partition.eq('validation'),'selector_validation_fold'].eq(-1).all()
    assert set(audit.loc[audit.partition.ne('validation'),'selector_validation_fold'])==set(range(5))
