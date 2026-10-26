"""Spatial extraction, feature alignment and leakage-free expert selection."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, LineString, box

from synthesis.population.spatial.secondary_nn import tools
from synthesis.population.spatial.secondary_nn import feature_encoding as encoding
from synthesis.population.spatial.secondary_nn import short_trip_classifier as selector
from synthesis.population.spatial.secondary_nn import training_split
from synthesis.population.spatial.secondary_nn.chain_state import CHAIN_STATE_FEATURES
from synthesis.population.spatial.secondary_nn.distribution_scores import ensemble_score_components, mixture_score


def features(rows):
    return gpd.GeoDataFrame([dict(osm_id=str(i), name="", public=True, **row)
                             for i, row in enumerate(rows)], geometry="geometry", crs=2056)


def test_public_water_access_produces_one_connected_visit_point():
    raw = features([
        dict(kind="water", geometry=box(0, 0, 200, 100)),
        dict(kind="path", geometry=LineString([(0, -5), (100, -5)])),
        dict(kind="path", geometry=LineString([(100, -5), (200, -5)])),
        dict(kind="river", geometry=LineString([(400, 0), (400, 200)])),
        dict(kind="path", geometry=LineString([(405, 0), (405, 200)])),
    ])
    result = tools.derive_water_features(raw, 10)
    assert len(result.loc[result.kind == "shore_path"]) == 1
    assert 200 <= result.loc[result.kind == "shoreline"].geometry.length.sum() < 220
    candidates = tools.candidate_points(result)
    assert set(candidates.kind) == {"shore_path", "riverside_path"}
    assert candidates.loc[candidates.kind == "shore_path"].geometry.iloc[0].equals(Point(100, -5))
    assert not tools.public_access({"access": "private"})
    assert not tools.public_access({"foot": "no"})


def test_park_point_uses_public_path_inside_area():
    raw = features([dict(kind="park", geometry=box(0, 0, 100, 100)),
                    dict(kind="path", geometry=LineString([(0, 10), (100, 10)]))])
    point = tools.candidate_points(raw).geometry.iloc[0]
    assert point.equals(Point(50, 10))


def test_new_facilities_deduplicate_companies_and_only_offer_leisure_other():
    companies = gpd.GeoDataFrame(dict(destination_id=[1], number_employees=[10],
        offers_work=[True], offers_shop=[False], offers_education=[False],
        offers_work_secondary=[False], offers_home_secondary=[False],
        offers_education_secondary=[False], offers_other=[False], offers_leisure=[False],
        ovgk=["A"], municipality_id=[1], municipality_type=["urban"]), geometry=[Point(0, 0)], crs=2056)
    raw = features([dict(kind="viewpoint", geometry=Point(49, 0)),
                    dict(kind="garden", geometry=Point(150, 0)),
                    dict(kind="attraction", geometry=Point(160, 0)),
                    dict(kind="worship", geometry=Point(400, 0))])
    municipalities = gpd.GeoDataFrame(dict(municipality_id=[1], municipality_type=["urban"]), geometry=[box(-1000, -1000, 1000, 1000)], crs=2056)
    ovgk = gpd.GeoDataFrame(dict(ovgk=["D", "A"]), geometry=[box(-1000, -1000, 1000, 1000)] * 2, crs=2056)
    result = tools.add_osm_destinations(companies, raw, municipalities, ovgk)
    added = result.loc[result.destination_id.astype(str).str.startswith("OSM_")]
    assert len(added) == 2
    assert added.number_employees.eq(1).all()
    assert added.offers_work.eq(False).all()
    assert added.offers_leisure.all() and added.offers_other.all()
    assert not added[[c for c in added if c.startswith("offers_") and c not in ("offers_leisure", "offers_other")]].any().any()
    assert added.ovgk.eq("A").all()
    assert result.destination_id.is_unique


def test_context_uses_clipped_intersections_and_does_not_duplicate_boundary_points():
    cells = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=2056)
    raw = features([dict(kind="park", geometry=box(50, 0, 150, 100)),
                    dict(kind="piste", geometry=LineString([(0, 50), (200, 50)])),
                    dict(kind="viewpoint", geometry=Point(100, 50))])
    context = tools.aggregate_osm_features(cells, raw)
    np.testing.assert_allclose(context.osm_park_area_m2, [5000, 5000])
    np.testing.assert_allclose(context.osm_ski_piste_length_m, [100, 100])
    assert context.osm_viewpoint_count.sum() == 1


def test_environmental_tensor_order_matches_saved_candidate_schema():
    cells = gpd.GeoDataFrame({c: [i + 1, i + 2] for i, c in enumerate(tools.OSM_FEATURES)}, geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs=2056)
    cells["centroid"] = cells.geometry.representative_point()
    tensor = np.arange(21, dtype=float)[None, None, :].repeat(2, axis=1)
    mask = np.array([[True, False]])
    result = tools.enrich_candidate_tensor(tensor, np.array([.5, 2.5]), np.array([.5, .5]), mask, cells)
    assert result.shape[-1] == len(encoding.CANDIDATE_FEATURES)
    for i, column in enumerate(tools.OSM_FEATURES):
        index = encoding.CANDIDATE_FEATURES.index(column)
        assert result[0, 0, index] == i + 1
        assert result[0, 1, index] == 0
    assert result[0, 0, encoding.CANDIDATE_FEATURES.index("urban_core")] == 13


def test_mobility_recodes_bike_and_subscription_data_consistently():
    persons = pd.DataFrame(dict(driving_license=[True, False], bike_availability=[0, 2],
                                subscriptions=["HTA+VA", "GA"], ovgk=["B", None]))
    constants = SimpleNamespace(BIKE_AVAILABILITY_ALWAYS=0, BIKE_AVAILABILITY_SOMETIMES=1, BIKE_AVAILABILITY_NEVER=2)
    result = tools.mobility_features(persons, constants, survey=True)
    assert result.bike_available.tolist() == [1, 0]
    assert result.pt_halbtax.tolist() == [1, 0]
    assert result.pt_regional.tolist() == [1, 0]
    assert result.pt_ga.tolist() == [0, 1]
    assert result.home_ovgk_none.tolist() == [0, 1]


def test_energy_score_penalizes_wrong_direction_and_does_not_cancel_predictions():
    import torch
    right = ensemble_score_components([[1000, 0]] * 10, [[-1000, 0]] * 10, [0, 0], [1000, 0])
    stats = {key: torch.tensor(value) for key, value in right.items()}
    assert mixture_score(torch.tensor(1.), stats, 'energy') == 0
    assert mixture_score(torch.tensor(0.), stats, 'energy') == 2
    assert mixture_score(torch.tensor(.5), stats, 'energy') > 0


def selector_frame(n=100):
    frame = pd.DataFrame(dict(person_id=np.arange(n), trip_id=np.zeros(n, dtype=int),
        age=np.arange(n) % 70 + 18, sex=np.arange(n) % 2, employed=np.ones(n),
        car_availability=np.arange(n) % 2, income_class=np.arange(n) % 9,
        daily_longest_distance_from_home=np.ones(n) * 5000,
        daily_crowfly_total=np.ones(n) * 12000, daily_longest_distance_from_work=np.ones(n) * 5000,
        crowfly_consumed_before_trip=np.zeros(n), trip_position_class=np.zeros(n),
        departure_time_normalized=np.ones(n) * .5, activity_duration_h=np.ones(n),
        target_distance=np.ones(n) * 1000, purpose=["leisure"] * n, origin_purpose=["home"] * n,
        origin_x=np.zeros(n), origin_y=np.zeros(n), home_x=np.zeros(n), home_y=np.zeros(n),
        work_x=np.ones(n) * 100, work_y=np.ones(n) * 100, has_work=np.ones(n, dtype=bool),
        destination_x=np.ones(n) * 1000, destination_y=np.zeros(n), person_weight=np.ones(n)))
    frame['chain_state'] = [np.zeros(len(CHAIN_STATE_FEATURES))] * n
    frame['observed_leaf'] = 'test'
    frame['partition'] = np.where(np.arange(n) % 5 == 0, 'validation', np.where(np.arange(n) % 5 == 1, 'stopping', 'train'))
    frame['fold'] = np.where(frame.partition.eq('validation'), -1, np.arange(n) % 3)
    frame["activity_chain"] = [np.ones(6)] * n
    frame[tools.MOBILITY_FEATURES] = np.zeros((n, len(tools.MOBILITY_FEATURES)))
    return frame


def context_cells():
    cells = gpd.GeoDataFrame({c: [1., 2.] for c in encoding.STATIC_CANDIDATE_FEATURES}, geometry=[box(0, 0, 100, 100), box(1000, 0, 1100, 100)], crs=2056)
    cells["centroid"] = cells.geometry.representative_point()
    return cells


def test_selector_validation_is_person_disjoint_and_features_match_inference():
    frame = selector_frame()
    comparisons = frame[["person_id", "trip_id"]].copy()
    for column, value in ensemble_score_components([[1000, 0]] * 10, [[-1000, 0]] * 10, [0, 0], [1000, 0]).items():
        comparisons[column] = value
    comparisons['short_leaf_probability'] = np.where(frame.age > 50, .8, .2)
    comparisons['hierarchy_leaf_probability'] = 1 - comparisons.short_leaf_probability
    comparisons['prediction_fold'] = frame.fold
    for expert in ['short', 'hierarchy']:
        comparisons[expert + '_mean_distance_m'] = 1000.
        comparisons[expert + '_long_share'] = 0.
    model, audit = selector.train_selector(frame, comparisons, tools.OriginContext(context_cells()), seed=1, max_iter=10, min_epochs=0)
    assert audit.groupby("person_id").partition.nunique().eq(1).all()
    assert set(audit.partition) == {"train", "stopping", "validation"}
    assert "target_distance" not in model.feature_names
    assert "destination_x" not in model.feature_names
    assert model.metrics["experts_exclude_outer_validation"]
    row = frame.iloc[0]
    inputs = selector.prediction_inputs(row)
    renamed = dict(age=row.age, income_class=row.income_class, daily_longest=row.daily_longest_distance_from_home,
        daily_total=row.daily_crowfly_total, daily_longest_work=row.daily_longest_distance_from_work,
        activity_chain_vector=row.activity_chain, sex=row.sex, employed=row.employed,
        car_availability=row.car_availability, consumed_before=row.crowfly_consumed_before_trip,
        trip_position=row.trip_position_class, departure_time=row.departure_time_normalized,
        activity_duration_h=row.activity_duration_h, purpose=row.purpose, origin_purpose=row.origin_purpose,
        mobility_vector=inputs["mobility_vector"], origin_x=row.origin_x, origin_y=row.origin_y,
        home_x=row.home_x, home_y=row.home_y, work_x=row.work_x, work_y=row.work_y, has_work=row.has_work, chain_state_vector=row.chain_state)
    matrix, *_ = selector._feature_matrix(frame.iloc[:1], model.origin_context, model.static_scaler, model.dynamic_scaler)
    import torch
    matrix = model.feature_scaler.transform(np.log1p(np.maximum(matrix, 0))).astype(np.float32)
    with torch.inference_mode():
        expected = torch.sigmoid(model.network(torch.from_numpy(matrix))).item()
    assert model.predict_probability(**renamed) == pytest.approx(expected)


def test_expert_predictions_repeat_reproducibly_and_use_no_target_distance():
    class Wrapper:
        def expert_leaf_probabilities(self, leaf, **kwargs):
            assert kwargs['target_distance'] == 0
            return .8, .2
        def predict(self, **kwargs):
            assert kwargs["target_distance"] == 0
            offset = 1000 if kwargs["force_expert"] == "short" else -1000
            return "test", Point(offset + kwargs["rng"].normal(), 0)
    frame = selector_frame(2)
    a = selector.compare_experts(frame, Wrapper(), draws=10, seed=13)
    b = selector.compare_experts(frame, Wrapper(), draws=10, seed=13)
    pd.testing.assert_frame_equal(a, b)
    assert a.energy_short_target.lt(a.energy_hierarchy_target).all()


def test_cross_fitted_people_and_outer_validation_never_overlap():
    persons = pd.DataFrame(dict(person_id=np.arange(100), weekend=False))
    class Context:
        fold = -1
        def config(self, key):
            return {"random_seed": 1, "secondary_nn_crossfit_folds": 3,
                    "secondary_nn_selector_validation_fraction": .2,
                    "secondary_nn_crossfit_fold": self.fold,
                    "secondary_nn_use_target_distance": False}[key]
        def stage(self, stage):
            return persons if stage == "data.microcensus.persons" else split
    context = Context()
    split = training_split.execute(context)
    expert = training_split.expert_persons(context, persons)
    assert len(expert) == 80
    assert set(expert.person_id).isdisjoint(split.loc[split.partition.eq('validation'), 'person_id'])
    for fold in range(3):
        context.fold = fold
        expert = training_split.expert_persons(context, persons)
        assert set(expert.person_id).isdisjoint(split.loc[split.fold.eq(fold), 'person_id'])
        assert set(expert.person_id).isdisjoint(split.loc[split.partition.eq('validation'), 'person_id'])


def test_raw_osm_geometry_extraction_includes_multipolygons(tmp_path):
    from data.osm.secondary_features import read_features
    osm = tmp_path / "test.osm"
    osm.write_text('''<osm version="0.6">
<node id="1" lon="8.50" lat="47.30"/><node id="2" lon="8.51" lat="47.30"/>
<node id="3" lon="8.51" lat="47.31"/><node id="4" lon="8.50" lat="47.31"/>
<node id="5" lon="8.505" lat="47.305"><tag k="tourism" v="viewpoint"/></node>
<node id="6" lon="8.506" lat="47.306"><tag k="aerialway" v="station"/></node>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/></way>
<way id="11"><nd ref="3"/><nd ref="4"/><nd ref="1"/></way>
<way id="12"><nd ref="1"/><nd ref="3"/><tag k="piste:type" v="downhill"/></way>
<relation id="20"><member type="way" ref="10" role="outer"/><member type="way" ref="11" role="outer"/>
<tag k="type" v="multipolygon"/><tag k="leisure" v="park"/><tag k="name" v="Test park"/></relation>
</osm>''')
    result = read_features(osm, str(tmp_path))
    assert {"park", "viewpoint", "cable_station", "piste"}.issubset(result.kind)
    park = result.loc[result.kind == "park"].iloc[0]
    assert park.osm_id == "r20" and park.geometry.area > 0
    assert not list(tmp_path.glob("osm_locations_*"))


def test_synthetic_subscription_categories_match_survey_priorities():
    population = pd.DataFrame(dict(pt_subscription=[0, 1, 2, 3, 4], bike_availability=1))
    out = tools.mobility_features(population)
    assert out.pt_ga.tolist() == [0, 1, 0, 0, 0]
    assert out.pt_regional.tolist() == [0, 0, 1, 0, 1]
    assert out.pt_halbtax.tolist() == [0, 0, 0, 1, 1]


class Progress:
    def __init__(self, iterable=None):
        self.iterable = iterable
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        return iter(self.iterable)
    def update(self, *args):
        pass


@pytest.mark.parametrize("distance_enabled, loss_kind", [(True, "crps_energy"), (True, "relative"), (False, "crps_energy")])
def test_all_experts_and_selector_train_reload_and_predict(tmp_path, monkeypatch, distance_enabled, loss_kind):
    """Exercise the real stage chain on a small Swiss geometry and population."""
    import importlib
    from data.constants import Constants
    from synthesis.population.spatial.secondary_nn import h3 as hierarchy
    from synthesis.population.spatial.secondary_nn.model_wrappers import HierarchicalLocationChoiceModel
    import torch
    torch.set_num_threads(2)
    class Detour:
        def get_cell_indices(self, x, y):
            return np.zeros(np.asarray(x).reshape(-1).shape, dtype=int)
        def get_detour_factor_by_index(self, a, b):
            return np.ones(np.broadcast_shapes(np.shape(a), np.shape(b)))
    class Context:
        def __init__(self):
            self.configs = dict(random_seed=17, threads=2, secondary_nn_use_target_distance=distance_enabled,
                secondary_nn_selector_draws=10, secondary_nn_selector_max_iter=5,
                secondary_nn_selector_min_samples_leaf=2)
            if loss_kind != "crps_energy":
                self.configs["secondary_nn_distance_loss_kind"] = loss_kind
            self.stages = {}
            self.directory = tmp_path
        @property
        def working_directory(self):
            return str(self.directory)
        def path(self):
            self.directory.mkdir(parents=True, exist_ok=True)
            return str(self.directory)
        def stage(self, name, config=None, alias=None):
            if alias is not None or name.endswith('cross_fitted_experts'):
                return self.stages.get(alias or name)
            return self.stages[name]
        def config(self, key, default=None, **kwargs):
            if key not in self.configs:
                self.configs[key] = default
            return self.configs[key]
        def progress(self, iterable=None, **kwargs):
            return Progress(iterable)
    context = Context()
    constants = Constants()
    rng = np.random.default_rng(21)
    n_people, n_dest = 80, 160
    xy = np.column_stack([2680000 + rng.uniform(-15000, 15000, n_dest),
                          1250000 + rng.uniform(-8000, 8000, n_dest)])
    # Dense pairs guarantee multiple finest-level alternatives in occupied branches.
    xy = np.concatenate([xy, xy + [250, 250]], axis=0)
    destinations = gpd.GeoDataFrame(dict(destination_id=np.arange(len(xy)), number_employees=1,
        ovgk="B", municipality_type="urban", offers_work=False, offers_education=False,
        **{f"offers_{p}": p in ("leisure", "other") for p in selector.SECONDARY_ACTIVITIES}),
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=2056)
    destinations["destination_id"] = destinations.destination_id.astype(object)
    destinations.loc[len(destinations) - 1, "destination_id"] = "OSM_test"
    persons = pd.DataFrame(dict(person_id=np.arange(n_people), age=rng.integers(18, 80, n_people),
        sex=rng.integers(0, 2, n_people), employed=True, car_availability=rng.integers(0, 3, n_people),
        income_class=rng.integers(0, 9, n_people), home_x=2680000., home_y=1250000.,
        work_x=2681000., work_y=1251000., weekend=False, person_weight=1.,
        driving_license=True, bike_availability=0, subscriptions_ga=True))
    records = []
    chains = []
    for person_id in range(n_people):
        for trip_id in range(3):
            target = xy[int(rng.integers(len(xy)))]
            origin = target - [350, 250]
            records.append(dict(person_id=person_id, trip_id=trip_id,
                origin_x=origin[0], origin_y=origin[1], destination_x=target[0], destination_y=target[1],
                purpose="leisure", origin_purpose="home"))
            chains.append(dict(person_id=person_id, trip_id=trip_id,
                daily_longest_distance_from_home=15000., daily_crowfly_total=20000.,
                crowfly_consumed_before_trip=500. * trip_id, trip_position_class=trip_id / 2,
                departure_time_normalized=.5, daily_longest_distance_from_work=14000.,
                activity_duration_h=2., target_distance=float(np.linalg.norm(target - origin)),
                trip_destination_distance_from_home=float(np.linalg.norm(target - [2680000, 1250000])),
                trip_origin_distance_from_home=float(np.linalg.norm(origin - [2680000, 1250000])),
                activity_chain=np.array([0, 0, 3, 0, 0, 0]), chain_state=np.zeros(len(CHAIN_STATE_FEATURES))))
    context.stages.update({
        "data.constants": constants, "data.microcensus.persons": persons,
        "data.microcensus.trips": (pd.DataFrame(records), None, None, None),
        "synthesis.population.destinations": destinations,
        "data.statent.statent": pd.DataFrame(dict(enterprise_id=np.arange(len(xy)), noga="93")),
        "data.spatial.swiss_border": gpd.GeoDataFrame(geometry=[box(2600000, 1200000, 2750000, 1300000)], crs=2056),
        "data.spatial.ovgk": gpd.GeoDataFrame(dict(ovgk=["B"]), geometry=[box(2600000, 1200000, 2750000, 1300000)], crs=2056),
        "data.osm.secondary_features": features([dict(kind="park", geometry=box(2679000, 1249000, 2681000, 1251000))]),
        "synthesis.population.spatial.secondary_nn.mz_chains": pd.DataFrame(chains),
        "synthesis.population.spatial.secondary.detour_factors.factors": Detour(),
    })
    split_stage = "synthesis.population.spatial.secondary_nn.training_split"
    training_split.configure(context)
    context.stages[split_stage] = training_split.execute(context)
    context.stages["synthesis.population.spatial.secondary_nn.h3"] = hierarchy.execute(context)
    seen = []
    for name in ["regional_model", "subregional_model", "local_model", "short_range_model"]:
        module = importlib.import_module("synthesis.population.spatial.secondary_nn." + name)
        module.configure(context)
        context.configs[name + "_epochs"] = 1
        context.configs["secondary_nn_model_min_epochs"] = 0
        if name == "short_range_model":
            context.configs["short_range_model_min_epochs"] = 0
        context.directory = tmp_path / name
        monkeypatch.setattr(module, "plot_distance_analysis" if name == "short_range_model" else "plot_analysis", lambda *args, **kwargs: None)
        original_mobility = module.training_mobility
        def mobility(ctx, frame, _original=original_mobility):
            reserved = context.stages[split_stage].query("partition == 'validation'").person_id
            if not distance_enabled:
                assert set(frame.person_id).isdisjoint(reserved)
            return _original(ctx, frame)
        monkeypatch.setattr(module, "training_mobility", mobility)
        original = module.fit_person_trip_matrix
        def fit(*args, _original=original, **kwargs):
            assert kwargs["chain_state_matrix"].shape[1] == len(CHAIN_STATE_FEATURES)
            assert kwargs["mobility_matrix"].shape[1] == len(tools.MOBILITY_FEATURES)
            assert kwargs["mobility_matrix"][:, tools.MOBILITY_FEATURES.index("pt_ga")].all()
            seen.append(name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, "fit_person_trip_matrix", fit)
        if name == "short_range_model":
            original_train = module.train_choice_model
            def check_short_choices(*args, _train=original_train, **kwargs):
                from sklearn.neighbors import KDTree
                audit = pd.read_parquet(context.directory / "short_range_choice_sets.parquet")
                used = audit.loc[audit.reason.eq("included")]
                raw = pd.DataFrame(records).merge(used[["person_id", "trip_id"]],
                    on=["person_id", "trip_id"], how="right", validate="one_to_one")
                np.testing.assert_array_equal(kwargs["validation_groups"], raw.person_id)
                cells = context.stages["synthesis.population.spatial.secondary_nn.h3"][1]["level_2"]
                ids = cells.h3_index.astype(str).tolist()
                centres = np.column_stack([cells["centroid"].x, cells["centroid"].y])
                mapped = context.stages["synthesis.population.spatial.secondary_nn.h3"][0]["destinations"]
                eligible_ids = mapped.merge(destinations.loc[destinations.offers_leisure, ["destination_id"]],
                    on="destination_id").level_2.astype(str)
                support = np.isin(ids, eligible_ids)
                query_tree = KDTree(centres)
                for i, row in raw.iterrows():
                    selected = tools.origin_candidate_indices(query_tree, row.origin_x, row.origin_y,
                        support, radius_m=1800, min_candidates=6, expand_step_m=200)
                    assert kwargs["valid_mask"][i].sum() == len(selected)
                    if kwargs["candidate_xy"] is not None:
                        np.testing.assert_allclose(kwargs["candidate_xy"][i, :len(selected)], centres[selected])
                    assert kwargs["valid_mask"][i, kwargs["y"][i]]
                return _train(*args, **kwargs)
            monkeypatch.setattr(module, "train_choice_model", check_short_choices)
        context.path()
        context.stages["synthesis.population.spatial.secondary_nn." + name] = module.execute(context)
        import json
        history = json.loads((context.directory / "loss_history.json").read_text())
        expected_kind = (context.configs.get("secondary_nn_short_distance_loss_kind") or loss_kind) if name == "short_range_model" and distance_enabled else loss_kind
        assert history["distance_loss_kind"] == expected_kind
        expected_auxiliary = .02 if name == "short_range_model" and distance_enabled and expected_kind == "crps_energy" else 0.
        expected_auxiliary = context.configs.get("secondary_nn_short_relative_distance_loss_weight", .02) if expected_auxiliary else 0.
        assert history["auxiliary_relative_distance_loss_weight"] == expected_auxiliary
        expected_distribution = context.configs.get("secondary_nn_distribution_loss_weight", .1)
        if name == "short_range_model" and distance_enabled:
            override = context.configs.get("secondary_nn_short_distribution_loss_weight")
            if override is not None:
                expected_distribution = override
        expected_weight = expected_distribution if expected_kind == "crps_energy" else context.configs.get("secondary_nn_distance_loss_weight", .07)
        assert history["distance_loss_weight"] == expected_weight
        expected_factor = (context.configs.get("secondary_nn_short_relative_underestimation_factor", 2.)
            if name == "short_range_model" and distance_enabled and expected_kind == "relative" else 1.)
        assert history["relative_distance_underestimation_factor"] == expected_factor
        assert np.isfinite(history["train_losses"]).all()
    assert len(seen) == 4
    if distance_enabled:
        model = HierarchicalLocationChoiceModel.build(context, optimize=False)
        row = pd.DataFrame(records).merge(pd.DataFrame(chains), on=["person_id", "trip_id"]).merge(persons, on="person_id").iloc[0]
        row["has_work"] = True
        for column, value in tools.mobility_features(persons, constants, survey=True).iloc[0].items():
            row[column] = value
        inputs = selector.prediction_inputs(row)
        inputs["target_distance"] = row.target_distance
        predicted = model.predict(**inputs, rng=np.random.default_rng(11))
        assert predicted[1].geom_type == "Point"
        assert all(field in model.c.person_trip_cols for field in CHAIN_STATE_FEATURES)
        assert "target_distance" in model.c.person_trip_cols
        assert model.last_selected_expert == "short"
        inputs["target_distance"] = 50000.
        model.predict(**inputs, rng=np.random.default_rng(11))
        assert model.last_selected_expert == "hierarchy"
        return
    from synthesis.population.spatial.secondary_nn import cross_fitted_experts
    final = {name: context.stages[cross_fitted_experts.PREFIX + name] for name in cross_fitted_experts.MODELS}
    cross_fitted_experts.configure(context)
    for fold in range(3):
        context.configs['secondary_nn_crossfit_fold'] = fold
        for name in cross_fitted_experts.MODELS:
            module = importlib.import_module(cross_fitted_experts.PREFIX + name)
            context.directory = tmp_path / f'fold_{fold}' / name
            context.path()
            value = module.execute(context)
            context.stages[cross_fitted_experts.PREFIX + name] = value
            context.stages[f'fold_{fold}_{name}'] = value
    context.configs['secondary_nn_crossfit_fold'] = -1
    context.stages.update({cross_fitted_experts.PREFIX + name: value for name, value in final.items()})
    context.directory = tmp_path / 'cross_fitted'
    context.path()
    context.stages[cross_fitted_experts.PREFIX + 'cross_fitted_experts'] = cross_fitted_experts.execute(context)
    selector.configure(context)
    context.directory = tmp_path / "selector"
    context.path()
    result = selector.execute(context)
    context.stages["synthesis.population.spatial.secondary_nn.short_trip_classifier"] = result
    model = HierarchicalLocationChoiceModel.build(context, optimize=False)
    frame = selector.prepare_training_data(context)
    inputs = selector.prediction_inputs(frame.iloc[0])
    predicted = model.predict(**inputs, rng=np.random.default_rng(11))
    assert predicted[1].geom_type == "Point"
    inputs["target_distance"] = 50000.0
    same = model.predict(**inputs, rng=np.random.default_rng(11))
    assert predicted[0] == same[0] and predicted[1].equals(same[1])
    import joblib
    saved = joblib.load(result[0])
    assert saved.metrics["draws_per_expert"] == 10
    assert saved.metrics['experts_exclude_outer_validation']
    assert saved.metrics['development_predictions_are_cross_fitted']
    assert saved.metrics['probabilistic_routing']


def test_branched_waterfront_is_one_place_with_stable_identity():
    raw = features([
        dict(kind="water", geometry=box(0, 0, 200, 100)),
        dict(kind="path", geometry=LineString([(0, -5), (100, -5), (200, -5)])),
        dict(kind="path", geometry=LineString([(100, -5), (100, -15)])),
    ])
    result = tools.derive_water_features(raw, 20)
    assert len(result.loc[result.kind == "shore_path"]) == 1
    assert len(tools.candidate_points(result)) == 1
    identity = result.loc[result.kind == "shore_path", "osm_id"].iloc[0]
    shuffled = tools.derive_water_features(raw.iloc[::-1], 20)
    assert shuffled.loc[shuffled.kind == "shore_path", "osm_id"].iloc[0] == identity


def test_closed_osm_way_with_path_and_park_keeps_polygon_area(tmp_path):
    from data.osm.secondary_features import read_features
    osm = tmp_path / "mixed.osm"
    osm.write_text('''<osm version="0.6">
<node id="1" lon="8.50" lat="47.30"/><node id="2" lon="8.51" lat="47.30"/>
<node id="3" lon="8.51" lat="47.31"/><node id="4" lon="8.50" lat="47.31"/>
<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
<tag k="leisure" v="park"/><tag k="highway" v="footway"/></way>
</osm>''')
    result = read_features(osm, str(tmp_path))
    assert result.loc[result.kind == "park"].geometry.area.sum() > 0
    assert len(result.loc[result.kind == "park"]) == 1
    assert result.loc[result.kind == "path"].geometry.length.sum() > 0


def test_accessibility_handles_categorical_classes_and_uncovered_homes():
    ovgk = gpd.GeoDataFrame(dict(ovgk=pd.Categorical(["A", "B"])),
                            geometry=[box(0, 0, 100, 100), box(0, 0, 200, 200)], crs=2056)
    result = tools.accessibility_at_points([50, 150, 300], [50, 150, 300], ovgk)
    assert result.tolist() == ["A", "B", "None"]


def test_secondary_stage_graphs_schedule_enrichment_and_selector_conditionally():
    from pathlib import Path
    import yaml
    from synpp.pipeline import process_stages
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config_dib_10.yml").read_text())["config"]
    for algorithm, distance in [("rda", True), ("nn", True), ("nn", False)]:
        settings = dict(config, locations_algorithm=algorithm, secondary_nn_use_target_distance=distance)
        registry = process_stages([dict(descriptor="synthesis.population.spatial.secondary.locations",
                                        config={}, downstream=[])], settings)
        names = {stage["wrapper"].name for stage in registry.values()}
        assert ("data.osm.secondary_features" in names) == (algorithm == "nn")
        assert ("synthesis.population.spatial.secondary_nn.short_trip_classifier" in names) == (algorithm == "nn" and not distance)
        assert ("synthesis.population.spatial.secondary_nn.training_split" in names) == (algorithm == "nn" and not distance)
