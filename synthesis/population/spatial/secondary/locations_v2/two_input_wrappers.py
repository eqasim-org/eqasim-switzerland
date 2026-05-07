from matplotlib.style import context
import numpy as np
from synthesis.population.spatial.secondary.locations_v2.locations_v2_helpers import _prepare_destination_level2_index, _reverse_tree
import torch
import logging
from .two_input_features import CANDIDATE_FEATURES, STATIC_CANDIDATE_FEATURES, DYNAMIC_CANDIDATE_FEATURES, transform_candidate_matrix, transform_person_trip_vector
from .two_input_nn import TwoInputChoiceModel, predict_two_input_proba
from .hierarchical_model_utils import build_dynamic_vector

def _torch_load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_float_array(values):
    return np.asarray(values, dtype=np.float64)


def _candidate_column_indices(candidate_cols):
    static_indices = [candidate_cols.index(name) for name in STATIC_CANDIDATE_FEATURES]
    dynamic_indices = [candidate_cols.index(name) for name in DYNAMIC_CANDIDATE_FEATURES]
    return static_indices, dynamic_indices


def _freeze_candidate_attr_map(attr_map, max_candidates=None):
    """Freeze and optionally pad all per-key attribute arrays to max_candidates.

    Padding with zeros ensures every key's arrays have the same length, which
    gives the model fixed-shape tensors on every call — preventing torch.compile
    from recompiling on each new candidate-set size.  Padded positions carry
    value 0 for every attribute, so amenity masks (e.g. num_statent > 0) are
    automatically False for those slots.  The 'children' list is never padded;
    it always reflects the true candidate H3 IDs.
    """
    frozen = {}
    for key, attrs in attr_map.items():
        x = _to_float_array(attrs["x"])
        y = _to_float_array(attrs["y"])
        num_statent = _to_float_array(attrs["num_statent"])
        employees = _to_float_array(attrs["employees"])
        urban_core = _to_float_array(attrs["urban_core"])
        urban = _to_float_array(attrs["urban"])
        education = _to_float_array(attrs["education"])
        shop = _to_float_array(attrs["shop"])
        leisure = _to_float_array(attrs["leisure"])
        ovgk_share_a = _to_float_array(attrs["ovgk_share_a"])
        ovgk_share_b = _to_float_array(attrs["ovgk_share_b"])
        ovgk_share_c = _to_float_array(attrs["ovgk_share_c"])
        ovgk_share_d = _to_float_array(attrs["ovgk_share_d"])
        ovgk_share_none = _to_float_array(attrs["ovgk_share_none"])
        outside_fraction = _to_float_array(attrs["outside_fraction"])
        static_feature_map = {
            "num_statent": num_statent,
            "employees": employees,
            "urban_core": urban_core,
            "urban": urban,
            "education": education,
            "shop": shop,
            "leisure": leisure,
            "ovgk_share_a": ovgk_share_a,
            "ovgk_share_b": ovgk_share_b,
            "ovgk_share_c": ovgk_share_c,
            "ovgk_share_d": ovgk_share_d,
            "ovgk_share_none": ovgk_share_none,
            "outside_fraction": outside_fraction,
        }
        static_features = np.column_stack([static_feature_map[name] for name in STATIC_CANDIDATE_FEATURES]).astype(np.float64)

        # Pad to max_candidates so every call produces the same tensor shape.
        # 'pad' is the number of rows to append.  If the arrays were already
        # padded on a previous freeze (e.g. after save/load), current length
        # already equals max_candidates and pad == 0 — idempotent.
        n_children = len(attrs["children"])  # true (unpadded) candidate count
        pad = max(0, (max_candidates - len(x))) if max_candidates is not None else 0
        if pad > 0:
            z1d = np.zeros(pad, dtype=np.float64)
            x = np.concatenate([x, z1d])
            y = np.concatenate([y, z1d])
            for name in static_feature_map:
                static_feature_map[name] = np.concatenate([static_feature_map[name], z1d])
            static_features = np.vstack([static_features, np.zeros((pad, static_features.shape[1]), dtype=np.float64)])
            # refresh local references after padding
            num_statent = static_feature_map["num_statent"]
            employees = static_feature_map["employees"]
            urban_core = static_feature_map["urban_core"]
            urban = static_feature_map["urban"]
            education = static_feature_map["education"]
            shop = static_feature_map["shop"]
            leisure = static_feature_map["leisure"]
            ovgk_share_a = static_feature_map["ovgk_share_a"]
            ovgk_share_b = static_feature_map["ovgk_share_b"]
            ovgk_share_c = static_feature_map["ovgk_share_c"]
            ovgk_share_d = static_feature_map["ovgk_share_d"]
            ovgk_share_none = static_feature_map["ovgk_share_none"]
            outside_fraction = static_feature_map["outside_fraction"]

        frozen[key] = {
            "children": list(attrs["children"]),
            "n_children": n_children,
            "x": x,
            "y": y,
            "num_statent": num_statent,
            "employees": employees,
            "urban_core": urban_core,
            "urban": urban,
            "education": education,
            "shop": shop,
            "leisure": leisure,
            "ovgk_share_a": ovgk_share_a,
            "ovgk_share_b": ovgk_share_b,
            "ovgk_share_c": ovgk_share_c,
            "ovgk_share_d": ovgk_share_d,
            "ovgk_share_none": ovgk_share_none,
            "outside_fraction": outside_fraction,
            "static_features": static_features,
        }
    return frozen


class CoarseLevel0TwoInputWrapper:
    def __init__(self, model, person_scaler, candidate_scaler, person_trip_cols, candidate_cols, all_h3, centroid_x, centroid_y, static_candidate_features, purpose_categories):
        self.model = model
        self.person_scaler = person_scaler
        self.candidate_scaler = candidate_scaler
        self.person_trip_cols = list(person_trip_cols)
        self.candidate_cols = list(candidate_cols)
        self.all_h3 = list(all_h3)
        self.centroid_x = np.asarray(centroid_x, dtype=np.float64)
        self.centroid_y = np.asarray(centroid_y, dtype=np.float64)
        self.static_candidate_features = np.asarray(static_candidate_features, dtype=np.float32)
        self.static_feature_indices, self.dynamic_feature_indices = _candidate_column_indices(self.candidate_cols)
        assert self.static_candidate_features.ndim == 2, "coarse static_candidate_features must be 2D"
        assert self.static_candidate_features.shape[1] == len(self.static_feature_indices), (
            f"coarse static_candidate_features width mismatch: expected {len(self.static_feature_indices)} got {self.static_candidate_features.shape[1]}"
        )
        self.purpose_categories = [str(p) for p in purpose_categories]
        self.mask = self.build_mask()

    def save(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": {
                    "person_input_dim": self.model.person_input_dim,
                    "candidate_input_dim": self.model.candidate_input_dim,
                    "person_hidden_dim": self.model.person_hidden_dim,
                    "hidden_dims": self.model.hidden_dims,
                    "dropout_rate": self.model.dropout_rate,
                },
                "person_scaler": self.person_scaler,
                "candidate_scaler": self.candidate_scaler,
                "person_trip_cols": self.person_trip_cols,
                "candidate_cols": self.candidate_cols,
                "all_h3": self.all_h3,
                "centroid_x": self.centroid_x,
                "centroid_y": self.centroid_y,
                "static_candidate_features": self.static_candidate_features,
                "purpose_categories": self.purpose_categories,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location=None):
        state = _torch_load_checkpoint(path, map_location=map_location)
        cfg = state["model_config"]
        model = TwoInputChoiceModel(
            person_input_dim=cfg["person_input_dim"],
            candidate_input_dim=cfg["candidate_input_dim"],
            person_hidden_dim=cfg["person_hidden_dim"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout_rate=cfg["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return cls(
            model=model,
            person_scaler=state["person_scaler"],
            candidate_scaler=state["candidate_scaler"],
            person_trip_cols=state["person_trip_cols"],
            candidate_cols=state["candidate_cols"],
            all_h3=state["all_h3"],
            centroid_x=state["centroid_x"],
            centroid_y=state["centroid_y"],
            static_candidate_features=state["static_candidate_features"],
            purpose_categories=state["purpose_categories"],
        )

    def predict_from_inputs(self, person_matrix, candidate_tensor, candidate_mask=None, rng=None, return_probabilities=False, internal_mask=False, purpose=None):
        if internal_mask:
            cand_mask = self.mask.get(purpose)
            if cand_mask is None:
                cand_mask = self.mask.get("statent")
        else:
            cand_mask = candidate_mask
    
        probs = predict_two_input_proba(self.model, person_matrix, candidate_tensor, cand_mask)        
        probs = probs.cpu().numpy()
        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        if return_probabilities:
            return indices, probs
        return indices

    def _build_candidate_matrix(self, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work):        
        dynamic_matrix = build_dynamic_vector(hx=home_x, hy=home_y, wx=work_x, wy=work_y, ox=origin_x, oy=origin_y, centroid_x=self.centroid_x, centroid_y=self.centroid_y, has_work=has_work)
        static_matrix = self.static_candidate_features
        return np.concatenate([dynamic_matrix, static_matrix], axis=1)

    def predict_level0(self, home_x, home_y, work_x, work_y, origin_x, origin_y, age, sex, employed, car_availability, income_class, 
                       daily_longest_distance_from_home, daily_crowfly_total, crowfly_consumed_before_trip, trip_position_class, purpose, 
                       candidate_mask=None, rng=None):
        safe_daily_longest = float(daily_longest_distance_from_home) if np.isfinite(daily_longest_distance_from_home) and float(daily_longest_distance_from_home) >= 0.0 else 0.0
        safe_daily_total = float(daily_crowfly_total) if np.isfinite(daily_crowfly_total) and float(daily_crowfly_total) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0

        person = transform_person_trip_vector(
            age=age,
            sex=sex,
            employed=employed,
            car_availability=car_availability,
            income_class=income_class,
            daily_longest=safe_daily_longest,
            daily_total=safe_daily_total,
            consumed_before=safe_consumed,
            trip_position=safe_trip_position,
            purpose=purpose,
            purpose_categories=self.purpose_categories,
            scaler=self.person_scaler,
        )
        has_work = np.isfinite(work_x) and np.isfinite(work_y)
        candidate = self._build_candidate_matrix(home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate = transform_candidate_matrix(candidate, self.candidate_scaler)
        candidate_tensor = candidate[None, :, :]

        indices = self.predict_from_inputs(person, candidate_tensor, valid_mask=None, candidate_mask=candidate_mask, rng=rng, return_probabilities=False)
        return self.all_h3[int(indices[0])]

    def build_mask(self):
        static_pos = {name: idx for idx, name in enumerate(STATIC_CANDIDATE_FEATURES)}
        masks = dict()
        masks["statent"] = self.static_candidate_features[:, static_pos["num_statent"]] > 0.0
        # masks["employees"] = self.static_candidate_features[:, static_pos["employees"]] > 0.0
        masks["shop"] = self.static_candidate_features[:, static_pos["shop"]] > 0.0
        masks["leisure"] = self.static_candidate_features[:, static_pos["leisure"]] > 0.0
        masks["education"] = self.static_candidate_features[:, static_pos["education"]] > 0.0
        # masks["pt_access"] = self.static_candidate_features[:, static_pos["ovgk_share_none"]] < 0.999
        return masks




class MediumLevel1TwoInputWrapper:
    def __init__(self, model, person_scaler, candidate_scaler, person_trip_cols, candidate_cols, children_by_level0, level1_candidate_attributes_by_level0, purpose_categories):
        self.model = model
        self.person_scaler = person_scaler
        self.candidate_scaler = candidate_scaler
        self.person_trip_cols = list(person_trip_cols)
        self.candidate_cols = list(candidate_cols)
        self.static_feature_indices, self.dynamic_feature_indices = _candidate_column_indices(self.candidate_cols)
        self.children_by_level0 = children_by_level0
        # Compute the maximum number of level-1 children across all level-0 hexagons
        # BEFORE freezing so that every key's arrays are padded to the same length.
        # Fixed tensor shapes prevent torch.compile from recompiling on each new
        # candidate-set size.  Padded rows carry zeros, so amenity masks block them.
        _max_cand_m = max((len(a.get("children", [])) for a in level1_candidate_attributes_by_level0.values()), default=0)
        self.max_candidates = _max_cand_m
        self.level1_candidate_attributes_by_level0 = _freeze_candidate_attr_map(level1_candidate_attributes_by_level0, max_candidates=_max_cand_m)
        for level0_h3, attrs in self.level1_candidate_attributes_by_level0.items():
            static_features = attrs["static_features"]
            assert static_features.ndim == 2 and static_features.shape[1] == len(STATIC_CANDIDATE_FEATURES), (
                f"Invalid medium static_features format for level0_h3 '{level0_h3}'"
            )
            assert np.issubdtype(static_features.dtype, np.floating), (
                f"Invalid medium static_features dtype for level0_h3 '{level0_h3}'"
            )
        self.purpose_categories = [str(p) for p in purpose_categories]
        self.mask = self.build_mask()

    def save(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": {
                    "person_input_dim": self.model.person_input_dim,
                    "candidate_input_dim": self.model.candidate_input_dim,
                    "person_hidden_dim": self.model.person_hidden_dim,
                    "hidden_dims": self.model.hidden_dims,
                    "dropout_rate": self.model.dropout_rate,
                },
                "person_scaler": self.person_scaler,
                "candidate_scaler": self.candidate_scaler,
                "person_trip_cols": self.person_trip_cols,
                "candidate_cols": self.candidate_cols,
                "children_by_level0": self.children_by_level0,
                "level1_candidate_attributes_by_level0": self.level1_candidate_attributes_by_level0,
                "purpose_categories": self.purpose_categories,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location=None):
        state = _torch_load_checkpoint(path, map_location=map_location)
        cfg = state["model_config"]
        model = TwoInputChoiceModel(
            person_input_dim=cfg["person_input_dim"],
            candidate_input_dim=cfg["candidate_input_dim"],
            person_hidden_dim=cfg["person_hidden_dim"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout_rate=cfg["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return cls(
            model=model,
            person_scaler=state["person_scaler"],
            candidate_scaler=state["candidate_scaler"],
            person_trip_cols=state["person_trip_cols"],
            candidate_cols=state["candidate_cols"],
            children_by_level0=state["children_by_level0"],
            level1_candidate_attributes_by_level0=state["level1_candidate_attributes_by_level0"],
            purpose_categories=state["purpose_categories"],
        )

    def predict_from_inputs(self, person_matrix, candidate_tensor, valid_mask=None, rng=None, return_probabilities=False, 
                            internal_mask=False, purpose=None, level0_h3=None):
        if internal_mask:
            if level0_h3 is None:
                raise ValueError("level0_h3 must be provided when using internal_mask")
            level_cand_mask = self.mask.get(level0_h3)
            cand_mask = level_cand_mask.get(purpose)
            if cand_mask is None:
                cand_mask = level_cand_mask.get("statent")
        else:
            cand_mask = valid_mask
    
        probs = predict_two_input_proba(self.model, person_matrix, candidate_tensor, cand_mask)
        probs = probs.cpu().numpy()
        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        if return_probabilities:
            return indices, probs
        return indices

    def _build_candidate_matrix(self, attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work):
        dynamic_matrix = build_dynamic_vector(hx=home_x, hy=home_y, wx=work_x, wy=work_y, ox=origin_x, oy=origin_y, 
                                              centroid_x=attrs["x"], centroid_y=attrs["y"], has_work=has_work)
        static_matrix  = attrs["static_features"]
        return np.concatenate([dynamic_matrix, static_matrix], axis=1)

    def predict_level1(self, level0_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, age, daily_longest_distance_from_home, daily_crowfly_total, crowfly_consumed_before_trip, trip_position_class, sex, employed, car_availability, income_class, purpose, max_utility=False, rng=None, **kwargs):
        attrs = self.level1_candidate_attributes_by_level0.get(level0_h3)
        if attrs is None:
            raise KeyError(f"Unknown level0_h3 '{level0_h3}' in medium model candidates")

        children = attrs["children"]
        n_children = len(children)
        if n_children == 0:
            raise RuntimeError(f"No level1 candidates available for level0_h3 '{level0_h3}'")

        safe_daily_longest = float(daily_longest_distance_from_home) if np.isfinite(daily_longest_distance_from_home) and float(daily_longest_distance_from_home) >= 0.0 else 0.0
        safe_daily_total = float(daily_crowfly_total) if np.isfinite(daily_crowfly_total) and float(daily_crowfly_total) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0

        person = transform_person_trip_vector(
            age=age,
            sex=sex,
            employed=employed,
            car_availability=car_availability,
            income_class=income_class,
            daily_longest=safe_daily_longest,
            daily_total=safe_daily_total,
            consumed_before=safe_consumed,
            trip_position=safe_trip_position,
            purpose=purpose,
            purpose_categories=self.purpose_categories,
            scaler=self.person_scaler,
        )

        candidate = self._build_candidate_matrix(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y)
        candidate = transform_candidate_matrix(candidate, self.candidate_scaler)

        candidate_tensor = candidate[None, :, :].astype(np.float32)
        valid_mask = np.ones((1, n_children), dtype=bool)

        _, probs = self.predict_from_inputs(person, candidate_tensor, valid_mask, rng=rng, return_probabilities=True)
        probs = probs[0, :n_children]
        company_mask = np.asarray(attrs["num_statent"], dtype=np.float64)[:n_children] > 0.0
        if not np.any(company_mask):
            raise RuntimeError(f"No level1 candidates with num_statent > 0 for level0_h3 '{level0_h3}'")

        probs = probs * company_mask.astype(np.float64)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0:
            raise RuntimeError(f"No valid probability mass on non-empty level1 candidates for level0_h3 '{level0_h3}'")
        probs = probs / prob_sum

        chosen_idx = int(np.random.choice(n_children, p=probs)) if rng is None else int(rng.choice(n_children, p=probs))
        return {"level1_h3": children[chosen_idx], "choice_index": chosen_idx, "candidates": children, "probabilities": probs}

    def build_mask(self):                
        masks = dict()
        for level0_h3, attrs in self.level1_candidate_attributes_by_level0.items():
            masks[level0_h3] = dict() 
            masks[level0_h3]["statent"] = attrs["num_statent"] > 0.0
            # masks[level0_h3]["employees"] = self.static_candidate_features[:, static_pos["employees"]] > 0.0
            masks[level0_h3]["shop"] = attrs["shop"] > 0.0
            masks[level0_h3]["leisure"] = attrs["leisure"] > 0.0
            masks[level0_h3]["education"] = attrs["education"] > 0.0
            # masks[level0_h3]["pt_access"] = attrs["ovgk_share_none"] < 0.999
        return masks




class DetailedLevel2TwoInputWrapper:
    def __init__(self, model, person_scaler, candidate_scaler, person_trip_cols, candidate_cols, children_by_level1, level2_candidate_attributes_by_level1, purpose_categories):
        self.model = model
        self.person_scaler = person_scaler
        self.candidate_scaler = candidate_scaler
        self.person_trip_cols = list(person_trip_cols)
        self.candidate_cols = list(candidate_cols)
        self.static_feature_indices, self.dynamic_feature_indices = _candidate_column_indices(self.candidate_cols)
        self.children_by_level1 = children_by_level1
        # Same fixed-shape padding strategy as the medium wrapper.
        _max_cand_d = max((len(a.get("children", [])) for a in level2_candidate_attributes_by_level1.values()), default=0)
        self.max_candidates = _max_cand_d
        self.level2_candidate_attributes_by_level1 = _freeze_candidate_attr_map(level2_candidate_attributes_by_level1, max_candidates=_max_cand_d)
        for key, attrs in self.level2_candidate_attributes_by_level1.items():
            static_features = attrs["static_features"]
            assert static_features.ndim == 2 and static_features.shape[1] == len(STATIC_CANDIDATE_FEATURES), (
                f"Invalid detailed static_features format for key {key}"
            )
            assert np.issubdtype(static_features.dtype, np.floating), (
                f"Invalid detailed static_features dtype for key {key}"
            )
        self.purpose_categories = [str(p) for p in purpose_categories]
        self.mask = self.build_mask()

    def save(self, path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": {
                    "person_input_dim": self.model.person_input_dim,
                    "candidate_input_dim": self.model.candidate_input_dim,
                    "person_hidden_dim": self.model.person_hidden_dim,
                    "hidden_dims": self.model.hidden_dims,
                    "dropout_rate": self.model.dropout_rate,
                },
                "person_scaler": self.person_scaler,
                "candidate_scaler": self.candidate_scaler,
                "person_trip_cols": self.person_trip_cols,
                "candidate_cols": self.candidate_cols,
                "children_by_level1": self.children_by_level1,
                "level2_candidate_attributes_by_level1": self.level2_candidate_attributes_by_level1,
                "purpose_categories": self.purpose_categories,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location=None):
        state = _torch_load_checkpoint(path, map_location=map_location)
        cfg = state["model_config"]
        model = TwoInputChoiceModel(
            person_input_dim=cfg["person_input_dim"],
            candidate_input_dim=cfg["candidate_input_dim"],
            person_hidden_dim=cfg["person_hidden_dim"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout_rate=cfg["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return cls(
            model=model,
            person_scaler=state["person_scaler"],
            candidate_scaler=state["candidate_scaler"],
            person_trip_cols=state["person_trip_cols"],
            candidate_cols=state["candidate_cols"],
            children_by_level1=state["children_by_level1"],
            level2_candidate_attributes_by_level1=state["level2_candidate_attributes_by_level1"],
            purpose_categories=state["purpose_categories"],
        )

    def predict_from_inputs(self, person_matrix, candidate_tensor, valid_mask=None, rng=None, return_probabilities=False, 
                            internal_mask=False, purpose=None, level0_h3=None, level1_h3=None):
        if internal_mask:
            if level0_h3 is None or level1_h3 is None:
                raise ValueError("level0_h3 and level1_h3 must be provided when using internal_mask")
            key = (level0_h3, level1_h3)
            level_cand_mask = self.mask.get(key)
            cand_mask = level_cand_mask.get(purpose)
            if cand_mask is None:
                cand_mask = level_cand_mask.get("statent")
        else:
            cand_mask = valid_mask

        probs = predict_two_input_proba(self.model, person_matrix, candidate_tensor, cand_mask)
        probs = probs.cpu().numpy()
        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        if return_probabilities:
            return indices, probs
        return indices

    def _build_candidate_matrix(self, attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work):
        dynamic_matrix = build_dynamic_vector(hx=home_x, hy=home_y, wx=work_x, wy=work_y, ox=origin_x, oy=origin_y, 
                                              centroid_x=attrs["x"], centroid_y=attrs["y"], has_work=has_work)
        static_matrix  = attrs["static_features"]
        return np.concatenate([dynamic_matrix, static_matrix], axis=1)

    def predict_level2(self, level0_h3, level1_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, age, daily_longest_distance_from_home, daily_crowfly_total, crowfly_consumed_before_trip, trip_position_class, sex, employed, car_availability, income_class, purpose, max_utility=False, rng=None, **kwargs):
        key = (level0_h3, level1_h3)
        attrs = self.level2_candidate_attributes_by_level1.get(key)
        if attrs is None:
            raise KeyError(f"Unknown key {key} in detailed model candidates")

        children = attrs["children"]
        n_children = len(children)
        if n_children == 0:
            raise RuntimeError(f"No level2 candidates available for key {key}")

        safe_daily_longest = float(daily_longest_distance_from_home) if np.isfinite(daily_longest_distance_from_home) and float(daily_longest_distance_from_home) >= 0.0 else 0.0
        safe_daily_total = float(daily_crowfly_total) if np.isfinite(daily_crowfly_total) and float(daily_crowfly_total) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0

        person = transform_person_trip_vector(
            age=age,
            sex=sex,
            employed=employed,
            car_availability=car_availability,
            income_class=income_class,
            daily_longest=safe_daily_longest,
            daily_total=safe_daily_total,
            consumed_before=safe_consumed,
            trip_position=safe_trip_position,
            purpose=purpose,
            purpose_categories=self.purpose_categories,
            scaler=self.person_scaler,
        )

        candidate = self._build_candidate_matrix(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y)
        candidate = transform_candidate_matrix(candidate, self.candidate_scaler)

        candidate_tensor = candidate[None, :, :].astype(np.float32)
        valid_mask = np.ones((1, n_children), dtype=bool)

        _, probs = self.predict_from_inputs(person, candidate_tensor, valid_mask, rng=rng, return_probabilities=True)
        probs = probs[0, :n_children]
        company_mask = np.asarray(attrs["num_statent"], dtype=np.float64)[:n_children] > 0.0
        if not np.any(company_mask):
            raise RuntimeError(f"No level2 candidates with num_statent > 0 for key {key}")

        probs = probs * company_mask.astype(np.float64)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0:
            raise RuntimeError(f"No valid probability mass on non-empty level2 candidates for key {key}")
        probs = probs / prob_sum

        chosen_idx = int(np.random.choice(n_children, p=probs)) if rng is None else int(rng.choice(n_children, p=probs))
        return {"level2_h3": children[chosen_idx], "choice_index": chosen_idx, "candidates": children, "probabilities": probs}

    def build_mask(self):                
        masks = dict()
        for key, attrs in self.level2_candidate_attributes_by_level1.items():
            masks[key] = dict() 
            masks[key]["statent"] = attrs["num_statent"] > 0.0
            masks[key]["shop"] = attrs["shop"] > 0.0
            masks[key]["leisure"] = attrs["leisure"] > 0.0
            masks[key]["education"] = attrs["education"] > 0.0
        return masks




class LocationChoiceModelWrapper:
    logger: logging.Logger = logging.getLogger("synpp: LocationChoiceModelWrapper")

    def __init__(self, coarse_wrapper:CoarseLevel0TwoInputWrapper, medium_wrapper:MediumLevel1TwoInputWrapper, detailed_wrapper:DetailedLevel2TwoInputWrapper, optimize=True):        
        self.wrappers = dict(coarse=coarse_wrapper, medium=medium_wrapper, detailed=detailed_wrapper)        
        self.same_person_scaler = self.check_if_similar_person_scalers()
        self.build_internal_state()
        assert self.same_person_scaler, ("Person scalers differ across model levels; this may lead to inconsistent predictions if the same person features are used at multiple levels. "+
                                         "Please verify that the scalers are intended to be different, in that case, you need to modify the predict method.")
        self._assert_wrapper_compatibility()
        if optimize:
            self.log("Optimizing location choice model wrappers with torch.compile (if available)")
            self.optimize_wrapper()            

    #################### Static and class methods for loading/optimizing ####################
    def build_internal_state(self):
        self._optimized = False
        self._last_person_id = None
        self._last_person_vector = None
        self.destination_index = None
        self.destination_fallback = None
        self._purpose_categories = [str(p) for p in self.c.purpose_categories]
        self._purpose_identity = np.eye(len(self._purpose_categories), dtype=np.float32)
        self._purpose_index = {p: i for i, p in enumerate(self._purpose_categories)} 

    @classmethod
    def load(cls, coarse_path:str, medium_path:str, detailed_path:str, map_location=None, optimize=True):
        cls.log("Loading models")
        coarse_wrapper = CoarseLevel0TwoInputWrapper.load(coarse_path, map_location=map_location)
        medium_wrapper = MediumLevel1TwoInputWrapper.load(medium_path, map_location=map_location)
        detailed_wrapper = DetailedLevel2TwoInputWrapper.load(detailed_path, map_location=map_location)
        return cls(coarse_wrapper=coarse_wrapper, medium_wrapper=medium_wrapper, detailed_wrapper=detailed_wrapper, optimize=optimize)

    @classmethod
    def build(cls, context, map_location=None, optimize=True):
        coarse_model_path, person_scaler_path, coarse_candidate_scaler_path = context.stage("synthesis.population.spatial.secondary.locations_v2.coarse_model")
        medium_model_path, medium_candidate_scaler_path = context.stage("synthesis.population.spatial.secondary.locations_v2.medium_model")
        detailed_model_path, detailed_candidate_scaler_path = context.stage("synthesis.population.spatial.secondary.locations_v2.detailed_model")
        obj = cls.load(coarse_path=coarse_model_path, medium_path=medium_model_path, detailed_path=detailed_model_path, map_location=map_location, optimize=optimize)
        obj.build_candidates_and_trees(context)
        return obj
    
    def build_candidates_and_trees(self, context):
        h3_data, h3_geo, h3_tree = context.stage("synthesis.population.spatial.secondary.locations_v2.h3")
        self.destination_index, self.destination_fallback = _prepare_destination_level2_index(context, h3_data)
        # self.reverse_h3_tree = _reverse_tree(h3_tree)

    @classmethod
    def log(cls, message, level=logging.INFO):
        cls.logger.log(level, "\t\t" + message)

    def optimize_wrapper(self, enable_compile=True, compile_mode="max-autotune"):
        for key in self.wrappers:
            self.wrappers[key] = self._optimize_wrapper(self.wrappers[key], enable_compile=enable_compile)
        self._optimized = True

    def _optimize_wrapper(self, wrapper, enable_compile=True, compile_mode="max-autotune"):
        wrapper.model = wrapper.model.float()
        wrapper.model.eval()

        if not enable_compile:
            return wrapper

        if not hasattr(torch, "compile"):
            self.log("torch.compile not available in this PyTorch build; continuing without compile", level=logging.WARNING)
            return wrapper

        try:
            wrapper.model = torch.compile(wrapper.model, mode=compile_mode)
        except Exception as e:
            self.log(f"torch.compile failed ({e}); continuing with eager model", level=logging.WARNING)

        return wrapper

    @property
    def m(self):
        return self.wrappers["medium"]
        
    @property
    def d(self):
        return self.wrappers["detailed"]
        
    @property
    def c(self):
        return self.wrappers["coarse"]

    def _same_scaler(self, a, b):
        return (a.get_params() == b.get_params() and np.allclose(a.quantiles_, b.quantiles_) and np.allclose(a.references_, b.references_))

    def check_if_similar_person_scalers(self):        
        return (self._same_scaler(self.c.person_scaler, self.m.person_scaler) and self._same_scaler(self.c.person_scaler, self.d.person_scaler))

    def _assert_wrapper_compatibility(self):
        assert self.c.candidate_cols == self.m.candidate_cols == self.d.candidate_cols, (
            "Candidate feature columns differ across wrapper levels."
        )
        assert self.c.candidate_cols == list(CANDIDATE_FEATURES), (
            "Candidate feature columns do not match two_input_features.CANDIDATE_FEATURES"
        )
        assert self.c.person_trip_cols == self.m.person_trip_cols == self.d.person_trip_cols, (
            "Person feature columns differ across wrapper levels."
        )
        assert self.c.purpose_categories == self.m.purpose_categories == self.d.purpose_categories, (
            "Purpose categories differ across wrapper levels."
        )
        assert int(self.c.model.person_input_dim) == int(self.m.model.person_input_dim) == int(self.d.model.person_input_dim), (
            "Person input dimensions differ across wrapper levels."
        )
        assert len(self.c.person_trip_cols) == int(self.c.model.person_input_dim), (
            "Person feature column count does not match model input dimension."
        )
        assert int(self.c.model.candidate_input_dim) == int(self.m.model.candidate_input_dim) == int(self.d.model.candidate_input_dim), (
            "Candidate input dimensions differ across wrapper levels."
        )
        assert int(self.c.model.candidate_input_dim) == len(CANDIDATE_FEATURES), (
            "Candidate input dimension does not match CANDIDATE_FEATURES."
        )

        # Coarse wrapper static candidate features should be persisted and aligned with candidate columns.
        assert getattr(self.c, "static_candidate_features", None) is not None, "Missing coarse static_candidate_features"
        assert self.c.static_candidate_features.ndim == 2, "coarse static_candidate_features must be a 2D matrix"
        assert self.c.static_candidate_features.shape[1] == len(STATIC_CANDIDATE_FEATURES), (
            "coarse static_candidate_features width mismatch with candidate columns"
        )

        # Medium wrapper pre-frozen attributes should include static features with coherent dimensions.
        for level0_h3, attrs in self.m.level1_candidate_attributes_by_level0.items():
            assert "static_features" in attrs and attrs["static_features"].ndim == 2, (
                f"Missing/invalid medium static_features for level0_h3 '{level0_h3}'"
            )
            assert attrs["static_features"].shape[1] == len(STATIC_CANDIDATE_FEATURES), (
                f"Medium static_features width mismatch for level0_h3 '{level0_h3}'"
            )
            assert len(attrs["children"]) == attrs["static_features"].shape[0], (
                f"Medium children/static_features length mismatch for level0_h3 '{level0_h3}'"
            )

        # Detailed wrapper pre-frozen attributes should include static features with coherent dimensions.
        for key, attrs in self.d.level2_candidate_attributes_by_level1.items():
            assert "static_features" in attrs and attrs["static_features"].ndim == 2, (
                f"Missing/invalid detailed static_features for key {key}"
            )
            assert attrs["static_features"].shape[1] == len(STATIC_CANDIDATE_FEATURES), (
                f"Detailed static_features width mismatch for key {key}"
            )
            assert len(attrs["children"]) == attrs["static_features"].shape[0], (
                f"Detailed children/static_features length mismatch for key {key}"
            )

    def assert_ready_for_prediction(self):
        assert self.destination_index is not None and self.destination_fallback is not None, (
            "Destination pools are not initialized. Call build_candidates_and_trees(context) before predict()."
        )
        # assert self.reverse_h3_tree is not None, (
        #     "H3 tree is not initialized. Call build_candidates_and_trees(context) before predict()."
        # )

    ##################### helper methods for prediction steps #####################    
    def _build_shared_person_vector(self, person_id, age, sex, employed, car_availability, income_class, daily_longest_distance_from_home,
                                    daily_crowfly_total, crowfly_consumed_before_trip, trip_position_class, purpose):
        # NOTE: The person vector includes trip-varying features (purpose, daily_longest,
        # daily_total, consumed_before, trip_position).  Caching on person_id alone is
        # incorrect for persons with multiple secondary trips — each trip has its own
        # values for all of those features.  We therefore always recompute the vector.
        purpose_hot = self._purpose_identity[self._purpose_index[purpose]]
        return transform_person_trip_vector(age=age, sex=sex, employed=employed, car_availability=car_availability, income_class=income_class,
                                            daily_longest=daily_longest_distance_from_home, daily_total=daily_crowfly_total, consumed_before=crowfly_consumed_before_trip, 
                                            trip_position=trip_position_class, purpose_hot=purpose_hot, scaler=self.c.person_scaler)

    def _predict_level0_from_person(self, person, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work, rng, purpose):
        candidate = self.c._build_candidate_matrix(home_x=home_x, home_y=home_y, work_x=work_x, work_y=work_y, origin_x=origin_x, origin_y=origin_y, has_work=has_work)
        candidate = transform_candidate_matrix(candidate, self.c.candidate_scaler)
        candidate_tensor = candidate[None, :, :]
        indices = self.c.predict_from_inputs(person, candidate_tensor, rng=rng, return_probabilities=False, internal_mask=True, purpose=purpose)
        return self.c.all_h3[indices[0]]

    def _predict_level1_from_person(self, person, level0_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work, purpose, rng):
        attrs = self.m.level1_candidate_attributes_by_level0.get(level0_h3)
        if attrs is None:
            raise KeyError(f"Unknown level0_h3 '{level0_h3}' in medium model candidates")

        children = attrs["children"]
        n_children = len(children)  # true (unpadded) count
        if n_children == 0:
            raise RuntimeError(f"No level1 candidates available for level0_h3 '{level0_h3}'")

        candidate = self.m._build_candidate_matrix(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate = transform_candidate_matrix(candidate, self.m.candidate_scaler)
        candidate_tensor = candidate[None, :, :]      
        idx = self.m.predict_from_inputs(person, candidate_tensor, rng=rng, return_probabilities=False, internal_mask=True, purpose=purpose, level0_h3=level0_h3)
        # Clamp to valid range: padded positions should never be chosen (they are
        # masked with -1e9 utilities), but floating-point edge cases can occur.
        safe_idx = min(int(idx[0]), n_children - 1)
        return children[safe_idx]

    def _predict_level2_from_person(self, person, level0_h3, level1_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work, purpose, rng):
        key = (level0_h3, level1_h3)
        attrs = self.d.level2_candidate_attributes_by_level1.get(key)
        if attrs is None:
            raise KeyError(f"Unknown key {key} in detailed model candidates")

        children = attrs["children"]
        n_children = len(children)  # true (unpadded) count
        if n_children == 0:
            raise RuntimeError(f"No level2 candidates available for key {key}")

        candidate = self.d._build_candidate_matrix(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate = transform_candidate_matrix(candidate, self.d.candidate_scaler)
        candidate_tensor = candidate[None, :, :].astype(np.float32)
        idx = self.d.predict_from_inputs(person, candidate_tensor, rng=rng, return_probabilities=False, internal_mask=True, purpose=purpose, 
                                        level0_h3=level0_h3, level1_h3=level1_h3)
        # Clamp to valid range: padded positions should never be chosen (they are
        # masked with -1e9 utilities), but floating-point edge cases can occur.
        safe_idx = min(int(idx[0]), n_children - 1)
        return children[safe_idx]

    ##################### Main prediction method that runs the full three-level prediction process #####################

    def predict(self, person_id, home_x, home_y, work_x, work_y, origin_x, origin_y, age, daily_longest_distance_from_home, daily_crowfly_total, crowfly_consumed_before_trip, 
                trip_position_class, sex, employed, car_availability, income_class, purpose, has_work, has_education, rng):

        person = self._build_shared_person_vector(person_id = person_id, age=age, sex=sex, employed=employed, car_availability=car_availability, income_class=income_class,
                                                  daily_longest_distance_from_home=daily_longest_distance_from_home,trip_position_class=trip_position_class,
                                                  daily_crowfly_total=daily_crowfly_total, crowfly_consumed_before_trip=crowfly_consumed_before_trip, purpose=purpose)

        level0_h3 = self._predict_level0_from_person(person=person, home_x=home_x, home_y=home_y, work_x=work_x, work_y=work_y, origin_x=origin_x, origin_y=origin_y, 
                                                     has_work=has_work, rng=rng, purpose=purpose)

        level1_h3 = self._predict_level1_from_person(person=person, level0_h3=level0_h3, home_x=home_x, home_y=home_y, work_x=work_x,
                                                     work_y=work_y, origin_x=origin_x, origin_y=origin_y, has_work=has_work, purpose=purpose, rng=rng)

        level2_h3 = self._predict_level2_from_person(person=person, level0_h3=level0_h3, level1_h3=level1_h3, home_x=home_x, home_y=home_y, work_x=work_x,
                                                 work_y=work_y, origin_x=origin_x, origin_y=origin_y, has_work=has_work, purpose=purpose, rng=rng)

        destination_id, geom = self.sample_company_in_l2(purpose, level1_h3, level2_h3, rng, car_availability)
        return destination_id, geom


    def sample_company_in_l2(self, purpose, level1_h3, level2_h3, rng, car_availability=None):

        purpose_index = self.destination_index.get(purpose)
        if purpose_index is not None:
            pool = purpose_index.get(level2_h3)
            if pool:
                return pool[rng.randint(len(pool))]

        purpose_fallback = self.destination_fallback.get(purpose)
        if purpose_fallback is not None:
            pool = purpose_fallback.get(level1_h3)
            if pool:
                return pool[rng.randint(len(pool))]
        
        purpose_fallback = self.destination_fallback.get('other')
        if purpose_fallback is not None:
            pool = purpose_fallback.get(level1_h3)
            if pool:
                self.log(f"Using fallback pool for level1_h3 '{level1_h3}' and purpose '{purpose}' with {len(pool)} candidates", level=logging.WARNING) 
                return pool[rng.randint(len(pool))]
            
        raise RuntimeError(
            f"No candidates in destination pool for purpose '{purpose}', "
            f"level1_h3 '{level1_h3}', level2_h3 '{level2_h3}'"
        )
