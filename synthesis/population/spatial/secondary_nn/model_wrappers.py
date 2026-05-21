import numpy as np
from .location_helpers import _prepare_destination_level2_index, _reverse_tree
import torch
from shapely.geometry import Point
import logging
from .feature_encoding import (CANDIDATE_FEATURES, STATIC_CANDIDATE_FEATURES, DYNAMIC_CANDIDATE_FEATURES, N_PERSON_STATIC,
                               transform_candidate_static_matrix, transform_candidate_dynamic_matrix, transform_person_static_vector, 
                               transform_person_dynamic_vector, transform_person_trip_vector)

from .choice_model import NeuralChoiceModel, predict_choice_proba
from .hierarchical_utils import build_dynamic_vector, ORIGIN_PURPOSE_REMAP

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


def _freeze_candidate_attr_map(attr_map, max_candidates=None, candidate_static_scaler=None):
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
        # Build static feature matrix in STATIC_CANDIDATE_FEATURES order (scaled block first, passthrough block after).
        # If attrs is already frozen (loaded from a checkpoint) it already has a "static_features" matrix; reuse it
        # directly to avoid a KeyError on the individual per-feature keys that are no longer stored.
        if "static_features" in attrs:
            static_features = np.asarray(attrs["static_features"], dtype=np.float64)
        else:
            static_features = np.column_stack([
                _to_float_array(attrs[name]) for name in STATIC_CANDIDATE_FEATURES
            ]).astype(np.float64)  # [N, 17]

        # Pad to max_candidates so every call produces the same tensor shape — prevents
        # torch.compile from recompiling on each new candidate-set size.  Padded rows
        # carry zeros, so amenity masks (num_statent > 0, etc.) are False for them.
        n_children = len(attrs["children"])  # true (unpadded) candidate count
        pad = max(0, (max_candidates - len(x))) if max_candidates is not None else 0
        if pad > 0:
            z1d = np.zeros(pad, dtype=np.float64)
            x = np.concatenate([x, z1d])
            y = np.concatenate([y, z1d])
            static_features = np.vstack([static_features, np.zeros((pad, static_features.shape[1]), dtype=np.float64)])

        _ss = transform_candidate_static_matrix(static_features, candidate_static_scaler).astype(np.float32) if candidate_static_scaler is not None else static_features.astype(np.float32)
        frozen[key] = {
            "children": list(attrs["children"]),
            "n_children": n_children,
            "x": x,
            "y": y,
            # Unscaled matrix kept so the torch tensor can always be rebuilt (e.g. after
            # loading a checkpoint without a GPU) via transform_candidate_static_matrix.
            "static_features": static_features,
            "scaled_static_features_torch": torch.tensor(_ss, dtype=torch.float32).unsqueeze(0),
        }
    return frozen


class RegionalChoiceWrapper:
    def __init__(self, model, person_static_scaler, person_dynamic_scaler, candidate_static_scaler, candidate_dynamic_scaler, 
                 person_trip_cols, candidate_cols, all_h3, centroid_x, centroid_y, static_candidate_features, purpose_categories):
        self.model = model
        self.person_static_scaler    = person_static_scaler
        self.person_dynamic_scaler   = person_dynamic_scaler
        self.candidate_static_scaler  = candidate_static_scaler
        self.candidate_dynamic_scaler = candidate_dynamic_scaler
        self.person_trip_cols = list(person_trip_cols)
        self.candidate_cols = list(candidate_cols)
        self.all_h3 = list(all_h3)
        self.centroid_x = np.asarray(centroid_x, dtype=np.float64)
        self.centroid_y = np.asarray(centroid_y, dtype=np.float64)
        self.static_candidate_features = np.asarray(static_candidate_features, dtype=np.float32)
        # Pre-scale static features once at init — avoids per-trip scaler.transform() calls.
        scaled_static_candidate_features = transform_candidate_static_matrix(self.static_candidate_features, self.candidate_static_scaler).astype(np.float32)
        self.scaled_static_features_torch = torch.tensor(scaled_static_candidate_features, dtype=torch.float32).unsqueeze(0)  # [1, N, 13]
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
                    "hidden_dim": self.model.hidden_dim,
                    "dropout_rate": self.model.dropout_rate,
                },
                "person_static_scaler":    self.person_static_scaler,
                "person_dynamic_scaler":   self.person_dynamic_scaler,
                "candidate_static_scaler":  self.candidate_static_scaler,
                "candidate_dynamic_scaler": self.candidate_dynamic_scaler,
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
        model = NeuralChoiceModel(
            person_input_dim=cfg["person_input_dim"],
            candidate_input_dim=cfg["candidate_input_dim"],
            person_hidden_dim=cfg["person_hidden_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout_rate=cfg["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return cls(
            model=model,
            person_static_scaler=state["person_static_scaler"],
            person_dynamic_scaler=state["person_dynamic_scaler"],
            candidate_static_scaler=state["candidate_static_scaler"],
            candidate_dynamic_scaler=state["candidate_dynamic_scaler"],
            person_trip_cols=state["person_trip_cols"],
            candidate_cols=state["candidate_cols"],
            all_h3=state["all_h3"],
            centroid_x=state["centroid_x"],
            centroid_y=state["centroid_y"],
            static_candidate_features=state["static_candidate_features"],
            purpose_categories=state["purpose_categories"],
        )

    def predict_from_inputs(self, person_matrix, candidate_static, candidate_dynamic, candidate_mask=None, rng=None, return_probabilities=False, internal_mask=False, purpose=None):
        if internal_mask:
            cand_mask = self.mask.get(purpose)
            if cand_mask is None:
                cand_mask = self.mask.get("statent")
        else:
            cand_mask = candidate_mask
        p_static  = person_matrix[:, :N_PERSON_STATIC]
        p_dynamic = person_matrix[:, N_PERSON_STATIC:]
        probs = predict_choice_proba(self.model, p_static, p_dynamic, candidate_static, candidate_dynamic, cand_mask)
        probs = probs.cpu().numpy()
        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        if return_probabilities:
            return indices, probs
        return indices

    def _build_candidate_dynamic(self, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work):
        return build_dynamic_vector(hx=home_x, hy=home_y, wx=work_x, wy=work_y, ox=origin_x, oy=origin_y, centroid_x=self.centroid_x, centroid_y=self.centroid_y, has_work=has_work)

    def predict_level0(self, home_x, home_y, work_x, work_y, origin_x, origin_y, age, sex, employed, car_availability, income_class,
                       daily_longest_distance_from_home, daily_crowfly_total, daily_longest_distance_from_work,
                       crowfly_consumed_before_trip, trip_position_class, departure_time_normalized, target_distance, activity_chain_vector, purpose, origin_purpose,
                       candidate_mask=None, rng=None):
        safe_daily_longest = float(daily_longest_distance_from_home) if np.isfinite(daily_longest_distance_from_home) and float(daily_longest_distance_from_home) >= 0.0 else 0.0
        safe_daily_total = float(daily_crowfly_total) if np.isfinite(daily_crowfly_total) and float(daily_crowfly_total) >= 0.0 else 0.0
        safe_daily_longest_work = float(daily_longest_distance_from_work) if np.isfinite(daily_longest_distance_from_work) and float(daily_longest_distance_from_work) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0
        safe_departure_time = float(departure_time_normalized) if np.isfinite(departure_time_normalized) else 0.5
        safe_target_distance = float(target_distance) if np.isfinite(target_distance) and float(target_distance) >= 0.0 else 0.0

        purpose_idx = self.purpose_categories.index(str(purpose)) if str(purpose) in self.purpose_categories else 0
        purpose_hot = np.eye(len(self.purpose_categories), dtype=np.float32)[purpose_idx]
        remapped_origin = ORIGIN_PURPOSE_REMAP.get(str(origin_purpose), str(origin_purpose))
        origin_purpose_idx = self.purpose_categories.index(remapped_origin) if remapped_origin in self.purpose_categories else -1
        origin_purpose_hot = np.eye(len(self.purpose_categories), dtype=np.float32)[origin_purpose_idx] if origin_purpose_idx >= 0 else np.zeros(len(self.purpose_categories), dtype=np.float32)
        person = transform_person_trip_vector(age=age, income_class=income_class, daily_longest=safe_daily_longest, daily_total=safe_daily_total,
            daily_longest_work=safe_daily_longest_work, activity_chain_vector=activity_chain_vector,
            sex=sex, employed=employed, car_availability=car_availability,
            consumed_before=safe_consumed, trip_position=safe_trip_position, departure_time=safe_departure_time, target_distance=safe_target_distance,
            purpose_hot=purpose_hot, origin_purpose_hot=origin_purpose_hot,
            static_scaler=self.person_static_scaler, dynamic_scaler=self.person_dynamic_scaler)
        has_work = np.isfinite(work_x) and np.isfinite(work_y)
        candidate_dynamic        = self._build_candidate_dynamic(home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate_dynamic_scaled = transform_candidate_dynamic_matrix(candidate_dynamic, self.candidate_dynamic_scaler)[None, :, :]
        candidate_static_scaled  = self.scaled_static_features_torch
        indices = self.predict_from_inputs(person, candidate_static_scaled, candidate_dynamic_scaled, candidate_mask=candidate_mask, rng=rng, return_probabilities=False)
        return self.all_h3[int(indices[0])]

    def build_mask(self):
        static_pos = {name: idx for idx, name in enumerate(STATIC_CANDIDATE_FEATURES)}
        masks = dict()
        masks["statent"]   = torch.tensor(self.static_candidate_features[:, static_pos["num_statent"]] > 0.0, dtype=torch.bool)
        masks["shop"]      = torch.tensor(self.static_candidate_features[:, static_pos["shop"]]        > 0.0, dtype=torch.bool)
        masks["leisure"]   = torch.tensor(self.static_candidate_features[:, static_pos["leisure"]]     > 0.0, dtype=torch.bool)
        masks["education_secondary"] = torch.tensor(self.static_candidate_features[:, static_pos["education"]]   > 0.0, dtype=torch.bool)
        return masks




class DistrictChoiceWrapper:
    logger: logging.Logger = logging.getLogger("synpp: HierarchicalLocationChoiceModel")

    def __init__(self, model, person_static_scaler, person_dynamic_scaler, candidate_static_scaler, candidate_dynamic_scaler, person_trip_cols, candidate_cols, 
                 children_by_level0, level1_candidate_attributes_by_level0, purpose_categories):
        self.model = model
        self.person_static_scaler    = person_static_scaler
        self.person_dynamic_scaler   = person_dynamic_scaler
        self.candidate_static_scaler  = candidate_static_scaler
        self.candidate_dynamic_scaler = candidate_dynamic_scaler
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
        self.level1_candidate_attributes_by_level0 = _freeze_candidate_attr_map(level1_candidate_attributes_by_level0, max_candidates=_max_cand_m, candidate_static_scaler=candidate_static_scaler)
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
                    "hidden_dim": self.model.hidden_dim,
                    "dropout_rate": self.model.dropout_rate,
                },
                "person_static_scaler":    self.person_static_scaler,
                "person_dynamic_scaler":   self.person_dynamic_scaler,
                "candidate_static_scaler":  self.candidate_static_scaler,
                "candidate_dynamic_scaler": self.candidate_dynamic_scaler,
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
        model = NeuralChoiceModel(
            person_input_dim=cfg["person_input_dim"],
            candidate_input_dim=cfg["candidate_input_dim"],
            person_hidden_dim=cfg["person_hidden_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout_rate=cfg["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return cls(
            model=model,
            person_static_scaler=state["person_static_scaler"],
            person_dynamic_scaler=state["person_dynamic_scaler"],
            candidate_static_scaler=state["candidate_static_scaler"],
            candidate_dynamic_scaler=state["candidate_dynamic_scaler"],
            person_trip_cols=state["person_trip_cols"],
            candidate_cols=state["candidate_cols"],
            children_by_level0=state["children_by_level0"],
            level1_candidate_attributes_by_level0=state["level1_candidate_attributes_by_level0"],
            purpose_categories=state["purpose_categories"],
        )

    def predict_from_inputs(self, person_matrix, candidate_static, candidate_dynamic, valid_mask=None, rng=None, return_probabilities=False,
                            internal_mask=False, purpose=None, level0_h3=None):
        if internal_mask:
            if level0_h3 is None:
                raise ValueError("level0_h3 must be provided when using internal_mask")
            level_cand_mask = self.mask.get(level0_h3)
            if level_cand_mask is None:
                self.logger.warning(f"No candidate mask found for level0_h3 '{level0_h3}'; no internal mask will be applied")
                cand_mask = None
            else:
                cand_mask = level_cand_mask.get(purpose)
                if cand_mask is None:
                    cand_mask = level_cand_mask.get("statent")
        else:
            cand_mask = valid_mask
        p_static  = person_matrix[:, :N_PERSON_STATIC]
        p_dynamic = person_matrix[:, N_PERSON_STATIC:]
        probs = predict_choice_proba(self.model, p_static, p_dynamic, candidate_static, candidate_dynamic, cand_mask)
        probs = probs.cpu().numpy()
        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        if return_probabilities:
            return indices, probs
        return indices

    def _build_candidate_dynamic(self, attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work):
        return build_dynamic_vector(hx=home_x, hy=home_y, wx=work_x, wy=work_y, ox=origin_x, oy=origin_y,
                                    centroid_x=attrs["x"], centroid_y=attrs["y"], has_work=has_work)

    def build_mask(self):
        _si = {name: i for i, name in enumerate(STATIC_CANDIDATE_FEATURES)}
        masks = dict()
        for level0_h3, attrs in self.level1_candidate_attributes_by_level0.items():
            sf = attrs["static_features"]
            masks[level0_h3] = {
                "statent":             sf[:, _si["num_statent"]] > 0.0,
                "shop":                sf[:, _si["shop"]]        > 0.0,
                "leisure":             sf[:, _si["leisure"]]     > 0.0,
                "education_secondary": sf[:, _si["education"]]   > 0.0,
            }
        return masks




class LocalChoiceWrapper:
    def __init__(self, model, person_static_scaler, person_dynamic_scaler, candidate_static_scaler, candidate_dynamic_scaler, person_trip_cols, candidate_cols, children_by_level1, level2_candidate_attributes_by_level1, purpose_categories):
        self.model = model
        self.person_static_scaler    = person_static_scaler
        self.person_dynamic_scaler   = person_dynamic_scaler
        self.candidate_static_scaler  = candidate_static_scaler
        self.candidate_dynamic_scaler = candidate_dynamic_scaler
        self.person_trip_cols = list(person_trip_cols)
        self.candidate_cols = list(candidate_cols)
        self.static_feature_indices, self.dynamic_feature_indices = _candidate_column_indices(self.candidate_cols)
        self.children_by_level1 = children_by_level1
        # Same fixed-shape padding strategy as the medium wrapper.
        _max_cand_d = max((len(a.get("children", [])) for a in level2_candidate_attributes_by_level1.values()), default=0)
        self.max_candidates = _max_cand_d
        self.level2_candidate_attributes_by_level1 = _freeze_candidate_attr_map(level2_candidate_attributes_by_level1, max_candidates=_max_cand_d, candidate_static_scaler=candidate_static_scaler)
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
                    "hidden_dim": self.model.hidden_dim,
                    "dropout_rate": self.model.dropout_rate,
                },
                "person_static_scaler":    self.person_static_scaler,
                "person_dynamic_scaler":   self.person_dynamic_scaler,
                "candidate_static_scaler":  self.candidate_static_scaler,
                "candidate_dynamic_scaler": self.candidate_dynamic_scaler,
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
        model = NeuralChoiceModel(
            person_input_dim=cfg["person_input_dim"],
            candidate_input_dim=cfg["candidate_input_dim"],
            person_hidden_dim=cfg["person_hidden_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout_rate=cfg["dropout_rate"],
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        return cls(
            model=model,
            person_static_scaler=state["person_static_scaler"],
            person_dynamic_scaler=state["person_dynamic_scaler"],
            candidate_static_scaler=state["candidate_static_scaler"],
            candidate_dynamic_scaler=state["candidate_dynamic_scaler"],
            person_trip_cols=state["person_trip_cols"],
            candidate_cols=state["candidate_cols"],
            children_by_level1=state["children_by_level1"],
            level2_candidate_attributes_by_level1=state["level2_candidate_attributes_by_level1"],
            purpose_categories=state["purpose_categories"],
        )

    def predict_from_inputs(self, person_matrix, candidate_static, candidate_dynamic, valid_mask=None, rng=None, return_probabilities=False,
                            internal_mask=False, purpose=None, level0_h3=None, level1_h3=None):
        if internal_mask:
            if level0_h3 is None or level1_h3 is None:
                raise ValueError("level0_h3 and level1_h3 must be provided when using internal_mask")
            key = (level0_h3, level1_h3)
            level_cand_mask = self.mask.get(key)
            if level_cand_mask is None:
                cand_mask = None
            else:
                cand_mask = level_cand_mask.get(purpose)
                if cand_mask is None:
                    cand_mask = level_cand_mask.get("statent")
        else:
            cand_mask = valid_mask
        p_static  = person_matrix[:, :N_PERSON_STATIC]
        p_dynamic = person_matrix[:, N_PERSON_STATIC:]
        probs = predict_choice_proba(self.model, p_static, p_dynamic, candidate_static, candidate_dynamic, cand_mask)
        probs = probs.cpu().numpy()
        chooser = np.random.choice if rng is None else rng.choice
        indices = np.array([chooser(probs.shape[1], p=probs[i]) for i in range(probs.shape[0])], dtype=np.int64)
        if return_probabilities:
            return indices, probs
        return indices

    def _build_candidate_dynamic(self, attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work):
        return build_dynamic_vector(hx=home_x, hy=home_y, wx=work_x, wy=work_y, ox=origin_x, oy=origin_y,
                                    centroid_x=attrs["x"], centroid_y=attrs["y"], has_work=has_work)

    def predict_level2(self, level0_h3, level1_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, age, daily_longest_distance_from_home, daily_crowfly_total, daily_longest_distance_from_work, crowfly_consumed_before_trip, trip_position_class, departure_time_normalized, target_distance, activity_chain_vector, sex, employed, car_availability, income_class, purpose, origin_purpose, max_utility=False, rng=None, **kwargs):
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
        safe_daily_longest_work = float(daily_longest_distance_from_work) if np.isfinite(daily_longest_distance_from_work) and float(daily_longest_distance_from_work) >= 0.0 else 0.0
        safe_consumed = float(crowfly_consumed_before_trip) if np.isfinite(crowfly_consumed_before_trip) and float(crowfly_consumed_before_trip) >= 0.0 else 0.0
        safe_trip_position = float(trip_position_class) if np.isfinite(trip_position_class) else 2.0
        safe_departure_time = float(departure_time_normalized) if np.isfinite(departure_time_normalized) else 0.5
        safe_target_distance = float(target_distance) if np.isfinite(target_distance) and float(target_distance) >= 0.0 else 0.0

        purpose_idx = self.purpose_categories.index(str(purpose)) if str(purpose) in self.purpose_categories else 0
        purpose_hot = np.eye(len(self.purpose_categories), dtype=np.float32)[purpose_idx]
        remapped_origin = ORIGIN_PURPOSE_REMAP.get(str(origin_purpose), str(origin_purpose))
        origin_purpose_idx = self.purpose_categories.index(remapped_origin) if remapped_origin in self.purpose_categories else -1
        origin_purpose_hot = np.eye(len(self.purpose_categories), dtype=np.float32)[origin_purpose_idx] if origin_purpose_idx >= 0 else np.zeros(len(self.purpose_categories), dtype=np.float32)
        person = transform_person_trip_vector(age=age, income_class=income_class, daily_longest=safe_daily_longest, daily_total=safe_daily_total,
            daily_longest_work=safe_daily_longest_work, activity_chain_vector=activity_chain_vector,
            sex=sex, employed=employed, car_availability=car_availability,
            consumed_before=safe_consumed, trip_position=safe_trip_position, departure_time=safe_departure_time, target_distance=safe_target_distance,
            purpose_hot=purpose_hot, origin_purpose_hot=origin_purpose_hot,
            static_scaler=self.person_static_scaler, dynamic_scaler=self.person_dynamic_scaler)
        has_work = np.isfinite(work_x) and np.isfinite(work_y)
        candidate_dynamic        = self._build_candidate_dynamic(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate_dynamic_scaled = transform_candidate_dynamic_matrix(candidate_dynamic, self.candidate_dynamic_scaler)[None, :, :]
        candidate_static_scaled  = attrs["scaled_static_features_torch"]

        _, probs = self.predict_from_inputs(person, candidate_static_scaled, candidate_dynamic_scaled, rng=rng, return_probabilities=True)
        probs = probs[0, :n_children]
        company_mask = attrs["static_features"][:n_children, STATIC_CANDIDATE_FEATURES.index("num_statent")] > 0.0
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
        _si = {name: i for i, name in enumerate(STATIC_CANDIDATE_FEATURES)}
        masks = dict()
        for key, attrs in self.level2_candidate_attributes_by_level1.items():
            sf = attrs["static_features"]
            masks[key] = {
                "statent":             sf[:, _si["num_statent"]] > 0.0,
                "shop":                sf[:, _si["shop"]]        > 0.0,
                "leisure":             sf[:, _si["leisure"]]     > 0.0,
                "education_secondary": sf[:, _si["education"]]   > 0.0,
            }
        return masks




class HierarchicalLocationChoiceModel:
    logger: logging.Logger = logging.getLogger("synpp: HierarchicalLocationChoiceModel")

    def __init__(self, coarse_wrapper:RegionalChoiceWrapper, medium_wrapper:DistrictChoiceWrapper, detailed_wrapper:LocalChoiceWrapper, optimize=True):        
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
        self._last_person_static = None  # cached scaled static vector per person_id
        self.destination_index = None
        self.destination_fallback = None
        self.l1_siblings = None  # {level1_h3: [sibling_level1_h3s]} — built once, used for O(1) fallback widening
        self._purpose_categories = [str(p) for p in self.c.purpose_categories]
        self._purpose_identity = np.eye(len(self._purpose_categories), dtype=np.float32)
        self._purpose_index = {p: i for i, p in enumerate(self._purpose_categories)}

    @classmethod
    def load(cls, coarse_path:str, medium_path:str, detailed_path:str, map_location=None, optimize=True):
        cls.log("Loading models")
        coarse_wrapper = RegionalChoiceWrapper.load(coarse_path, map_location=map_location)
        medium_wrapper = DistrictChoiceWrapper.load(medium_path, map_location=map_location)
        detailed_wrapper = LocalChoiceWrapper.load(detailed_path, map_location=map_location)
        return cls(coarse_wrapper=coarse_wrapper, medium_wrapper=medium_wrapper, detailed_wrapper=detailed_wrapper, optimize=optimize)

    @classmethod
    def build(cls, context, map_location=None, optimize=True):
        regional_model_path, _, _ = context.stage("synthesis.population.spatial.secondary_nn.regional_model")
        subregional_model_path = context.stage("synthesis.population.spatial.secondary_nn.subregional_model")[0]
        local_model_path = context.stage("synthesis.population.spatial.secondary_nn.local_model")[0]
        obj = cls.load(coarse_path=regional_model_path, medium_path=subregional_model_path, detailed_path=local_model_path, map_location=map_location, optimize=optimize)
        obj.build_candidates_and_trees(context)
        return obj
    
    def build_candidates_and_trees(self, context):
        h3_data, h3_geo, h3_tree = context.stage("synthesis.population.spatial.secondary_nn.h3")
        (self.destination_index, self.destination_fallback, self.car_available_probabilities, 
            self.no_car_available_probabilities) = _prepare_destination_level2_index(context, h3_data, h3_geo)
        self.l1_siblings = _reverse_tree(h3_tree)  # {level1: [sibling_level1s]} — built once for O(1) fallback widening

    @classmethod
    def log(cls, message, level=logging.INFO):
        cls.logger.log(level, "\t\t" + message)

    def optimize_wrapper(self, enable_compile=True, compile_mode="max-autotune"):
        for key in self.wrappers:
            self.wrappers[key] = self._optimize_wrapper(self.wrappers[key], enable_compile=enable_compile, compile_mode=compile_mode)
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
        return (self._same_scaler(self.c.person_static_scaler,  self.m.person_static_scaler) and
                self._same_scaler(self.c.person_static_scaler,  self.d.person_static_scaler) and
                self._same_scaler(self.c.person_dynamic_scaler, self.m.person_dynamic_scaler) and
                self._same_scaler(self.c.person_dynamic_scaler, self.d.person_dynamic_scaler))

    def _assert_wrapper_compatibility(self):
        assert self.c.candidate_cols == self.m.candidate_cols == self.d.candidate_cols, (
            "Candidate feature columns differ across wrapper levels."
        )
        assert self.c.candidate_cols == list(CANDIDATE_FEATURES), (
            "Candidate feature columns do not match feature_encoding.CANDIDATE_FEATURES"
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

        for key, attrs in self.d.level2_candidate_attributes_by_level1.items():
            assert "static_features" in attrs and attrs["static_features"].ndim == 2, (
                f"Missing/invalid detailed static_features for key {key}"
            )
            assert attrs["static_features"].shape[1] == len(STATIC_CANDIDATE_FEATURES), (
                f"Detailed static_features width mismatch for key {key}"
            )

    def assert_ready_for_prediction(self):
        assert self.destination_index is not None and self.destination_fallback is not None and self.l1_siblings is not None, (
            "Destination pools / sibling map are not initialized. Call build_candidates_and_trees(context) before predict()."
        )

    def _get_cached_person_static(self, person_id, age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_vector, sex, employed, car_availability):
        if person_id != self._last_person_id:
            self._last_person_static = transform_person_static_vector(age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_vector, sex, employed, car_availability, self.c.person_static_scaler)
            self._last_person_id = person_id
        return self._last_person_static

    def _build_person_dynamic(self, consumed_before, trip_position, departure_time, activity_duration_h, target_distance, purpose, origin_purpose):
        purpose_hot = self._purpose_identity[self._purpose_index[purpose]]
        remapped_origin = ORIGIN_PURPOSE_REMAP.get(origin_purpose, origin_purpose)
        origin_idx = self._purpose_index[remapped_origin]
        origin_purpose_hot = self._purpose_identity[origin_idx]
        return transform_person_dynamic_vector(consumed_before, trip_position, departure_time, activity_duration_h, target_distance, purpose_hot, origin_purpose_hot, self.c.person_dynamic_scaler)

    def _predict_level0_from_person(self, person_matrix, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work, rng, purpose):
        candidate_dynamic        = self.c._build_candidate_dynamic(home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate_dynamic_scaled = transform_candidate_dynamic_matrix(candidate_dynamic, self.c.candidate_dynamic_scaler)[None, :, :]
        candidate_static_scaled  = self.c.scaled_static_features_torch
        indices = self.c.predict_from_inputs(person_matrix, candidate_static_scaled, candidate_dynamic_scaled, rng=rng, return_probabilities=False, internal_mask=True, purpose=purpose)
        return self.c.all_h3[indices[0]]

    def _predict_level1_from_person(self, person_matrix, level0_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work, purpose, rng):
        attrs = self.m.level1_candidate_attributes_by_level0.get(level0_h3)
        if attrs is None:
            raise KeyError(f"Unknown level0_h3 '{level0_h3}' in medium model candidates")
        children = attrs["children"]
        n_children = attrs["n_children"]
        if n_children == 0:
            raise RuntimeError(f"No level1 candidates available for level0_h3 '{level0_h3}'")
        candidate_dynamic        = self.m._build_candidate_dynamic(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate_dynamic_scaled = transform_candidate_dynamic_matrix(candidate_dynamic, self.m.candidate_dynamic_scaler)[None, :, :]
        candidate_static_scaled  = attrs["scaled_static_features_torch"]
        idx = self.m.predict_from_inputs(person_matrix, candidate_static_scaled, candidate_dynamic_scaled, rng=rng, return_probabilities=False, internal_mask=True, purpose=purpose, level0_h3=level0_h3)
        safe_idx = min(int(idx[0]), n_children - 1)
        return children[safe_idx]

    def _predict_level2_from_person(self, person_matrix, level0_h3, level1_h3, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work, purpose, rng):
        key = (level0_h3, level1_h3)
        attrs = self.d.level2_candidate_attributes_by_level1.get(key)
        if attrs is None:
            raise KeyError(f"Unknown key {key} in detailed model candidates")
        children = attrs["children"]
        n_children = attrs["n_children"]
        if n_children == 0:
            raise RuntimeError(f"No level2 candidates available for key {key}")
        candidate_dynamic        = self.d._build_candidate_dynamic(attrs, home_x, home_y, work_x, work_y, origin_x, origin_y, has_work)
        candidate_dynamic_scaled = transform_candidate_dynamic_matrix(candidate_dynamic, self.d.candidate_dynamic_scaler)[None, :, :]
        candidate_static_scaled  = attrs["scaled_static_features_torch"]
        idx = self.d.predict_from_inputs(person_matrix, candidate_static_scaled, candidate_dynamic_scaled, rng=rng, return_probabilities=False, internal_mask=True, purpose=purpose, level0_h3=level0_h3, level1_h3=level1_h3)
        safe_idx = min(int(idx[0]), n_children - 1)
        return children[safe_idx]

    ##################### Main prediction method that runs the full three-level prediction process #####################

    def predict(self, person_id, home_x, home_y, work_x, work_y, origin_x, origin_y, age, daily_longest_distance_from_home, daily_crowfly_total, daily_longest_distance_from_work,
            crowfly_consumed_before_trip, trip_position_class, departure_time_normalized, activity_duration_h, target_distance, activity_chain_vector, sex, employed, car_availability, income_class, purpose, origin_purpose,
                has_work, has_education, rng):
        person_static  = self._get_cached_person_static(person_id, age, income_class, daily_longest_distance_from_home, daily_crowfly_total, daily_longest_distance_from_work, activity_chain_vector, sex, employed, car_availability)
        person_dynamic = self._build_person_dynamic(crowfly_consumed_before_trip, trip_position_class, departure_time_normalized, activity_duration_h, target_distance, purpose, origin_purpose)
        person_matrix  = np.concatenate([person_static, person_dynamic], axis=1)
        level0_h3 = self._predict_level0_from_person(person_matrix, home_x=home_x, home_y=home_y, work_x=work_x, work_y=work_y, origin_x=origin_x, origin_y=origin_y, 
                                                     has_work=has_work, rng=rng, purpose=purpose)
        level1_h3 = self._predict_level1_from_person(person_matrix, level0_h3=level0_h3, home_x=home_x, home_y=home_y, work_x=work_x, work_y=work_y, 
                                                     origin_x=origin_x, origin_y=origin_y, has_work=has_work, purpose=purpose, rng=rng)
        level2_h3 = self._predict_level2_from_person(person_matrix, level0_h3=level0_h3, level1_h3=level1_h3, home_x=home_x, home_y=home_y, work_x=work_x, 
                                                     work_y=work_y, origin_x=origin_x, origin_y=origin_y, has_work=has_work, purpose=purpose, rng=rng)
        destination_id, geom = self.sample_company_in_l2(purpose, level1_h3, level2_h3, rng, car_availability, home_x, home_y)
        return destination_id, geom


    def sample_company_in_l2(self, purpose, level1_h3, level2_h3, rng, car_availability=None, home_x=None, home_y=None):
        probabilities = self.car_available_probabilities if car_availability else self.no_car_available_probabilities

        purpose_index = self.destination_index.get(purpose)
        purpose_prob = probabilities.get(purpose)
        if purpose_index is not None:
            pool = purpose_index.get(level2_h3)
            if pool:
                prob = purpose_prob.get(level2_h3)
                return pool[int(rng.choice(len(pool), p=prob))]

        purpose_fallback = self.destination_fallback.get(purpose)
        if purpose_fallback is not None:
            pool = purpose_fallback.get(level1_h3)
            if pool:
                prob = purpose_prob.get(level1_h3)
                return pool[int(rng.choice(len(pool), p=prob))]

        other_fallback = self.destination_fallback.get('other')
        other_prob = probabilities.get('other')
        if other_fallback is not None:
            pool = other_fallback.get(level1_h3)
            if pool:
                self.log(f"Using fallback pool for level1_h3 '{level1_h3}' and purpose '{purpose}' with {len(pool)} candidates", level=logging.WARNING)
                prob = other_prob.get(level1_h3)
                return pool[int(rng.choice(len(pool), p=prob))]

        # All local fallbacks exhausted — widen to sibling level1 cells (O(1) parent lookup, pre-built at init).
        sibling_l1s = self.l1_siblings.get(level1_h3)
        if sibling_l1s:
            for i in rng.permutation(len(sibling_l1s)):
                sib_l1 = sibling_l1s[i]
                if purpose_fallback is not None:
                    pool = purpose_fallback.get(sib_l1)
                    if pool:
                        prob = purpose_prob.get(sib_l1)
                        self.log(f"Widened to sibling level1 '{sib_l1}' for purpose '{purpose}' (original level1 '{level1_h3}' had no destinations)", level=logging.WARNING)
                        return pool[int(rng.choice(len(pool), p=prob))]
                if other_fallback is not None:
                    pool = other_fallback.get(sib_l1)
                    if pool:
                        prob = other_prob.get(sib_l1)
                        self.log(f"Widened to sibling level1 '{sib_l1}' (other fallback) for purpose '{purpose}' (original level1 '{level1_h3}' had no destinations)", level=logging.WARNING)
                        return pool[int(rng.choice(len(pool), p=prob))]
        
        raise RuntimeError(
            f"No candidates in destination pool for purpose '{purpose}', "
            f"level1_h3 '{level1_h3}', level2_h3 '{level2_h3}'"
        )
