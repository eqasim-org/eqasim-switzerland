from .tools import MOBILITY_FEATURES, OSM_FEATURES, OVGK_MEAN_FEATURES
from .chain_state import CHAIN_STATE_FEATURES
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from .hierarchical_utils import ORIGIN_PURPOSE_REMAP, SECONDARY_ACTIVITIES

ACTIVITY_CHAIN_N                = len(SECONDARY_ACTIVITIES)  # = 6; first 6 dims of encode_purpose output (SECONDARY_ACTIVITIES only)
PERSON_STATIC_NUMERIC_FEATURES  = ["age", "income_class", "daily_longest_distance_from_home", "daily_crowfly_total", "daily_longest_distance_from_work"] + [f"activity_chain_{a}" for a in SECONDARY_ACTIVITIES]
PERSON_STATIC_BINARY_FEATURES   = ["sex", "employed", "car_availability"] + MOBILITY_FEATURES
PERSON_STATIC_FEATURES          = PERSON_STATIC_NUMERIC_FEATURES + PERSON_STATIC_BINARY_FEATURES   # Numeric quantities scaled; mobility indicators and ordered ÖVGK pass through
PERSON_DYNAMIC_FEATURES         = ["crowfly_consumed_before_trip", "trip_position_class", "departure_time_normalized", "activity_duration_h", "target_distance"]  # 5 scaled dims + purpose one-hot + origin_purpose one-hot appended at build time
# Legacy aliases so any code still referencing the old names keeps compiling.
PERSON_TRIP_NUMERIC_FEATURES = PERSON_STATIC_NUMERIC_FEATURES + PERSON_DYNAMIC_FEATURES
PERSON_TRIP_BINARY_FEATURES  = PERSON_STATIC_BINARY_FEATURES
N_PERSON_STATIC = len(PERSON_STATIC_FEATURES)       # Shared split index for person vectors
# Remap raw trip origin_purpose labels to the same vocabulary as purpose (SECONDARY_ACTIVITIES).
# Scaled features (counts/quantities) come first; passthrough fractions come after.
# This enables simple contiguous slices: [:N_CANDIDATE_STATIC_SCALED] and [N_CANDIDATE_STATIC_SCALED:]
CANDIDATE_STATIC_SCALED_FEATURES = [
    "num_statent", "employees", "education", "shop", "leisure",
    "sport", "gastronomy", "accommodation", "cultural",
] + OSM_FEATURES
CANDIDATE_STATIC_PASSTHROUGH_FEATURES = [
    "urban_core", "urban",
    "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none",
    "outside_fraction",
] + OVGK_MEAN_FEATURES
STATIC_CANDIDATE_FEATURES = CANDIDATE_STATIC_SCALED_FEATURES + CANDIDATE_STATIC_PASSTHROUGH_FEATURES
N_CANDIDATE_STATIC_SCALED = len(CANDIDATE_STATIC_SCALED_FEATURES)   # Scaled environmental quantities precede fractions
DYNAMIC_CANDIDATE_FEATURES = ["dist_home", "dist_work", "dist_last", "detour_factor"]
CANDIDATE_FEATURES = DYNAMIC_CANDIDATE_FEATURES + STATIC_CANDIDATE_FEATURES
N_CANDIDATE_DYNAMIC = len(DYNAMIC_CANDIDATE_FEATURES)  # = 4; split index: candidate_vector[:N_CANDIDATE_DYNAMIC] is dynamic


def add_detour_factor_feature(candidate_tensor, origin_x, origin_y, candidate_x,
                              candidate_y, valid_mask, detour_factor):
    """Insert the origin-to-candidate detour factor after the distance features.

    Candidate coordinates are snapped in vectorized batches. For a shared 1-D
    candidate set (the regional model), destination cells are snapped only once;
    hierarchical 2-D choice sets snap valid entries only.
    """
    candidate_tensor = np.asarray(candidate_tensor)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    origin_indices = detour_factor.get_cell_indices(origin_x, origin_y)
    factors = np.zeros(valid_mask.shape, dtype=np.float64)

    candidate_x = np.asarray(candidate_x, dtype=np.float64)
    candidate_y = np.asarray(candidate_y, dtype=np.float64)
    if candidate_x.ndim == 1:
        destination_indices = detour_factor.get_cell_indices(candidate_x, candidate_y)
        factors[:, :] = detour_factor.get_detour_factor_by_index(
            origin_indices[:, None], destination_indices[None, :]
        )
        factors[~valid_mask] = 0.0
    else:
        rows, columns = np.nonzero(valid_mask)
        destination_indices = detour_factor.get_cell_indices(
            candidate_x[rows, columns], candidate_y[rows, columns]
        )
        factors[rows, columns] = detour_factor.get_detour_factor_by_index(
            origin_indices[rows], destination_indices
        )

    # The Numba builders emit three distances followed by all static columns.
    return np.concatenate(
        [candidate_tensor[:, :, :3], factors[:, :, None], candidate_tensor[:, :, 3:]],
        axis=2,
    )


def _make_quantile_transformer(n_rows, random_state=None):
    """All scalable features in this model are non-negative by construction (distances, counts,
    fractions, ages, …).  Using output_distribution='uniform' maps them to [0, 1], guaranteeing
    no negative values after transformation."""
    n_quantiles = int(min(n_rows, 1000, max(1, n_rows/10)))
    return QuantileTransformer(output_distribution="uniform", n_quantiles=n_quantiles, random_state=random_state)


def fit_person_trip_matrix(age, sex, employed, car_availability, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_matrix,
                           consumed_before, trip_position, departure_time, activity_duration_h, target_distance, purpose_series, origin_purpose_series, purpose_categories,
                           person_static_scaler=None, person_dynamic_scaler=None, use_target_distance=True, random_state=None, mobility_matrix=None, chain_state_matrix=None):
    age = np.asarray(age, dtype=np.float64); income_class = np.asarray(income_class, dtype=np.float64)
    daily_longest = np.asarray(daily_longest, dtype=np.float64); daily_total = np.asarray(daily_total, dtype=np.float64)
    daily_longest_work = np.asarray(daily_longest_work, dtype=np.float64)
    activity_chain_matrix = np.asarray(activity_chain_matrix, dtype=np.float64)  # [N, ACTIVITY_CHAIN_N]
    consumed_before = np.asarray(consumed_before, dtype=np.float64); trip_position = np.asarray(trip_position, dtype=np.float64)
    departure_time = np.asarray(departure_time, dtype=np.float64)
    activity_duration_h = np.asarray(activity_duration_h, dtype=np.float64)
    target_distance = np.asarray(target_distance, dtype=np.float64)
    static_numeric  = np.column_stack([age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_matrix]).astype(np.float64)  # [N, 11]
    dynamic_numeric = np.column_stack([consumed_before, trip_position, departure_time, activity_duration_h, target_distance]).astype(np.float64)                      # [N, 5]
    if person_dynamic_scaler is not None:
        expected = getattr(person_dynamic_scaler, "use_target_distance", True)
        if expected != use_target_distance:
            raise ValueError("Target-distance option differs from the fitted person scaler")
    if not use_target_distance:
        dynamic_numeric = dynamic_numeric[:, :-1]
    if chain_state_matrix is not None:
        state = np.asarray(chain_state_matrix, dtype=np.float64)
        if state.shape != (len(age), len(CHAIN_STATE_FEATURES)) or not np.isfinite(state).all():
            raise ValueError("Invalid remaining-chain feature matrix")
        dynamic_numeric = np.column_stack([dynamic_numeric, state[:, :-1]])
    if person_dynamic_scaler is not None:
        if getattr(person_dynamic_scaler, "chain_state_features", False) != (chain_state_matrix is not None):
            raise ValueError("Remaining-chain option differs from the fitted person scaler")
    if static_numeric.shape[0] == 0:
        raise ValueError("Cannot fit person features without training rows")
    if person_static_scaler is None:
        person_static_scaler  = _make_quantile_transformer(static_numeric.shape[0], random_state=random_state)
        static_scaled  = person_static_scaler.fit_transform(static_numeric).astype(np.float32)
    else:
        static_scaled  = person_static_scaler.transform(static_numeric).astype(np.float32)
    if person_dynamic_scaler is None:
        person_dynamic_scaler = _make_quantile_transformer(dynamic_numeric.shape[0], random_state=random_state)
        dynamic_scaled = person_dynamic_scaler.fit_transform(dynamic_numeric).astype(np.float32)
    else:
        dynamic_scaled = person_dynamic_scaler.transform(dynamic_numeric).astype(np.float32)
    if chain_state_matrix is not None:
        dynamic_scaled = np.column_stack([dynamic_scaled, state[:, -1]])
    person_dynamic_scaler.use_target_distance = use_target_distance
    person_dynamic_scaler.chain_state_features = chain_state_matrix is not None

    purpose_categories = [str(p) for p in purpose_categories]
    purpose_to_idx = {purpose: idx for idx, purpose in enumerate(purpose_categories)}
    purpose_idx = purpose_series.astype(str).map(purpose_to_idx).fillna(0).astype(np.int64).to_numpy()
    purpose_one_hot = np.eye(len(purpose_categories), dtype=np.float32)[purpose_idx]
    # origin_purpose: remap raw labels then one-hot encode (unknown values → all-zeros row)
    remapped_origin = origin_purpose_series.astype(str).map(lambda x: ORIGIN_PURPOSE_REMAP.get(x, x))
    origin_idx = remapped_origin.map(purpose_to_idx).fillna(-1).astype(np.int64).to_numpy()
    origin_purpose_one_hot = np.zeros((len(origin_idx), len(purpose_categories)), dtype=np.float32)
    valid_origin = origin_idx >= 0
    origin_purpose_one_hot[valid_origin] = np.eye(len(purpose_categories), dtype=np.float32)[origin_idx[valid_origin]]
    binary = np.column_stack([np.asarray(sex, dtype=np.float32), np.asarray(employed, dtype=np.float32), np.asarray(car_availability, dtype=np.float32)]).astype(np.float32)
    mobility = np.zeros((len(age), len(MOBILITY_FEATURES)), dtype=np.float32) if mobility_matrix is None else np.asarray(mobility_matrix, dtype=np.float32)
    if mobility.shape != (len(age), len(MOBILITY_FEATURES)) or not np.isfinite(mobility).all():
        raise ValueError("Invalid person mobility feature matrix")
    static_matrix  = np.concatenate([static_scaled, binary, mobility],                                          axis=1).astype(np.float32)
    dynamic_matrix = np.concatenate([dynamic_scaled, purpose_one_hot, origin_purpose_one_hot],         axis=1).astype(np.float32)  # [N, 5+2P]
    full_matrix    = np.concatenate([static_matrix,  dynamic_matrix],                                  axis=1).astype(np.float32)
    feature_names  = PERSON_STATIC_FEATURES + [name for name in PERSON_DYNAMIC_FEATURES if use_target_distance or name != "target_distance"] + (CHAIN_STATE_FEATURES if person_dynamic_scaler.chain_state_features else []) + [f"purpose_{p}" for p in purpose_categories] + [f"origin_purpose_{p}" for p in purpose_categories]
    return full_matrix, static_matrix, dynamic_matrix, person_static_scaler, person_dynamic_scaler, feature_names


def transform_person_static_vector(age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_vector, sex, employed, car_availability, static_scaler, mobility_vector=None):
    """Scale person quantities and append binary demographic/mobility attributes."""
    chain = np.asarray(activity_chain_vector, dtype=np.float64).flatten()[:ACTIVITY_CHAIN_N].reshape(1, -1)  # [1, 6]
    numeric = np.concatenate([np.array([[age, income_class, daily_longest, daily_total, daily_longest_work]], dtype=np.float64), chain], axis=1)  # [1, 11]
    numeric_scaled = static_scaler.transform(numeric).astype(np.float32)
    binary = np.array([[sex, employed, car_availability]], dtype=np.float32)
    mobility = np.zeros((1, len(MOBILITY_FEATURES)), dtype=np.float32) if mobility_vector is None else np.asarray(mobility_vector, dtype=np.float32).reshape(1, len(MOBILITY_FEATURES))
    return np.concatenate([numeric_scaled, binary, mobility], axis=1).astype(np.float32)

def transform_person_dynamic_vector(consumed_before, trip_position, departure_time, activity_duration_h, target_distance, purpose_hot, origin_purpose_hot, dynamic_scaler, chain_state_vector=None):
    """Encode trip state with the same optional distance and plan fields as fitting."""
    numeric = np.array([[consumed_before, trip_position, departure_time, activity_duration_h, target_distance]], dtype=np.float64)
    if not getattr(dynamic_scaler, "use_target_distance", True):
        numeric = numeric[:, :-1]
    if getattr(dynamic_scaler, "chain_state_features", False):
        if chain_state_vector is None:
            raise ValueError("Prediction requires remaining-chain features from the remaining plan")
        state = np.asarray(chain_state_vector, dtype=np.float64).reshape(1, -1)
        if state.shape[1] != len(CHAIN_STATE_FEATURES) or not np.isfinite(state).all():
            raise ValueError("Invalid remaining-chain prediction features")
        numeric = np.column_stack([numeric, state[:, :-1]])
    numeric_scaled = dynamic_scaler.transform(numeric).astype(np.float32)
    if getattr(dynamic_scaler, "chain_state_features", False):
        numeric_scaled = np.column_stack([numeric_scaled, state[:, -1]])
    return np.concatenate([numeric_scaled, [purpose_hot], [origin_purpose_hot]], axis=1).astype(np.float32)  # [1, 5+2P]

def transform_person_trip_vector(age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_vector, sex, employed, car_availability, consumed_before, trip_position, departure_time, activity_duration_h, target_distance, purpose_hot, origin_purpose_hot, static_scaler, dynamic_scaler, chain_state_vector=None, mobility_vector=None):
    """Combined convenience wrapper used by standalone predict_levelX analysis methods."""
    p_static  = transform_person_static_vector(age, income_class, daily_longest, daily_total, daily_longest_work, activity_chain_vector, sex, employed, car_availability, static_scaler, mobility_vector=mobility_vector)
    p_dynamic = transform_person_dynamic_vector(consumed_before, trip_position, departure_time, activity_duration_h, target_distance, purpose_hot, origin_purpose_hot, dynamic_scaler, chain_state_vector=chain_state_vector)  # [1, 5+2P]
    return np.concatenate([p_static, p_dynamic], axis=1).astype(np.float32)  # [1, 19+2P]


def fit_candidate_tensor(candidate_tensor, valid_mask, max_fit_rows=1000000, random_state=123, candidate_static_scaler=None, candidate_dynamic_scaler=None):
    candidate_tensor = np.asarray(candidate_tensor, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    flat = candidate_tensor.reshape(-1, candidate_tensor.shape[-1])
    valid_flat = valid_mask.reshape(-1)
    def _fit_or_transform(cols_flat, scaler):
        if scaler is not None:
            return scaler, scaler.transform(cols_flat)
        valid_vals = cols_flat[valid_flat]
        if valid_vals.shape[0] <= 1:
            return None, cols_flat
        fit_vals = valid_vals
        if max_fit_rows is not None and valid_vals.shape[0] > int(max_fit_rows):
            rs = np.random.RandomState(int(random_state))
            fit_vals = valid_vals[rs.choice(valid_vals.shape[0], size=int(max_fit_rows), replace=False)]
        sc = _make_quantile_transformer(fit_vals.shape[0], random_state=random_state); sc.fit(fit_vals)
        return sc, sc.transform(cols_flat)

    # Dynamic features (distances and detour factor) are non-negative.
    candidate_dynamic_scaler, scaled_dyn = _fit_or_transform(flat[:, :N_CANDIDATE_DYNAMIC], candidate_dynamic_scaler)
    # Static features: scaled block first (contiguous slice), passthrough block kept as-is.
    flat_stat = flat[:, N_CANDIDATE_DYNAMIC:]                                                                              # [N*M, 17]
    candidate_static_scaler, scaled_stat_cols = _fit_or_transform(flat_stat[:, :N_CANDIDATE_STATIC_SCALED], candidate_static_scaler)  # [N*M, 9]
    scaled_stat = flat_stat.copy()
    scaled_stat[:, :N_CANDIDATE_STATIC_SCALED] = scaled_stat_cols   # overwrite scaled block; passthrough block unchanged
    scaled_flat = np.concatenate([scaled_dyn, scaled_stat], axis=1)
    scaled = scaled_flat.reshape(candidate_tensor.shape).astype(np.float32)
    scaled[~valid_mask] = 0.0
    return scaled, candidate_static_scaler, candidate_dynamic_scaler


def transform_candidate_static_matrix(static_matrix, scaler):
    """Scale static candidate columns [N, 17].
    The first N_CANDIDATE_STATIC_SCALED columns are counts scaled to [0, 1] via QuantileTransformer(uniform).
    The remaining CANDIDATE_STATIC_PASSTHROUGH_FEATURES columns (already-normalised fractions) are kept as-is.
    """
    static_np = np.asarray(static_matrix, dtype=np.float64)   # [N, 17]
    result = static_np.astype(np.float32).copy()
    result[:, :N_CANDIDATE_STATIC_SCALED] = scaler.transform(static_np[:, :N_CANDIDATE_STATIC_SCALED]).astype(np.float32)
    return result

def transform_candidate_dynamic_matrix(dynamic_matrix, scaler):
    """Scale per-trip dynamic candidate columns [N, 4]."""
    return scaler.transform(np.asarray(dynamic_matrix, dtype=np.float64)).astype(np.float32)
