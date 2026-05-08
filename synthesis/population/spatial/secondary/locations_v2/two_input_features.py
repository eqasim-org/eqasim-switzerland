import numpy as np
from sklearn.preprocessing import QuantileTransformer

PERSON_STATIC_NUMERIC_FEATURES = ["age", "income_class", "daily_longest_distance_from_home", "daily_crowfly_total"]
PERSON_STATIC_BINARY_FEATURES  = ["sex", "employed", "car_availability"]
PERSON_STATIC_FEATURES         = PERSON_STATIC_NUMERIC_FEATURES + PERSON_STATIC_BINARY_FEATURES   # 7 dims; numeric scaled, binary pass-through
PERSON_DYNAMIC_FEATURES        = ["crowfly_consumed_before_trip", "trip_position_class"]  # 2 scaled dims + purpose one-hot appended at build time
# Legacy aliases so any code still referencing the old names keeps compiling.
PERSON_TRIP_NUMERIC_FEATURES = PERSON_STATIC_NUMERIC_FEATURES + PERSON_DYNAMIC_FEATURES
PERSON_TRIP_BINARY_FEATURES  = PERSON_STATIC_BINARY_FEATURES
N_PERSON_STATIC = len(PERSON_STATIC_FEATURES)       # = 7; split index: person_vector[:N_PERSON_STATIC] is static
STATIC_CANDIDATE_FEATURES = ["num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure",
                             "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none",
                             "outside_fraction"]
DYNAMIC_CANDIDATE_FEATURES = ["dist_home", "dist_work", "dist_last"]
CANDIDATE_FEATURES = DYNAMIC_CANDIDATE_FEATURES + STATIC_CANDIDATE_FEATURES
N_CANDIDATE_DYNAMIC = len(DYNAMIC_CANDIDATE_FEATURES)  # = 3; split index: candidate_vector[:N_CANDIDATE_DYNAMIC] is dynamic

def _make_quantile_transformer(n_rows, output_distribution):
    n_quantiles = int(min(1000, max(10, n_rows)))
    return QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles)


def fit_person_trip_matrix(age, sex, employed, car_availability, income_class, daily_longest, daily_total, consumed_before, trip_position, purpose_series, purpose_categories, person_static_scaler=None, person_dynamic_scaler=None):
    age = np.asarray(age, dtype=np.float64); income_class = np.asarray(income_class, dtype=np.float64)
    daily_longest = np.asarray(daily_longest, dtype=np.float64); daily_total = np.asarray(daily_total, dtype=np.float64)
    consumed_before = np.asarray(consumed_before, dtype=np.float64); trip_position = np.asarray(trip_position, dtype=np.float64)
    static_numeric  = np.column_stack([age, income_class, daily_longest, daily_total]).astype(np.float64)        # [N, 4]
    dynamic_numeric = np.column_stack([consumed_before, trip_position]).astype(np.float64)                       # [N, 2]
    if static_numeric.shape[0] > 1:
        if person_static_scaler is None:
            person_static_scaler  = _make_quantile_transformer(static_numeric.shape[0],  "normal")
            static_scaled  = person_static_scaler.fit_transform(static_numeric).astype(np.float32)
        else:
            static_scaled  = person_static_scaler.transform(static_numeric).astype(np.float32)
        if person_dynamic_scaler is None:
            person_dynamic_scaler = _make_quantile_transformer(dynamic_numeric.shape[0], "normal")
            dynamic_scaled = person_dynamic_scaler.fit_transform(dynamic_numeric).astype(np.float32)
        else:
            dynamic_scaled = person_dynamic_scaler.transform(dynamic_numeric).astype(np.float32)
    else:
        static_scaled = static_numeric.astype(np.float32)
        dynamic_scaled = dynamic_numeric.astype(np.float32)
        
    purpose_categories = [str(p) for p in purpose_categories]
    purpose_to_idx = {purpose: idx for idx, purpose in enumerate(purpose_categories)}
    purpose_idx = purpose_series.astype(str).map(purpose_to_idx).fillna(0).astype(np.int64).to_numpy()
    purpose_one_hot = np.eye(len(purpose_categories), dtype=np.float32)[purpose_idx]
    binary = np.column_stack([np.asarray(sex, dtype=np.float32), np.asarray(employed, dtype=np.float32), np.asarray(car_availability, dtype=np.float32)]).astype(np.float32)
    static_matrix  = np.concatenate([static_scaled,  binary],           axis=1).astype(np.float32)  # [N, 5]  layout: [static_num(2) | binary(3)]
    dynamic_matrix = np.concatenate([dynamic_scaled, purpose_one_hot],  axis=1).astype(np.float32)  # [N, 4+P] layout: [dynamic_num(4) | purpose(P)]
    full_matrix    = np.concatenate([static_matrix,  dynamic_matrix],   axis=1).astype(np.float32)  # [N, 5+4+P]
    feature_names  = PERSON_STATIC_FEATURES + PERSON_DYNAMIC_FEATURES + [f"purpose_{p}" for p in purpose_categories]
    return full_matrix, static_matrix, dynamic_matrix, person_static_scaler, person_dynamic_scaler, feature_names


def transform_person_static_vector(age, income_class, daily_longest, daily_total, sex, employed, car_availability, static_scaler):
    """Returns [1, 7]: [scaled_age, scaled_income, scaled_daily_longest, scaled_daily_total, sex, employed, car_availability]."""
    numeric = np.array([[age, income_class, daily_longest, daily_total]], dtype=np.float64)
    numeric_scaled = static_scaler.transform(numeric).astype(np.float32)
    binary = np.array([[sex, employed, car_availability]], dtype=np.float32)
    return np.concatenate([numeric_scaled, binary], axis=1).astype(np.float32)  # [1, 7]

def transform_person_dynamic_vector(consumed_before, trip_position, purpose_hot, dynamic_scaler):
    """Returns [1, 2+P]: [scaled_consumed_before, scaled_trip_position, purpose_one_hot]."""
    numeric = np.array([[consumed_before, trip_position]], dtype=np.float64)
    numeric_scaled = dynamic_scaler.transform(numeric).astype(np.float32)
    return np.concatenate([numeric_scaled, [purpose_hot]], axis=1).astype(np.float32)  # [1, 2+P]

def transform_person_trip_vector(age, income_class, daily_longest, daily_total, sex, employed, car_availability, consumed_before, trip_position, purpose_hot, static_scaler, dynamic_scaler):
    """Combined convenience wrapper used by standalone predict_levelX analysis methods."""
    p_static  = transform_person_static_vector(age, income_class, daily_longest, daily_total, sex, employed, car_availability, static_scaler)   # [1, 7]
    p_dynamic = transform_person_dynamic_vector(consumed_before, trip_position, purpose_hot, dynamic_scaler)  # [1, 2+P]
    return np.concatenate([p_static, p_dynamic], axis=1).astype(np.float32)  # [1, 9+P]


def fit_candidate_tensor(candidate_tensor, valid_mask, max_fit_rows=1000000, random_state=123, candidate_static_scaler=None, candidate_dynamic_scaler=None):
    candidate_tensor = np.asarray(candidate_tensor, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    flat = candidate_tensor.reshape(-1, candidate_tensor.shape[-1])   # [N*M, 16]
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
        sc = _make_quantile_transformer(fit_vals.shape[0], "normal"); sc.fit(fit_vals)
        return sc, sc.transform(cols_flat)
    candidate_dynamic_scaler, scaled_dyn  = _fit_or_transform(flat[:, :N_CANDIDATE_DYNAMIC], candidate_dynamic_scaler)  # [N*M, 3]
    candidate_static_scaler,  scaled_stat = _fit_or_transform(flat[:, N_CANDIDATE_DYNAMIC:],  candidate_static_scaler)   # [N*M, 13]
    scaled_flat = np.concatenate([scaled_dyn, scaled_stat], axis=1)
    scaled = scaled_flat.reshape(candidate_tensor.shape).astype(np.float32)
    scaled[~valid_mask] = 0.0
    return scaled, candidate_static_scaler, candidate_dynamic_scaler


def transform_candidate_static_matrix(static_matrix, scaler):
    """Scale pre-extracted static candidate columns [N, 13]."""
    return scaler.transform(np.asarray(static_matrix, dtype=np.float64)).astype(np.float32)

def transform_candidate_dynamic_matrix(dynamic_matrix, scaler):
    """Scale per-trip dynamic candidate columns [N, 3]."""
    return scaler.transform(np.asarray(dynamic_matrix, dtype=np.float64)).astype(np.float32)
