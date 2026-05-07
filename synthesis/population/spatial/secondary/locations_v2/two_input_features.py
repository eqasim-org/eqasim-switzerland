import numpy as np
from sklearn.preprocessing import QuantileTransformer

PERSON_TRIP_NUMERIC_FEATURES = [
    "age",
    "income_class",
    "daily_longest_distance_from_home",
    "daily_crowfly_total",
    "crowfly_consumed_before_trip",
    "trip_position_class",
]
PERSON_TRIP_BINARY_FEATURES = ["sex", "employed", "car_availability"]
STATIC_CANDIDATE_FEATURES = ["num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure", 
                             "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none", 
                             "outside_fraction"]

DYNAMIC_CANDIDATE_FEATURES = ["dist_home", "dist_work", "dist_last"]
CANDIDATE_FEATURES = DYNAMIC_CANDIDATE_FEATURES + STATIC_CANDIDATE_FEATURES

def _make_quantile_transformer(n_rows, output_distribution):
    n_quantiles = int(min(1000, max(10, n_rows)))
    return QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles)


def fit_person_trip_matrix(age, sex, employed, car_availability, income_class, daily_longest, daily_total, consumed_before, trip_position, purpose_series, purpose_categories, person_scaler = None):
    age = np.asarray(age, dtype=np.float64)
    income_class = np.asarray(income_class, dtype=np.float64)
    daily_longest = np.asarray(daily_longest, dtype=np.float64)
    daily_total = np.asarray(daily_total, dtype=np.float64)
    consumed_before = np.asarray(consumed_before, dtype=np.float64)
    trip_position = np.asarray(trip_position, dtype=np.float64)

    numeric = np.column_stack([age, income_class, daily_longest, daily_total, consumed_before, trip_position]).astype(np.float64)
    if numeric.shape[0] > 1:
        if person_scaler is None:
            scaler = _make_quantile_transformer(numeric.shape[0], output_distribution="normal")
            numeric_scaled = scaler.fit_transform(numeric).astype(np.float32)
        else:
            scaler = person_scaler
            numeric_scaled = scaler.transform(numeric).astype(np.float32)
    else:
        scaler = None
        numeric_scaled = numeric.astype(np.float32)

    purpose_categories = [str(p) for p in purpose_categories]
    purpose_to_idx = {purpose: idx for idx, purpose in enumerate(purpose_categories)}
    purpose_idx = purpose_series.astype(str).map(purpose_to_idx).fillna(0).astype(np.int64).to_numpy()
    purpose_one_hot = np.eye(len(purpose_categories), dtype=np.float32)[purpose_idx]

    binary = np.column_stack([
        np.asarray(sex, dtype=np.float32),
        np.asarray(employed, dtype=np.float32),
        np.asarray(car_availability, dtype=np.float32),
    ]).astype(np.float32)

    matrix = np.concatenate([numeric_scaled, binary, purpose_one_hot], axis=1).astype(np.float32)
    feature_names = PERSON_TRIP_NUMERIC_FEATURES + PERSON_TRIP_BINARY_FEATURES + [f"purpose_{p}" for p in purpose_categories]
    return matrix, scaler, feature_names


def transform_person_trip_vector(age, sex, employed, car_availability, income_class, daily_longest, daily_total, consumed_before, trip_position, purpose_hot, scaler):
    numeric = np.array([[age, income_class, daily_longest, daily_total, consumed_before, trip_position]], dtype=np.float64)    
    numeric = scaler.transform(numeric).astype(np.float32)    
    binary = np.array([[sex, employed, car_availability]], dtype=np.float32)
    return np.concatenate([numeric, binary, [purpose_hot]], axis=1).astype(np.float32)


def fit_candidate_tensor(candidate_tensor, valid_mask, max_fit_rows=1000000, random_state=123, candidate_scaler=None):
    candidate_tensor = np.asarray(candidate_tensor, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    flat = candidate_tensor.reshape(-1, candidate_tensor.shape[-1])
    valid_flat = valid_mask.reshape(-1)
    
    if candidate_scaler is not None:
        scaler = candidate_scaler
        scaled_flat = scaler.transform(flat)
    else:
        if np.any(valid_flat):
            valid_values = flat[valid_flat]
            if max_fit_rows is not None and valid_values.shape[0] > int(max_fit_rows):
                rng = np.random.RandomState(int(random_state))
                selected = rng.choice(valid_values.shape[0], size=int(max_fit_rows), replace=False)
                fit_values = valid_values[selected]
            else:
                fit_values = valid_values

            if valid_values.shape[0] > 1:
                scaler = _make_quantile_transformer(fit_values.shape[0], output_distribution="normal")
                scaler.fit(fit_values)
                scaled_flat = scaler.transform(flat)
            else:
                scaler = None
                scaled_flat = flat
        else:
            scaler = None
            scaled_flat = flat

    scaled = scaled_flat.reshape(candidate_tensor.shape).astype(np.float32)
    scaled[~valid_mask] = 0.0
    return scaled, scaler


def transform_candidate_matrix(candidate_matrix, scaler):
    return scaler.transform(candidate_matrix).astype(np.float32)
