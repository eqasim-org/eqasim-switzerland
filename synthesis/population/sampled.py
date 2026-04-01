import numpy as np
import logging
import pandas as pd
import numba

logger = logging.getLogger("synpp")



def _allocate_largest_remainder(counts, probability):
    counts = np.asarray(counts, dtype=float)
    expected = counts * probability
    base = np.floor(expected).astype(int)
    remainder = int(round(expected.sum() - base.sum()))

    if remainder > 0:
        order = np.argsort(-(expected - base))
        base[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(expected - base)
        for idx in order:
            if remainder == 0:
                break
            if base[idx] > 0:
                base[idx] -= 1
                remainder += 1

    return base


def _household_features(df, sampling_col, aggregation_col):
    hh = df[[sampling_col, aggregation_col]].drop_duplicates(sampling_col).copy()
    hh["persons"] = df.groupby(sampling_col)["person_id"].size().reindex(hh[sampling_col]).values

    if "employed" in df.columns:
        employed = (df["employed"] == 1).astype(int)
        hh["employed_persons"] = (
            df.assign(_employed_bin=employed)
            .groupby(sampling_col)["_employed_bin"]
            .sum()
            .reindex(hh[sampling_col])
            .values
        )
    else:
        hh["employed_persons"] = 0

    if "car_availability" in df.columns:
        assert set(df["car_availability"].unique()) <= {0, 1}, "Expected 0 or 1 for car availability at this stage"
        car_values = (df["car_availability"] == 1).astype(int)
        hh["car_available_persons"] = (
            df.assign(_car_bin=car_values)
            .groupby(sampling_col)["_car_bin"]
            .sum()
            .reindex(hh[sampling_col])
            .values
            .astype(int)
        )
    else:
        hh["car_available_persons"] = 0

    return hh

@numba.njit(cache=True)
def _objective(
    cur_persons,
    cur_employed,
    cur_car_persons,
    target_persons,
    target_employed,
    target_car_persons,
):
    wp = 1.0 / max(1.0, target_persons)
    we = 1.0 / max(1.0, target_employed)
    wc = 1.0 / max(1.0, target_car_persons)
    return (
        wp * abs(cur_persons - target_persons)
        + we * abs(cur_employed - target_employed)
        + wc * abs(cur_car_persons - target_car_persons)
    )


@numba.njit(cache=True)
def _sample_balanced_indices_numba(persons, employed, car_available, target_households, probability, seed):
    n = persons.shape[0]
    convergence_threshold = 1e-3

    if target_households <= 0:
        return np.empty(0, dtype=np.int64)
    if target_households >= n:
        return np.arange(n, dtype=np.int64)

    np.random.seed(seed)

    target_persons = int(np.rint(persons.sum() * probability))
    target_employed = int(np.rint(employed.sum() * probability))
    target_car_persons = int(np.rint(car_available.sum() * probability))

    n_restarts = max(4, min(12, int(np.ceil(np.log2(n + 1)) + 2)))
    max_iter =int(min(20000, (1/probability) * target_households + 1e3))
    non_selected_count = n - target_households

    best_objective = np.inf
    best_selected = np.zeros(n, dtype=np.bool_)
    has_best = False

    for _ in range(n_restarts):
        permutation = np.arange(n, dtype=np.int64)
        for i in range(n - 1, 0, -1):
            j = np.random.randint(i + 1)
            tmp = permutation[i]
            permutation[i] = permutation[j]
            permutation[j] = tmp

        selected = np.zeros(n, dtype=np.bool_)
        selected_idx = np.empty(target_households, dtype=np.int64)
        non_selected_idx = np.empty(non_selected_count, dtype=np.int64)

        for i in range(target_households):
            idx = permutation[i]
            selected[idx] = True
            selected_idx[i] = idx

        for i in range(non_selected_count):
            non_selected_idx[i] = permutation[target_households + i]

        cur_persons = 0
        cur_employed = 0
        cur_car_persons = 0

        for i in range(target_households):
            idx = selected_idx[i]
            cur_persons += persons[idx]
            cur_employed += employed[idx]
            cur_car_persons += car_available[idx]

        current = _objective(
            cur_persons,
            cur_employed,
            cur_car_persons,
            target_persons,
            target_employed,
            target_car_persons,
        )

        if current < best_objective:
            best_objective = current
            best_selected[:] = selected
            has_best = True
            if best_objective < convergence_threshold:
                break

        for _ in range(max_iter):
            out_pos = np.random.randint(target_households)
            in_pos = np.random.randint(non_selected_count)

            out_i = selected_idx[out_pos]
            in_i = non_selected_idx[in_pos]

            new_persons = cur_persons - persons[out_i] + persons[in_i]
            new_employed = cur_employed - employed[out_i] + employed[in_i]
            new_car_persons = cur_car_persons - car_available[out_i] + car_available[in_i]

            candidate = _objective(
                new_persons,
                new_employed,
                new_car_persons,
                target_persons,
                target_employed,
                target_car_persons,
            )

            if candidate < current or (candidate == current and np.random.random() < 0.01):
                selected[out_i] = False
                selected[in_i] = True

                selected_idx[out_pos] = in_i
                non_selected_idx[in_pos] = out_i

                cur_persons = new_persons
                cur_employed = new_employed
                cur_car_persons = new_car_persons
                current = candidate

                if current < best_objective:
                    best_objective = current
                    best_selected[:] = selected
                    has_best = True
                    if best_objective < convergence_threshold:
                        break

        if best_objective < convergence_threshold:
            break

    if not has_best:
        return np.empty(0, dtype=np.int64)

    result = np.empty(target_households, dtype=np.int64)
    cursor = 0
    for i in range(n):
        if best_selected[i]:
            if cursor < target_households:
                result[cursor] = i
                cursor += 1

    if cursor == target_households:
        return result

    fallback = np.empty(cursor, dtype=np.int64)
    for i in range(cursor):
        fallback[i] = result[i]
    return fallback


def _sample_balanced_group(group, target_households, probability, random):
    n = len(group)

    if target_households <= 0:
        return np.array([], dtype=group["_idx"].dtype)
    if target_households >= n:
        return group["_idx"].to_numpy()

    group_indices = group["_idx"].to_numpy(dtype=np.int64)
    persons = group["persons"].to_numpy(dtype=np.int64)
    employed = group["employed_persons"].to_numpy(dtype=np.int64)
    car_available = group["car_available_persons"].to_numpy(dtype=np.int64)

    seed = int(random.randint(0, np.iinfo(np.int32).max))
    selected_local_idx = _sample_balanced_indices_numba(
        persons,
        employed,
        car_available,
        int(target_households),
        probability,
        seed,
    )

    return group_indices[selected_local_idx]

def configure(context):
    context.stage("data.census.selected")
    
    context.config("input_downsampling")
    context.config("random_seed")


def execute(context):
    df = context.stage("data.census.selected")
    num_persons = len(df)

    probability = context.config("input_downsampling")

    if probability <= 0.0:
        logger.warning("Downsampling probability <= 0. Returning empty population.")
        return df.iloc[0:0].copy()
    if probability >= 1.0:
        return df

    aggregation_col = "home_zone_id" if "home_zone_id" in df.columns else "home_municipality_id"
    sampling_col = "household_id" if "household_id" in df.columns else "person_id"

    logger.info(f"Downsampling ({probability})")
    random = np.random.RandomState(context.config("random_seed"))

    hh = _household_features(df, sampling_col, aggregation_col).reset_index(drop=True)
    hh["_idx"] = hh.index

    logger.info(
        f"  Initial unique {sampling_col}: {hh.shape[0]}, persons: {df['person_id'].nunique()}",
    )

    groups = hh.groupby(aggregation_col, sort=False)
    household_counts = groups.size().values
    target_counts = _allocate_largest_remainder(household_counts, probability)

    kept_row_indices = []
    for ( _ , group_df ), target_n in context.progress(zip(groups, target_counts), label="Sampling population"):
        selected_idx = _sample_balanced_group(group_df, int(target_n), probability, random)
        if len(selected_idx) > 0:
            kept_row_indices.extend(selected_idx.tolist())

    kept_ids = hh.loc[kept_row_indices, sampling_col].values
    df = df[df[sampling_col].isin(kept_ids)].reset_index(drop=True)
    
    logger.info(f"  Sampled {sampling_col}: {len(kept_ids)}, persons: {df['person_id'].nunique()}")
    logger.info(f"Proportion of original population: {len(df) / num_persons}")


    return df
