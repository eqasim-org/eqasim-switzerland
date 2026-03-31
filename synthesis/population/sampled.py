import numpy as np
import logging
import pandas as pd

logger = logging.getLogger("synpp")


def _to_binary_indicator(series, positive_numeric=False):
    values = pd.Series(series)
    numeric = pd.to_numeric(values, errors="coerce")

    if positive_numeric:
        binary = np.where(numeric > 0.0, 1, 0)
    else:
        binary = np.where(numeric == 1.0, 1, 0)

    undecided = numeric.isna()
    if undecided.any():
        truthy = {
            "1",
            "true",
            "t",
            "yes",
            "y",
            "available",
            "always",
        }
        text = values.astype(str).str.strip().str.lower()
        binary = np.where(undecided & text.isin(truthy), 1, binary)

    return np.asarray(binary, dtype=int)


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
        employed = _to_binary_indicator(df["employed"], positive_numeric=False)
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
        # Person-level car availability: count how many agents in each sampled unit can use a car.
        car_values = _to_binary_indicator(df["car_availability"], positive_numeric=True)
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


def _objective(
    cur_persons,
    cur_employed,
    cur_car_persons,
    target_persons,
    target_employed,
    target_car_persons,
):
    wp = 1.0 / max(1.0, target_persons)
    we = 1.2 / max(1.0, target_employed)
    wc = 1.2 / max(1.0, target_car_persons)
    return (
        wp * abs(cur_persons - target_persons)
        + we * abs(cur_employed - target_employed)
        + wc * abs(cur_car_persons - target_car_persons)
    )


def _sample_balanced_group(group, target_households, random):
    n = len(group)
    if target_households <= 0:
        return np.array([], dtype=group["_idx"].dtype)
    if target_households >= n:
        return group["_idx"].values

    persons = group["persons"].to_numpy()
    employed = group["employed_persons"].to_numpy()
    car_available = group["car_available_persons"].to_numpy()

    ratio = target_households / n
    target_persons = int(round(persons.sum() * ratio))
    target_employed = int(round(employed.sum() * ratio))
    target_car_persons = int(round(car_available.sum() * ratio))

    n_restarts = max(4, min(12, int(np.ceil(np.log2(n + 1)) + 2)))
    max_iter = min(6000, 80 * target_households + 400)

    best_selected = None
    best_objective = np.inf

    for _ in range(n_restarts):
        selected = np.zeros(n, dtype=bool)
        selected[random.choice(n, size=target_households, replace=False)] = True

        cur_persons = persons[selected].sum()
        cur_employed = employed[selected].sum()
        cur_car_persons = car_available[selected].sum()
        current = _objective(
            cur_persons,
            cur_employed,
            cur_car_persons,
            target_persons,
            target_employed,
            target_car_persons,
        )

        for _ in range(max_iter):
            selected_idx = np.where(selected)[0]
            non_selected_idx = np.where(~selected)[0]

            out_i = selected_idx[random.randint(len(selected_idx))]
            in_i = non_selected_idx[random.randint(len(non_selected_idx))]

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

            if candidate < current or (candidate == current and random.rand() < 0.01):
                selected[out_i] = False
                selected[in_i] = True
                cur_persons = new_persons
                cur_employed = new_employed
                cur_car_persons = new_car_persons
                current = candidate

                if current < best_objective:
                    best_objective = current
                    best_selected = selected.copy()
                    if best_objective == 0.0:
                        break

        if best_objective == 0.0:
            break

    if best_selected is None:
        best_selected = selected

    return group.loc[best_selected, "_idx"].values

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
    for (group_key, group_df), target_n in zip(groups, target_counts):
        selected_idx = _sample_balanced_group(group_df.reset_index(drop=True), int(target_n), random)
        if len(selected_idx) > 0:
            kept_row_indices.extend(selected_idx.tolist())

        if group_df.shape[0] > 0:
            source_persons = group_df["persons"].sum()
            source_employed = group_df["employed_persons"].sum()
            source_car_persons = group_df["car_available_persons"].sum()
            kept = hh.loc[selected_idx] if len(selected_idx) > 0 else hh.iloc[0:0]
            logger.info(
                f"    {aggregation_col}={group_key}: hh {len(selected_idx)}/{len(group_df)} | "
                f"persons {int(kept['persons'].sum()) if len(selected_idx) > 0 else 0}/{int(source_persons)} | "
                f"employed {int(kept['employed_persons'].sum()) if len(selected_idx) > 0 else 0}/{int(source_employed)} | "
                f"car_persons {int(kept['car_available_persons'].sum()) if len(selected_idx) > 0 else 0}/{int(source_car_persons)}",
            )

    kept_ids = hh.loc[kept_row_indices, sampling_col].values
    df = df[df[sampling_col].isin(kept_ids)]
    logger.info(f"  Sampled {sampling_col}: {len(kept_ids)}, persons: {df['person_id'].nunique()}")
    logger.info(f"Proportion of original population: {len(df) / num_persons}")


    return df
