import numpy as np
import logging
import pandas as pd
import numba

logger = logging.getLogger("synpp")

BALANCE_COLS = [
    "employed",
    "car_availability",
    "sex",
    "age_class",
    "ovgk",
    "income_class",
    "driving_license",
    "is_student",
]

WEIGHTS = dict(
    employed=2.0,
    car_availability=1.0,
    sex=1.5,
    age_class=1.0,
    ovgk=1.0,
    income_class=1.0,
    driving_license=1.0,
    is_student=1.0,
)

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

    return hh


def _build_household_balance_matrix(df, hh, sampling_col, balance_columns):
    matrices = []
    used_columns = []

    for column in balance_columns:
        if column not in df.columns:
            continue

        categorical = pd.Categorical(df[column].astype("string").fillna("__nan__"))
        n_categories = len(categorical.categories)

        if n_categories <= 1:
            continue

        encoded = pd.DataFrame({
            sampling_col: df[sampling_col].values,
            "_category": categorical.codes,
        })

        counts = (
            encoded.groupby([sampling_col, "_category"]).size()
            .unstack("_category", fill_value=0)
            .reindex(hh[sampling_col], fill_value=0)
        )

        matrices.append(counts.to_numpy(dtype=np.int64))
        used_columns.append((column, n_categories))

    if len(matrices) == 0:
        return np.zeros((len(hh), 0), dtype=np.int64), used_columns, np.zeros(0, dtype=np.int64)

    block_sizes = np.asarray([matrix.shape[1] for matrix in matrices], dtype=np.int64)

    return np.concatenate(matrices, axis=1), used_columns, block_sizes


def _compute_target_balance(balance_counts, probability, block_sizes):
    target_balance = np.zeros(balance_counts.shape[1], dtype=np.int64)

    cursor = 0
    for block_size in block_sizes:
        end = cursor + int(block_size)
        if end > cursor:
            block_counts = balance_counts[:, cursor:end].sum(axis=0)
            target_balance[cursor:end] = _allocate_largest_remainder(block_counts, probability).astype(np.int64)
        cursor = end

    return target_balance

@numba.njit(cache=True)
def _objective(
    cur_persons,
    cur_balance,
    target_persons,
    target_balance,
    balance_block_starts,
    balance_block_sizes,
    balance_block_weights,
):
    person_den = max(1.0, target_persons * target_persons)
    person_diff = cur_persons - target_persons
    score = (person_diff * person_diff) / person_den

    n_blocks = balance_block_sizes.shape[0]
    if n_blocks == 0:
        return score

    weighted_balance_score = 0.0
    total_weight = 0.0
    for b in range(n_blocks):
        start = balance_block_starts[b]
        end = start + balance_block_sizes[b]
        block_size = max(1, balance_block_sizes[b])
        block_score = 0.0

        for i in range(start, end):
            den = max(1.0, target_balance[i] * target_balance[i])
            diff = cur_balance[i] - target_balance[i]
            block_score += (diff * diff) / den

        block_weight = max(0.0, balance_block_weights[b])
        weighted_balance_score += block_weight * (block_score / block_size)
        total_weight += block_weight

    if total_weight <= 0.0:
        return score

    return score + weighted_balance_score / total_weight


@numba.njit(cache=True)
def _sample_balanced_indices_numba(
    persons,
    balance_counts,
    target_balance,
    balance_block_starts,
    balance_block_sizes,
    balance_block_weights,
    target_households,
    probability,
    seed,
):
    n = persons.shape[0]
    n_balance = balance_counts.shape[1]
    convergence_threshold = 1e-3

    if target_households <= 0:
        return np.empty(0, dtype=np.int64)
    if target_households >= n:
        return np.arange(n, dtype=np.int64)

    np.random.seed(seed)

    target_persons = int(np.rint(persons.sum() * probability))
    n_restarts = max(4, min(16, int(np.ceil(np.log2(n + 1)) * 3 + 4)))
    max_iter = int(min(200000, (4.0 / probability) * target_households + 10000))
    non_selected_count = n - target_households

    best_objective = np.inf
    best_selected = np.zeros(n, dtype=np.bool_)
    has_best = False

    for _restart_idx in range(n_restarts):
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
        cur_balance = np.zeros(n_balance, dtype=np.int64)

        for i in range(target_households):
            idx = selected_idx[i]
            cur_persons += persons[idx]
            for j in range(n_balance):
                cur_balance[j] += balance_counts[idx, j]

        current = _objective(
            cur_persons,
            cur_balance,
            target_persons,
            target_balance,
            balance_block_starts,
            balance_block_sizes,
            balance_block_weights,
        )

        initial_temp = max(1e-8, 0.05 * max(1.0, current))

        if current < best_objective:
            best_objective = current
            best_selected[:] = selected
            has_best = True
            if best_objective < convergence_threshold:
                break

        for iter_idx in range(max_iter):
            out_pos = np.random.randint(target_households)
            in_pos = np.random.randint(non_selected_count)

            out_i = selected_idx[out_pos]
            in_i = non_selected_idx[in_pos]

            new_persons = cur_persons - persons[out_i] + persons[in_i]
            new_balance = np.empty(n_balance, dtype=np.int64)
            for j in range(n_balance):
                new_balance[j] = cur_balance[j] - balance_counts[out_i, j] + balance_counts[in_i, j]

            candidate = _objective(
                new_persons,
                new_balance,
                target_persons,
                target_balance,
                balance_block_starts,
                balance_block_sizes,
                balance_block_weights,
            )

            delta = candidate - current
            accept = False
            if delta <= 0.0:
                accept = True
            else:
                temperature = max(1e-8, initial_temp * (1.0 - (iter_idx / max_iter)))
                if np.random.random() < np.exp(-delta / temperature):
                    accept = True

            if accept:
                selected[out_i] = False
                selected[in_i] = True

                selected_idx[out_pos] = in_i
                non_selected_idx[in_pos] = out_i

                cur_persons = new_persons
                cur_balance = new_balance
                current = candidate

                if current < best_objective:
                    best_objective = current
                    best_selected[:] = selected
                    has_best = True
                    if best_objective < convergence_threshold:
                        break

        if best_objective < convergence_threshold:
            break

    if has_best and non_selected_count > 0:
        selected_idx = np.empty(target_households, dtype=np.int64)
        non_selected_idx = np.empty(non_selected_count, dtype=np.int64)

        selected_cursor = 0
        non_selected_cursor = 0
        for i in range(n):
            if best_selected[i]:
                selected_idx[selected_cursor] = i
                selected_cursor += 1
            else:
                non_selected_idx[non_selected_cursor] = i
                non_selected_cursor += 1

        cur_persons = 0
        cur_balance = np.zeros(n_balance, dtype=np.int64)
        for i in range(target_households):
            idx = selected_idx[i]
            cur_persons += persons[idx]
            for j in range(n_balance):
                cur_balance[j] += balance_counts[idx, j]

        current = _objective(
            cur_persons,
            cur_balance,
            target_persons,
            target_balance,
            balance_block_starts,
            balance_block_sizes,
            balance_block_weights,
        )

        improve_iter = int(min(300000, 15000 + (4.0 / probability) * target_households))
        for _improve_idx in range(improve_iter):
            out_pos = np.random.randint(target_households)
            in_pos = np.random.randint(non_selected_count)

            out_i = selected_idx[out_pos]
            in_i = non_selected_idx[in_pos]

            new_persons = cur_persons - persons[out_i] + persons[in_i]
            new_balance = np.empty(n_balance, dtype=np.int64)
            for j in range(n_balance):
                new_balance[j] = cur_balance[j] - balance_counts[out_i, j] + balance_counts[in_i, j]

            candidate = _objective(
                new_persons,
                new_balance,
                target_persons,
                target_balance,
                balance_block_starts,
                balance_block_sizes,
                balance_block_weights,
            )

            if candidate < current:
                selected_idx[out_pos] = in_i
                non_selected_idx[in_pos] = out_i
                best_selected[out_i] = False
                best_selected[in_i] = True

                cur_persons = new_persons
                cur_balance = new_balance
                current = candidate
                best_objective = candidate

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


def _sample_balanced_group(
    group,
    target_households,
    probability,
    random,
    group_balance_counts,
    balance_block_sizes,
    balance_block_weights,
):
    n = len(group)

    if target_households <= 0:
        return np.array([], dtype=group["_idx"].dtype)
    if target_households >= n:
        return group["_idx"].to_numpy()

    group_indices = group["_idx"].to_numpy(dtype=np.int64)
    persons = group["persons"].to_numpy(dtype=np.int64)
    balance_counts = np.asarray(group_balance_counts, dtype=np.int64)
    block_sizes = np.asarray(balance_block_sizes, dtype=np.int64)

    block_starts = np.zeros(block_sizes.shape[0], dtype=np.int64)
    cursor = 0
    for i in range(block_sizes.shape[0]):
        block_starts[i] = cursor
        cursor += block_sizes[i]

    target_balance = _compute_target_balance(balance_counts, probability, block_sizes)

    seed = int(random.randint(0, np.iinfo(np.int32).max))
    selected_local_idx = _sample_balanced_indices_numba(
        persons,
        balance_counts,
        target_balance,
        block_starts,
        block_sizes,
        balance_block_weights,
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

    balance_columns = BALANCE_COLS.copy()
    household_balance_counts, used_balance_columns, balance_block_sizes = _build_household_balance_matrix(
        df,
        hh,
        sampling_col,
        balance_columns,
    )
    balance_block_weights = np.asarray(
        [float(WEIGHTS.get(column, 1.0)) for column, _ in used_balance_columns],
        dtype=np.float64,
    )

    if len(used_balance_columns) == 0:
        logger.warning("No additional balance columns available for downsampling objective.")
    else:
        details = ", ".join([f"{column}({n_categories})" for column, n_categories in used_balance_columns])
        logger.info(f"  Balancing municipality shares for: {details}")
        weight_details = ", ".join([f"{column}={WEIGHTS.get(column, 1.0)}" for column, _ in used_balance_columns])
        logger.info(f"  Balance weights: {weight_details}")

    logger.info(
        f"  Initial unique {sampling_col}: {hh.shape[0]}, persons: {df['person_id'].nunique()}",
    )

    groups = hh.groupby(aggregation_col, sort=False)
    household_counts = groups.size().values
    target_counts = _allocate_largest_remainder(household_counts, probability)

    kept_row_indices = []
    for (_, group_df), target_n in context.progress(zip(groups, target_counts), label="Sampling population"):
        group_row_idx = group_df["_idx"].to_numpy(dtype=np.int64)
        group_balance_counts = household_balance_counts[group_row_idx]
        selected_idx = _sample_balanced_group(
            group_df,
            int(target_n),
            probability,
            random,
            group_balance_counts,
            balance_block_sizes,
            balance_block_weights,
        )
        if len(selected_idx) > 0:
            kept_row_indices.extend(selected_idx.tolist())

    kept_ids = hh.loc[kept_row_indices, sampling_col].values
    df = df[df[sampling_col].isin(kept_ids)].reset_index(drop=True)
    
    logger.info(f"  Sampled {sampling_col}: {len(kept_ids)}, persons: {df['person_id'].nunique()}")
    logger.info(f"Proportion of original population: {len(df) / num_persons}")


    return df
