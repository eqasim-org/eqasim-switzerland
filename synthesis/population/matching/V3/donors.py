"""Order-independent categorical donor search, batched by recipient profile."""

import numpy as np
import pandas as pd
from numba import njit, prange


@njit(cache=True, parallel=True)
def _search(source, weights, profiles, mandatory_count, threshold, inverse, order, offsets, uniforms):
    assigned = np.full(len(inverse), -1, np.int64)
    summaries = np.zeros((len(profiles), 3), np.int64)
    for q in prange(len(profiles)):
        mandatory = np.zeros(len(source), np.int32)
        optional = np.zeros(len(source), np.int32)
        histogram = np.zeros((mandatory_count + 1, source.shape[1] - mandatory_count + 1), np.int64)
        for j in range(len(source)):
            for c in range(source.shape[1]):
                if source[j, c] != profiles[q, c]:
                    if c < mandatory_count:
                        mandatory[j] += 1
                    else:
                        optional[j] += 1
            histogram[mandatory[j], optional[j]] += 1
        # Mandatory preferences are relaxed only when their exact pool is empty.
        m = 0
        while histogram[m].sum() == 0:
            m += 1
        radius, size = 0, histogram[m, 0]
        while size < threshold and radius < histogram.shape[1] - 1:
            radius += 1
            size += histogram[m, radius]
        candidates = np.empty(size, np.int64)
        cumulative = np.empty(size, np.float64)
        total, k = 0.0, 0
        for j in range(len(source)):
            if mandatory[j] == m and optional[j] <= radius:
                candidates[k] = j
                total += weights[j]
                cumulative[k] = total
                k += 1
        for pos in range(offsets[q], offsets[q + 1]):
            row = order[pos]
            selected = min(np.searchsorted(cumulative, uniforms[row] * total, side="right"), size - 1)
            assigned[row] = candidates[selected]
        summaries[q, 0], summaries[q, 1], summaries[q, 2] = m, radius, size
    return assigned, summaries


def select_donors(source, target, columns, mandatory, hard, *, seed, threshold=10, weight="person_weight"):
    """Return donor IDs and diagnostics in target order; never relax hard fields.

    Searches each unique profile once, within its exact hard-attribute block.
    Comparisons use category equality, not the magnitude of category codes.
    No recipient-by-donor distance matrix is retained. Random draws happen
    outside parallel JIT code, making results independent of thread scheduling.
    """
    if threshold < 1 or len(set(columns)) != len(columns):
        raise ValueError("Positive donor threshold and unique feature names required")
    if not set(hard).issubset(columns) or not set(mandatory).issubset(columns):
        raise ValueError("Hard and mandatory attributes must be matching columns")
    if source.mz_id.duplicated().any() or target.person_id.duplicated().any():
        raise ValueError("Matching requires unique donor/recipient identifiers")
    # Stable donor/recipient order also makes column permutations reproducible.
    source = source.sort_values("mz_id").reset_index(drop=True)
    weights = pd.to_numeric(source[weight], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(weights) & (weights > 0)).all():
        raise ValueError("Survey sampling weights must be finite and positive")
    preferences = sorted(set(mandatory) - set(hard))
    optional = sorted(set(columns) - set(hard) - set(preferences))
    names = sorted(hard) + preferences + optional
    encoded = np.empty((len(source) + len(target), len(names)), np.int32)
    for c, name in enumerate(names):
        values = pd.concat([source[name], target[name]], ignore_index=True)
        if name in hard:
            values = values.mask(values.eq(-1))
        encoded[:, c] = pd.factorize(values, sort=True)[0]
    donors, people = encoded[:len(source)], encoded[len(source):]
    h = len(hard)
    source_groups = pd.DataFrame(donors[:, :h]).groupby(list(range(h)), sort=True).indices
    target_groups = pd.DataFrame(people[:, :h]).groupby(list(range(h)), sort=True).indices
    uniforms = np.empty(len(target))
    person_order = np.argsort(target.person_id.to_numpy(), kind="stable")
    uniforms[person_order] = np.random.RandomState(seed).random_sample(len(target))
    ids = np.full(len(target), -1, np.int64)
    metadata = np.full((len(target), 3), -1, np.int64)
    metadata[:, 2] = 0
    for key, target_rows in target_groups.items():
        # Missing hard attributes are not a compatible hard group.
        if any(v < 0 for v in (key if isinstance(key, tuple) else (key,))):
            continue
        donor_rows = source_groups.get(key)
        if donor_rows is None or not len(donor_rows):
            continue
        profiles, inverse = np.unique(people[target_rows, h:], axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(profiles)))]
        chosen, summary = _search(np.ascontiguousarray(donors[donor_rows, h:]), weights[donor_rows],
            profiles, len(preferences), int(threshold), inverse, order, offsets, uniforms[target_rows])
        ids[target_rows] = source.mz_id.to_numpy()[donor_rows[chosen]]
        metadata[target_rows] = summary[inverse]
    details = pd.DataFrame({"person_id": target.person_id.to_numpy(), "mz_person_id": ids,
        "mandatory_mismatches": metadata[:, 0], "optional_radius": metadata[:, 1], "donor_pool_size": metadata[:, 2]})
    details["below_threshold"] = (ids >= 0) & (metadata[:, 2] < threshold)
    return ids, details
