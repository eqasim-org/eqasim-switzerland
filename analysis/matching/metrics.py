"""Proper predictive scores and household-clustered uncertainty estimates."""

import numpy as np
import pandas as pd

PURPOSES = ("work", "education", "shop", "leisure", "other")
MODES = ("car", "car_passenger", "pt", "walk", "bike")


def diary_outcomes(source, trips):
    """Summarize a complete observed day, retaining order in an additional chain label.

    Non-travellers have zero totals. First/last times are structurally zero for
    them, and a separate mobility event distinguishes that from midnight travel.
    Invalid measured times contribute no duration and are counted in data quality.
    """
    records, labels = [], []
    groups = {pid: group.sort_values("trip_id") for pid, group in trips.groupby("person_id")}
    for pid in source["mz_id"]:
        group = groups.get(pid)
        row = {"trip_count": 0., "travel_minutes": 0., "distance_km": 0.,
               "first_departure_hour": 0., "last_arrival_hour": 0.}
        row.update({"count_" + p: 0. for p in PURPOSES})
        row.update({"minutes_" + p: 0. for p in PURPOSES})
        row.update({"mode_count_" + m: 0. for m in MODES})
        row.update({"event_" + key: 0. for key in ("mobile", "work", "education", "car", "pt", "leisure")})
        if group is not None and len(group):
            dep = pd.to_numeric(group["departure_time"], errors="coerce").to_numpy(dtype=float)
            arr = pd.to_numeric(group["arrival_time"], errors="coerce").to_numpy(dtype=float)
            dist = pd.to_numeric(group["crowfly_distance"], errors="coerce").to_numpy(dtype=float)
            duration = arr - dep
            purpose = group["purpose"].astype(str).str.replace("_secondary", "", regex=False).to_numpy()
            mode = group["mode"].astype(str).str.replace("_loop", "", regex=False).to_numpy()
            dwell = np.r_[dep[1:] - arr[:-1], np.nan]
            row.update(trip_count=float(len(group)),
                       travel_minutes=float(np.nansum(np.where(duration >= 0, duration, np.nan))) / 60,
                       distance_km=float(np.nansum(np.where(dist >= 0, dist, np.nan))) / 1000,
                       first_departure_hour=float(np.nanmin(dep)) / 3600 if np.isfinite(dep).any() else 0.,
                       last_arrival_hour=float(np.nanmax(arr)) / 3600 if np.isfinite(arr).any() else 0.)
            row["event_mobile"] = 1.
            for p in PURPOSES:
                row["count_" + p] = float(np.sum(purpose == p))
                row["minutes_" + p] = float(np.nansum(np.where((purpose == p) & (dwell >= 0), dwell, np.nan))) / 60
            for m in MODES:
                row["mode_count_" + m] = float(np.sum(mode == m))
            for p in ("work", "education", "leisure"):
                row["event_" + p] = float(np.any(purpose == p) or group["origin_purpose"].eq(p).any())
            row["event_car"] = float(np.any(mode == "car"))
            row["event_pt"] = float(np.any(mode == "pt"))
            origin = str(group.iloc[0]["origin_purpose"])
            labels.append(origin + "|" + "|".join(f"{m}>{p}" for m, p in zip(mode, purpose)))
        else:
            labels.append("home|no trip")
        records.append(row)
    return pd.DataFrame(records, index=source.index), np.asarray(labels, dtype=object)


def outcome_scale(outcomes, training_rows, weights):
    continuous = [col for col in outcomes if not col.startswith("event_")]
    values = outcomes[continuous].to_numpy(dtype=float)
    train = values[training_rows]
    weight = np.asarray(weights)[training_rows]
    mean = np.average(train, axis=0, weights=weight)
    var = np.average((train - mean) ** 2, axis=0, weights=weight)
    # Units with zero training variance get scale one; test outcomes never set scales.
    scale = np.where(var > 1e-8, np.sqrt(var), 1.)
    return continuous, values / scale, scale


def energy_score(actual, samples):
    first = np.linalg.norm(samples - actual[:, None], axis=2).mean(axis=1)
    norms = np.sum(samples * samples, axis=2)
    inner = np.einsum("nmd,nkd->nmk", samples, samples)
    summed_norms = norms[:, :, None] + norms[:, None, :]
    squared = np.maximum(summed_norms - 2 * inner, 0)
    squared[squared < 8 * np.finfo(float).eps * np.maximum(summed_norms, 1)] = 0
    pair_distance = np.sqrt(squared)
    return np.maximum(first - .5 * pair_distance.mean(axis=(1, 2)), 0)


def score_ensembles(observed, labels, truth_rows, donor_rows, scaled_values):
    """Score the empirical predictive law for each held-out respondent.

    Energy and marginal CRPS use the empirical-distribution (V-statistic)
    convention. Equal ensemble sizes are required when comparing algorithms.
    Unmatched predictions retain missing scores and explicit zero coverage.
    """
    n, count = donor_rows.shape
    valid = (donor_rows >= 0).all(axis=1)
    result = {"covered": valid.astype(float)}

    def add(name, values):
        column = np.full(n, np.nan)
        column[valid] = values
        result[name] = column
    if not valid.any():
        result["energy"] = np.nan
        return pd.DataFrame(result)
    target = truth_rows[valid]
    chosen = donor_rows[valid]
    samples = scaled_values[chosen]
    actual = scaled_values[target]
    add("energy", energy_score(actual, samples))
    # A dependence-sensitive complement to the energy score.
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)]
    vs = np.zeros(len(target))
    for a, b in pairs:
        empirical = np.sqrt(np.abs(samples[:, :, a] - samples[:, :, b])).mean(axis=1)
        reference = np.sqrt(np.abs(actual[:, a] - actual[:, b]))
        vs += (empirical - reference) ** 2
    add("variogram", vs / len(pairs))
    for column in observed.columns:
        values = observed[column].to_numpy(dtype=float)
        y, pred = values[target], values[chosen]
        if column.startswith("event_"):
            probability = pred.mean(axis=1)
            add("brier_" + column[6:], (probability - y) ** 2)
            add("prob_" + column[6:], probability)
            add("truth_" + column[6:], y)
        else:
            sorted_pred = np.sort(pred, axis=1)
            coefficient = 2 * np.arange(1, count + 1) - count - 1
            crps = np.abs(pred - y[:, None]).mean(axis=1) - (sorted_pred * coefficient).sum(axis=1) / count**2
            add("crps_" + column, crps)
            add("mean_" + column, pred.mean(axis=1))
            add("truth_" + column, y)
            # Randomized ranks are produced later with the run's explicit RNG.
            add("less_" + column, (pred < y[:, None]).sum(axis=1))
            add("equal_" + column, (pred == y[:, None]).sum(axis=1))
            for coverage in (.5, .8, .95):
                lower, upper = np.quantile(pred, [(1 - coverage) / 2, (1 + coverage) / 2], axis=1)
                add(f"coverage_{coverage}_{column}", ((y >= lower) & (y <= upper)).astype(float))
                add(f"width_{coverage}_{column}", upper - lower)
    chosen_labels = labels[chosen]
    same = chosen_labels[:, :, None] == chosen_labels[:, None, :]
    add("chain_brier", same.mean(axis=(1, 2)) - 2 * (chosen_labels == labels[target, None]).mean(axis=1) + 1)
    return pd.DataFrame(result)


def clustered_interval(values, weights, groups, rng, n_bootstrap=500, *, denominator=None):
    """Ratio-of-weighted-totals interval, resampling independent households."""
    if n_bootstrap < 1:
        raise ValueError("The number of bootstrap replicates must be positive")
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    denominator = np.ones(len(values)) if denominator is None else np.asarray(denominator, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    valid &= np.isfinite(denominator) & (denominator >= 0)
    if not valid.any():
        return np.nan, np.nan, np.nan, 0, 0.
    values, weights, groups = values[valid], weights[valid], np.asarray(groups)[valid]
    codes, unique = pd.factorize(groups)
    totals = np.bincount(codes, weights=values * weights)
    mass = np.bincount(codes, weights=weights * denominator[valid])
    if mass.sum() <= 0:
        return np.nan, np.nan, np.nan, len(unique), float(weights.sum()**2 / np.sum(weights**2))
    estimate = float(totals.sum() / mass.sum())
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        draw = rng.randint(0, len(unique), len(unique))
        sampled_mass = mass[draw].sum()
        samples[i] = totals[draw].sum() / sampled_mass if sampled_mass else np.nan
    low, high = np.nanquantile(samples, [.025, .975])
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    return estimate, float(low), float(high), len(unique), ess
