"""Borrow nearby observed flows and independent parent evidence for tertiary origins."""

import numpy as np
import pandas as pd

# Best weighted held-out-year log loss among the tested settings.
DEFAULT_PRIOR_STRENGTH = 100.0
DEFAULT_NEIGHBORS = 50
DEFAULT_BANDWIDTH = 20000.0
# One shared weak penalty, selected from 0/.005/.01/.02 by held-out-year
# destination prediction. .005 improved log loss and Brier in all three years;
# stronger penalties did not. No age- or campus-specific coefficients are fit.
DEFAULT_DISTANCE_DECAY = .005  # per kilometre; e-folding distance 200 km


def regularise_distances(profiles, zones, municipalities, quarters, decay, observed_share=None):
    """Prefer nearer supported destinations without introducing structural zeros."""
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("Tertiary distance decay must be finite and nonnegative")
    if decay == 0:
        return profiles.copy()
    centres = {}
    for level, frame, field in [("municipality", municipalities, "municipality_id"),
                                ("quarter", quarters, "quarter_id")]:
        geometry = frame.to_crs(2056).geometry.centroid
        points = dict(zip(frame[field], zip(geometry.x, geometry.y)))
        for row in zones[zones.zone_level.eq(level)].itertuples():
            centres[row.zone_id] = points[row.zone_level_id]
    origin = np.array([centres[i] for i in profiles.index])
    destination = np.array([centres[i] for i in profiles.columns])
    distance = np.hypot(origin[:, 0, None] - destination[None, :, 0],
                        origin[:, 1, None] - destination[None, :, 1]) / 1000.
    if not np.isfinite(distance).all():
        raise ValueError("Tertiary distance regularisation requires finite zone centroids")
    # Strong local evidence needs less geographical regularisation. Reuse the
    # existing effective-sample-size shrinkage share; no new per-origin fit.
    uncertainty = np.ones(len(profiles)) if observed_share is None else 1. - np.asarray(observed_share, dtype=float)
    if uncertainty.shape != (len(profiles),) or not np.isfinite(uncertainty).all() or ((uncertainty < 0) | (uncertainty > 1)).any():
        raise ValueError("Observed origin shares must be finite probabilities matching profile rows")
    # Centre each row on its nearest supported destination for stable weights.
    values = profiles.to_numpy()
    minimum = np.where(values > 0, distance, np.inf).min(axis=1)
    weights = np.exp(np.maximum(-decay * uncertainty[:, None] * np.maximum(distance - minimum[:, None], 0.), -700.))
    values = values * weights
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=profiles.index, columns=profiles.columns)


def estimate_profiles(df_od, df_zones, df_municipalities, df_quarters, municipality_mapping,
                      prior_strength=DEFAULT_PRIOR_STRENGTH, neighbors=DEFAULT_NEIGHBORS,
                      bandwidth=DEFAULT_BANDWIDTH):
    """Shrink origin flows toward nearby observed origins or independent parent data.

    Effective sample size controls how much empirical evidence survives. A zone
    with no respondents can still receive students in the synthetic population;
    its modelled prior is recorded separately from observed commuting evidence.
    """
    if not np.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("Tertiary OD prior strength must be finite and nonnegative")
    if not np.isfinite(neighbors) or neighbors < 1 or int(neighbors) != neighbors:
        raise ValueError("Tertiary OD neighbor count must be a positive integer")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Tertiary OD bandwidth must be finite and positive")

    destinations = sorted(df_od["education_zone_id"].unique())
    if not destinations:
        raise ValueError("No usable domestic tertiary observations for OD fallback")
    origins = df_zones[df_zones["zone_level"].isin(["municipality", "quarter"])].copy()
    counts = np.zeros((len(origins), len(destinations)))
    respondents = np.zeros(len(origins), dtype=int)
    squared_weights = np.zeros(len(origins))
    for level in ["municipality", "quarter"]:
        field = "home_" + level + "_id"
        selected = origins["zone_level"].eq(level).to_numpy()
        ids = origins.loc[selected, "zone_level_id"]
        table = df_od.groupby([field, "education_zone_id"])["weight"].sum().unstack(fill_value=0)
        counts[selected] = table.reindex(index=ids, columns=destinations, fill_value=0).to_numpy()
        stats = df_od.assign(squared_weight=df_od["weight"] ** 2).groupby(field).agg(
            respondents=("weight", "size"), squared_weight=("squared_weight", "sum"))
        stats = stats.reindex(ids, fill_value=0)
        respondents[selected] = stats["respondents"].to_numpy()
        squared_weights[selected] = stats["squared_weight"].to_numpy()

    totals = counts.sum(axis=1)
    effective_n = np.divide(totals ** 2, squared_weights,
                            out=np.zeros(len(origins)), where=squared_weights > 0)
    empirical = np.divide(counts, totals[:, None], out=np.zeros_like(counts), where=totals[:, None] > 0)
    profiles = np.zeros_like(empirical)
    diagnostics = []

    municipality_indices = np.flatnonzero(origins["zone_level"].eq("municipality"))
    municipality_ids = origins.iloc[municipality_indices]["zone_level_id"]
    municipality_index = dict(zip(municipality_ids, municipality_indices))
    centroids = df_municipalities.set_index("municipality_id").geometry.centroid.reindex(municipality_ids)
    municipality_coordinates = np.column_stack([centroids.x, centroids.y])
    donors = municipality_indices[totals[municipality_indices] > 0]
    donor_positions = np.flatnonzero(totals[municipality_indices] > 0)
    donor_coordinates = municipality_coordinates[donor_positions]

    def regional_prior(coordinate, excluded):
        distances = np.linalg.norm(donor_coordinates - coordinate, axis=1)
        eligible = ~np.isin(donors, excluded) & np.isfinite(distances)
        chosen = np.flatnonzero(eligible)
        chosen = chosen[np.argsort(distances[chosen], kind="stable")[:int(neighbors)]]
        if not len(chosen):
            return None, 0, np.nan
        # Limit donor influence so a large city does not overwhelm nearby towns.
        log_weights = -0.5 * (distances[chosen] / bandwidth) ** 2
        weights = np.exp(log_weights - log_weights.max()) * np.minimum(effective_n[donors[chosen]], 10.0)
        weights /= weights.sum()
        return weights @ empirical[donors[chosen]], len(chosen), distances[chosen].max()

    for position, index in enumerate(municipality_indices):
        prior, donor_count, furthest = regional_prior(municipality_coordinates[position], [index])
        alpha = effective_n[index] / (effective_n[index] + prior_strength) if effective_n[index] > 0 else 0.0
        if prior is None:
            if totals[index] == 0:
                raise ValueError("No observed nearby municipality for tertiary origin %s" % origins.iloc[index]["zone_id"])
            prior = empirical[index]
            alpha = 1.0
        profiles[index] = alpha * empirical[index] + (1 - alpha) * prior
        diagnostics.append((index, "nearby_municipalities" if donor_count else "observed_only",
                            alpha, donor_count, furthest))

    parent_mapping = municipality_mapping.set_index("deprecated_municipality_id")["municipality_id"]
    quarters = df_quarters.set_index("quarter_id")
    for index in np.flatnonzero(origins["zone_level"].eq("quarter")):
        quarter_id = origins.iloc[index]["zone_level_id"]
        parent_id = parent_mapping.get(quarters.loc[quarter_id, "source_municipality_id"])
        parent_index = municipality_index.get(parent_id)
        prior = None
        donor_count, furthest = 0, np.nan
        # Remove this quarter from its parent's evidence; do not count it twice.
        if parent_index is not None:
            residual = counts[parent_index] - counts[index]
            if np.any(residual < -1e-6):
                raise ValueError("Quarter observations do not belong to their mapped municipality: %s" % quarter_id)
            residual = np.maximum(residual, 0)
            if residual.sum() > 0:
                parent_prior = residual / residual.sum()
                residual_squares = max(0.0, squared_weights[parent_index] - squared_weights[index])
                parent_effective_n = residual.sum() ** 2 / residual_squares if residual_squares else 0.0
                parent_alpha = parent_effective_n / (parent_effective_n + prior_strength) if parent_effective_n > 0 else 0.0
                regional, donor_count, furthest = regional_prior(
                    np.array([quarters.loc[quarter_id].geometry.centroid.x,
                              quarters.loc[quarter_id].geometry.centroid.y]), [parent_index])
                prior = parent_prior if regional is None else parent_alpha * parent_prior + (1 - parent_alpha) * regional
        if prior is None:
            prior, donor_count, furthest = regional_prior(
                np.array([quarters.loc[quarter_id].geometry.centroid.x,
                          quarters.loc[quarter_id].geometry.centroid.y]),
                [parent_index] if parent_index is not None else [])
            source = "nearby_municipalities"
        else:
            source = "parent_excluding_quarter"
        alpha = effective_n[index] / (effective_n[index] + prior_strength) if effective_n[index] > 0 else 0.0
        if prior is None:
            if totals[index] == 0:
                raise ValueError("No independent tertiary evidence for quarter %s" % quarter_id)
            prior, alpha, source = empirical[index], 1.0, "observed_only"
        profiles[index] = alpha * empirical[index] + (1 - alpha) * prior
        diagnostics.append((index, source, alpha, donor_count, furthest))

    diagnostics = pd.DataFrame(diagnostics, columns=["index", "prior_source", "observed_share", "regional_donors", "furthest_donor_m"])
    diagnostics = diagnostics.set_index("index").reindex(range(len(origins)))
    diagnostics["respondents"] = respondents
    diagnostics["effective_n"] = effective_n
    diagnostics["fallback_share"] = 1 - diagnostics["observed_share"]
    diagnostics = origins.reset_index(drop=True).join(diagnostics).drop(columns=["zone_level_id"])
    return pd.DataFrame(profiles, index=origins["zone_id"], columns=destinations), diagnostics


def build_smoothed_matrix(df_od, df_zones, df_municipalities, df_quarters, municipality_mapping,
                          prior_strength=DEFAULT_PRIOR_STRENGTH, neighbors=DEFAULT_NEIGHBORS,
                          bandwidth=DEFAULT_BANDWIDTH, distance_decay=DEFAULT_DISTANCE_DECAY):
    """Place supported profiles in the exact zone order consumed by assignment."""
    profiles, diagnostics = estimate_profiles(
        df_od, df_zones, df_municipalities, df_quarters, municipality_mapping,
        prior_strength, neighbors, bandwidth)
    profiles = regularise_distances(profiles, df_zones, df_municipalities, df_quarters,
                                   distance_decay, diagnostics["observed_share"].to_numpy())
    diagnostics["distance_decay_per_km"] = distance_decay
    zone_positions = {zone_id: i for i, zone_id in enumerate(df_zones["zone_id"])}
    rows = [zone_positions[zone_id] for zone_id in profiles.index]
    columns = [zone_positions[zone_id] for zone_id in profiles.columns]
    pdf = np.full((len(df_zones), len(df_zones)), np.nan)
    pdf[rows] = 0
    pdf[np.ix_(rows, columns)] = profiles.to_numpy()
    cdf = np.cumsum(pdf, axis=1)
    cdf[rows, -1] = 1.0
    return pdf, cdf, diagnostics
