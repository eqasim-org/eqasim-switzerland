"""Keep each municipality whole using resident-plurality continuous contour labels."""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ..activity_chain_features import FEATURES, CLUSTER_COLUMN, effective_sample
from .contours import assign_points, sample_fields, check_partition

CITY_IDS = [351, 6621, 5586, 261, 2701, 230, 3203]


def municipal_units(municipalities, border):
    border = shapely.force_2d(border)
    units = (
        municipalities[["municipality_id", "municipality_name", "geometry"]]
        .to_crs(2056)
        .copy()
    )
    if units.municipality_id.isna().any() or units.municipality_id.duplicated().any():
        raise ValueError("Municipality identifiers must be unique and present")
    keep = units.municipality_id.gt(0) & shapely.covers(
        border, units.geometry.representative_point().to_numpy()
    )
    units = units.loc[keep].sort_values("municipality_id").reset_index(drop=True)
    units.geometry = shapely.intersection(
        shapely.make_valid(shapely.force_2d(units.geometry.to_numpy())), border
    )
    units = units.loc[units.geometry.area.gt(0)].reset_index(drop=True)
    if (
        abs(units.geometry.area.sum() - units.geometry.union_all().area)
        > border.area * 1e-9
    ):
        raise ValueError("Municipalities have overlapping area")
    return units


def point_units(xy, units):
    """Locate points; tie breaks follow municipality ID, with bounded working memory."""
    if not units.municipality_id.is_monotonic_increasing:
        raise ValueError("Municipalities must be sorted by identifier")
    geometries = units.geometry.to_numpy()
    shapely.prepare(geometries)
    tree = shapely.STRtree(geometries)
    result = np.full(len(xy), len(units), dtype=np.int32)
    for start in range(0, len(xy), 100000):
        points = shapely.points(xy[start : start + 100000])
        pairs = tree.query(points)
        hit = shapely.intersects(geometries[pairs[1]], points[pairs[0]])
        np.minimum.at(result[start : start + len(points)], pairs[0, hit], pairs[1, hit])
    result[result == len(units)] = -1
    return result


def population_homes(population, border):
    eligible = (
        population.population_type.eq(1)
        & population.type_of_residence.eq(1)
        & population.age.ge(6)
    )
    x, y = population.home_x.to_numpy(float), population.home_y.to_numpy(float)
    valid = eligible.to_numpy() & np.isfinite(x) & np.isfinite(y)
    xmin, ymin, xmax, ymax = border.bounds
    valid &= (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    ids = np.flatnonzero(valid)
    ids = ids[shapely.intersects_xy(border, x[ids], y[ids])]
    # Count repeated coordinates as separate residents, rather than one household vote.
    homes = (
        pd.DataFrame({"home_x": x[ids], "home_y": y[ids]})
        .value_counts(sort=False)
        .rename("population")
        .reset_index()
    )
    return homes[["home_x", "home_y"]].to_numpy(), homes.population.to_numpy(np.int64)


def municipality_votes(units, source_regions, membership, original, mass):
    located = membership >= 0
    votes = (
        np.bincount(
            membership[located] * 4 + original[located] - 1,
            weights=mass[located],
            minlength=len(units) * 4,
        )
        .reshape(-1, 4)
        .astype(np.int64)
    )
    area_votes = np.column_stack(
        [
            shapely.area(shapely.intersection(units.geometry.to_numpy(), row.geometry))
            for row in source_regions.sort_values(CLUSTER_COLUMN).itertuples()
        ]
    )
    counts = votes.sum(axis=1)
    winners = votes.argmax(axis=1) + 1
    empty = counts == 0
    winners[empty] = area_votes[empty].argmax(axis=1) + 1
    result = units.copy()
    for j in range(4):
        result[f"residents_group_{j + 1}"] = votes[:, j]
        result[f"area_share_group_{j + 1}"] = (
            area_votes[:, j] / result.geometry.area.to_numpy()
        )
    result[CLUSTER_COLUMN] = winners
    result["population"] = counts
    result["winning_population"] = votes[np.arange(len(units)), winners - 1]
    result["winning_share"] = np.divide(
        result.winning_population.to_numpy(),
        counts,
        out=np.full(len(units), np.nan),
        where=counts > 0,
    )
    result["changed_population"] = counts - result.winning_population
    result["original_resident_groups"] = (votes > 0).sum(axis=1)
    result["original_area_groups"] = (
        area_votes > np.maximum(result.geometry.area.to_numpy()[:, None] * 1e-9, 0.001)
    ).sum(axis=1)
    result["assignment_method"] = np.where(
        empty, "largest contour area (no eligible residents)", "largest resident count"
    )
    result["tied_resident_vote"] = ~empty & (
        (votes == votes.max(axis=1)[:, None]).sum(axis=1) > 1
    )
    return result


def municipalize_contours(source, population, municipalities, border, diaries):
    border = shapely.force_2d(border)
    units = municipal_units(municipalities, border)
    xy, mass = population_homes(population, border)
    if mass.sum() == 0:
        raise ValueError("Municipal assignment requires located eligible residents")
    if mass.sum() != source["profiles"].population.sum():
        raise ValueError(
            "Municipal voting and original profiles must use the same resident population"
        )
    original = assign_points(source["regions"], xy)
    membership = point_units(xy, units)
    units = municipality_votes(units, source["regions"], membership, original, mass)

    # Dissolve entire municipal polygons and retain original contours only in uncovered areas.
    uncovered = border.difference(units.geometry.union_all())
    rows = []
    for group in range(1, 5):
        geometry = units.loc[units[CLUSTER_COLUMN].eq(group)].geometry.union_all()
        original_geometry = (
            source["regions"]
            .loc[source["regions"][CLUSTER_COLUMN].eq(group)]
            .geometry.union_all()
        )
        geometry = shapely.union_all(
            [geometry, uncovered.intersection(original_geometry)]
        )
        rows.append({CLUSTER_COLUMN: group, "geometry": geometry})
    regions = gpd.GeoDataFrame(rows, crs=2056)
    check_partition(regions, border)
    for row in regions.itertuples():
        members = units.loc[units[CLUSTER_COLUMN].eq(getattr(row, CLUSTER_COLUMN))]
        missing = shapely.area(
            shapely.difference(members.geometry.to_numpy(), row.geometry)
        )
        if (
            missing > np.maximum(0.001, members.geometry.area.to_numpy() * 1e-10)
        ).any():
            raise RuntimeError(
                "A municipality was split by the final region geometries"
            )

    # Recalculate resident and diary profiles after the municipal assignments change.
    labels = original.copy()
    labels[membership >= 0] = units[CLUSTER_COLUMN].to_numpy()[
        membership[membership >= 0]
    ]
    survey_xy = diaries[["home_x", "home_y"]].to_numpy()
    survey_original = assign_points(source["regions"], survey_xy)
    survey_membership = point_units(survey_xy, units)
    survey_labels = survey_original.copy()
    survey_labels[survey_membership >= 0] = units[CLUSTER_COLUMN].to_numpy()[
        survey_membership[survey_membership >= 0]
    ]
    predicted = sample_fields(source["x"], source["y"], source["fields"], xy)
    profiles = []
    for group in range(1, 5):
        mask = labels == group
        observed = diaries.loc[survey_labels == group]
        if not mask.any() or observed.empty:
            raise ValueError(
                "Municipal snapping must retain four populated, survey-supported groups"
            )
        row = {
            CLUSTER_COLUMN: group,
            "population": int(mass[mask].sum()),
            "population_share": float(mass[mask].sum() / mass.sum()),
            "survey_persons": len(observed),
            "survey_effective_sample": effective_sample(
                observed.person_weight.to_numpy()
            ),
        }
        for j, feature in enumerate(FEATURES):
            row[feature] = float(np.average(predicted[mask, j], weights=mass[mask]))
            row["survey_" + feature] = float(
                np.average(observed[feature], weights=observed.person_weight)
            )
        profiles.append(row)
    profiles = pd.DataFrame(profiles)
    regions = regions.merge(profiles, on=CLUSTER_COLUMN, validate="one_to_one")
    comparison = pd.concat(
        [
            source["profiles"].assign(method="Continuous contours"),
            profiles.assign(method="Whole municipalities"),
        ],
        ignore_index=True,
    )
    city_comparison = (
        units.loc[units.municipality_id.isin(CITY_IDS)].drop(columns="geometry").copy()
    )
    city_comparison["final_resident_groups"] = 1
    city_comparison["final_area_groups"] = 1
    moved = int(mass[labels != original].sum())
    flow = np.bincount(
        (original - 1) * 4 + labels - 1, weights=mass, minlength=16
    ).astype(np.int64)
    population_flows = pd.DataFrame(
        {
            "original_group": np.repeat(np.arange(1, 5), 4),
            "municipal_group": np.tile(np.arange(1, 5), 4),
            "population": flow,
        }
    )
    metadata = {
        "method": "Whole municipality assigned to the original contour group containing the largest eligible resident count",
        "population": int(mass.sum()),
        "fitting_diaries": len(diaries),
        "municipal_units": len(units),
        "populated_units": int(units.population.gt(0).sum()),
        "empty_units": int(units.population.eq(0).sum()),
        "previously_split_resident_units": int(
            units.original_resident_groups.gt(1).sum()
        ),
        "previously_split_area_units": int(units.original_area_groups.gt(1).sum()),
        "moved_population": moved,
        "moved_population_share": float(moved / mass.sum()),
        "moved_diaries": int((survey_labels != survey_original).sum()),
        "population_outside_municipalities": int(mass[membership < 0].sum()),
        "diaries_outside_municipalities": int((survey_membership < 0).sum()),
        "uncovered_area_km2": float(uncovered.area / 1e6),
        "plurality_below_half_units": int(
            (units.population.gt(0) & units.winning_share.lt(0.5)).sum()
        ),
        "ties": int(units.tied_resident_vote.sum()),
        "source_metadata": source["metadata"],
        "assignment_note": "Votes use actual STATPOP homes, permanent main-residence age 6+, without downsampling. Spatial municipality ties use the smallest municipality ID; vote ties use the smallest group number. Empty units use largest original contour area.",
        "coverage_note": "National areas outside municipal coverage retain original contour labels. Every municipal polygon is assigned whole; boundaries are administrative and are not smoothed afterwards.",
        "target_note": "The upstream high-activity target applies before municipal snapping; resulting population shares can change. No trip rates are rescaled.",
        "scope_note": "Municipal integrity does not force a multi-municipality agglomeration to share one group. Observed profiles use the fitting survey; matching is evaluated separately.",
    }
    return {
        "municipalities": units,
        "regions": regions,
        "profiles": profiles,
        "profiles_comparison": comparison,
        "city_comparison": city_comparison,
        "population_flows": population_flows,
        "metadata": metadata,
    }
