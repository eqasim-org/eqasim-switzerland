"""Four whole-municipality activity zones from prepared Microcensus diaries.

This is the public Synpp stage for matching. Adaptive kernels estimate work,
shopping, leisure and total daily trip rates; continuous four-feature contours
are then assigned to entire municipalities by resident plurality.
"""

import hashlib
from pathlib import Path

import numpy as np
import shapely

from .activity_chain_features import (
    FEATURES,
    daily_trip_counts,
    day_mask,
    valid_diary_mask,
)
from .activity_zones.surfaces import population_cells, kernel_surfaces
from .activity_zones.contours import smooth_contours
from .activity_zones.municipalities import municipalize_contours


def configure(context):
    for stage in (
        "data.microcensus.persons",
        "data.microcensus.trips",
        "data.microcensus.activity_chains",
        "data.statpop.persons",
        "data.spatial.municipalities",
        "data.spatial.swiss_border",
    ):
        context.stage(stage)
    context.config("random_seed")
    context.config("specific_day_scenario", "workday")
    context.config("activity_zones_neighbours", 400)
    context.config("activity_zones_min_bandwidth_m", 2000.0)
    context.config("activity_zones_min_cell_m", 2000)
    context.config("activity_zones_max_cell_m", 32000)
    context.config("activity_zones_target_cell_population", 15000)
    context.config("activity_zones_geometry_step_m", 1000.0)
    context.config("activity_zones_gaussian_smoothing_m", 0.0)
    context.config("activity_zones_high_population_share", 0.16)


def execute(context):

    # Read prepared diaries and full resident homes; no raw survey files are read.

    border = context.stage("data.spatial.swiss_border").to_crs(2056).union_all()
    population = context.stage("data.statpop.persons")
    municipalities = context.stage("data.spatial.municipalities")[0]
    diaries = prepare_diaries(
        context.stage("data.microcensus.persons"),
        context.stage("data.microcensus.activity_chains"),
        context.stage("data.microcensus.trips")[0],
        border,
        context.config("specific_day_scenario"),
    )

    # Adapt spatial support to population, then pool nearby weighted survey diaries.

    cells, audit = population_cells(
        population,
        border,
        minimum=context.config("activity_zones_min_cell_m"),
        maximum=context.config("activity_zones_max_cell_m"),
        target=context.config("activity_zones_target_cell_population"),
    )
    neighbours = context.config("activity_zones_neighbours")
    minimum = context.config("activity_zones_min_bandwidth_m")
    rates, support, bandwidth = kernel_surfaces(
        diaries,
        cells[["x", "y"]].to_numpy(),
        neighbours,
        minimum,
    )
    cells["effective_sample"] = support
    cells["bandwidth_m"] = bandwidth
    for j, feature in enumerate(FEATURES):
        cells[feature] = rates[:, j]
        cells[feature + "_density"] = cells[feature] * cells.population / cells.area_km2

    # Reserve the high-activity tail and cluster the other three four-feature regions.

    contours = smooth_contours(
        cells,
        population,
        diaries,
        border,
        step=context.config("activity_zones_geometry_step_m"),
        sigma=context.config("activity_zones_gaussian_smoothing_m"),
        high_share=context.config("activity_zones_high_population_share"),
        seed=context.config("random_seed"),
    )
    contours["metadata"]["source_surface_metadata"] = {
        **audit,
        "seed": int(context.config("random_seed")),
        "day": context.config("specific_day_scenario"),
        "fitting_persons": len(diaries),
        "selected_neighbours": int(neighbours),
        "minimum_bandwidth_m": float(minimum),
        "target_cell_population": context.config(
            "activity_zones_target_cell_population"
        ),
        "effective_sample_min": float(support.min()),
        "bandwidth_median_m": float(np.median(bandwidth)),
        "bandwidth_max_m": float(bandwidth.max()),
        "selection_note": "Fixed matching-oriented settings; no bandwidth or segmentation search is run here.",
    }

    # Keep each municipality whole using actual resident counts in the contours.

    result = municipalize_contours(
        contours, population, municipalities, border, diaries
    )
    result["cells"] = cells
    result["contours"] = contours

    # Return assignments and diagnostics; the analysis stage owns report exports.

    return result


def prepare_diaries(persons, chains, trips, border, day):
    """Build the fitting cohort from prepared stages, retaining genuine zero days."""
    if persons.person_id.isna().any() or persons.person_id.duplicated().any():
        raise ValueError("Unique, non-missing survey person IDs required")
    valid = valid_diary_mask(chains)
    roster = persons[
        ["person_id", "person_weight", "home_x", "home_y", "workday", "weekend", "day"]
    ].copy()
    roster = roster.merge(
        chains[["person_id", "activity_chain"]],
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    if roster.activity_chain.isna().any():
        raise ValueError("Prepared survey persons are missing activity chains")
    finite = np.isfinite(roster[["home_x", "home_y"]]).all(axis=1)
    inside = np.zeros(len(roster), dtype=bool)
    inside[finite] = shapely.intersects_xy(
        border, roster.loc[finite, "home_x"], roster.loc[finite, "home_y"]
    )
    usable = roster.person_id.map(valid).fillna(False)
    roster = (
        roster.loc[usable & finite & inside & day_mask(roster, day)]
        .sort_values("person_id")
        .reset_index(drop=True)
    )
    if (~np.isfinite(roster.person_weight) | roster.person_weight.le(0)).any():
        raise ValueError("Survey weights must be finite and positive")
    counts = daily_trip_counts(trips).reindex(roster.person_id, fill_value=0)
    roster[list(FEATURES)] = counts.to_numpy()
    if not np.array_equal(roster.activity_chain.str.count("-"), roster.total_trips):
        raise ValueError("Prepared diary and trip counts disagree")
    return roster


def validate(context):
    """Include imported helpers in Synpp's cache identity."""
    directory = Path(__file__).parent
    paths = [
        directory / "activity_chain_features.py",
        *sorted((directory / "activity_zones").glob("*.py")),
    ]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
