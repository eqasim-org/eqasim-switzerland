"""Company counts at pre-matching home and usual-work anchors.

Use the same STATENT index and radius as destination preparation. Counts refer
to establishments within the radius, not employees or companies per square km.
"""

import numpy as np
import pandas as pd

from data.statent.density import impute
from dmc.constants import constants
from .features import numeric


def add_company_densities(context, source, population, work_locations, const):
    """Return independent frames enriched on both sides of diary matching.

    Query unique coordinates across both datasets and anchors just once. No
    diary destination is used: survey work addresses and assigned population
    work anchors are available before matching. Unknown coordinates stay NaN;
    known locations with no nearby establishments receive a genuine zero.
    """
    source, population = source.copy(), population.copy()
    work = work_locations.set_index("person_id")
    if not work.index.is_unique:
        raise ValueError("Company density requires one work anchor per person")
    if "geometry" in work:
        geometry = work.geometry
        if getattr(work, "crs", None) is not None:
            geometry = work.to_crs(const.LV95).geometry
        population["work_x"] = population.person_id.map(geometry.x)
        population["work_y"] = population.person_id.map(geometry.y)
    elif {"x", "y"}.issubset(work.columns):
        population["work_x"] = population.person_id.map(work.x)
        population["work_y"] = population.person_id.map(work.y)

    # Retain valid existing anchor-specific counts and collect missing queries.

    queries, destinations = [], []
    for frame, is_source in ((source, True), (population, False)):
        employed = numeric(frame, "employed")
        if is_source:
            employed = employed.eq(1)
        elif const.census == "are_synpop":
            employed = employed.isin([1, 3])
        else:
            employed = employed.eq(const.EMPLOYED)
        for anchor in ("home", "work"):
            column = f"{anchor}_companies_density"
            counts = numeric(frame, column).to_numpy(copy=True)
            counts[counts < 0] = np.nan
            coordinates = np.column_stack([numeric(frame, anchor + "_x"),
                                           numeric(frame, anchor + "_y")])
            applicable = np.ones(len(frame), dtype=bool) if anchor == "home" else employed.to_numpy()
            counts[~applicable] = np.nan
            valid = np.isfinite(coordinates).all(axis=1) & (coordinates > 0).all(axis=1)
            missing = np.flatnonzero(applicable & valid & ~np.isfinite(counts))
            queries.append(coordinates[missing])
            destinations.append((frame, column, counts, missing))

    # Reuse the pipeline's established radius and chunked STATENT count query.

    coordinates = np.concatenate(queries)
    if len(coordinates):
        unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
        points = pd.DataFrame(unique, columns=["x", "y"])
        density = impute(context, points, radius=constants.EMPLOYEES_DENSITY_RADIUS,
                         measure="companies", point_type="V2 home/work anchor")
        queried = density.companies_density.to_numpy()[inverse]
    else:
        queried = np.empty(0)

    # Restore original row order and keep missing-value indicators for the model.

    offset = 0
    for frame, column, counts, missing in destinations:
        counts[missing] = queried[offset:offset + len(missing)]
        offset += len(missing)
        frame[column] = counts
    return source, population
