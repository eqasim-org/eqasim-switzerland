"""Population-adaptive cells and survey-weighted adaptive trip-rate surfaces."""

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from ..activity_chain_features import FEATURES


def population_cells(persons, border, minimum=2000, maximum=32000, target=15000):
    for name, value in [("minimum", minimum), ("maximum", maximum), ("target", target)]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    ratio = maximum / minimum
    if ratio < 1 or ratio != 2 ** round(np.log2(ratio)):
        raise ValueError("Maximum/minimum cell ratio must be a power of two")
    bounds = border.bounds
    origin = np.floor(np.array(bounds[:2]) / maximum) * maximum
    n = 2 ** int(np.ceil(np.log2(max(np.array(bounds[2:]) - origin) / minimum)))
    raw = (
        persons.population_type.eq(1)
        & persons.type_of_residence.eq(1)
        & persons.age.ge(6)
    )
    x = persons.home_x.to_numpy(dtype=float)
    y = persons.home_y.to_numpy(dtype=float)
    valid = raw.to_numpy() & np.isfinite(x) & np.isfinite(y)
    valid &= (x >= bounds[0]) & (x <= bounds[2]) & (y >= bounds[1]) & (y <= bounds[3])
    indices = np.flatnonzero(valid)
    indices = indices[shapely.intersects_xy(border, x[indices], y[indices])]
    if not len(indices):
        raise ValueError("No valid geocoded STATPOP residents age 6+")
    ix = np.floor((x[indices] - origin[0]) / minimum).astype(int)
    iy = np.floor((y[indices] - origin[1]) / minimum).astype(int)
    keys = iy * n + ix
    mass = np.bincount(keys, minlength=n * n).reshape(n, n)
    sx = np.bincount(keys, weights=x[indices], minlength=n * n).reshape(n, n)
    sy = np.bincount(keys, weights=y[indices], minlength=n * n).reshape(n, n)
    rows = []

    def split(a, b, size):
        count = int(mass[b : b + size, a : a + size].sum())
        width = size * minimum
        if size > 1 and (width > maximum or count > target):
            half = size // 2
            for dy in [0, half]:
                for dx in [0, half]:
                    split(a + dx, b + dy, half)
            return
        rect = shapely.box(
            origin[0] + a * minimum,
            origin[1] + b * minimum,
            origin[0] + (a + size) * minimum,
            origin[1] + (b + size) * minimum,
        )
        if not rect.intersects(border):
            return
        geometry = rect.intersection(border)
        if geometry.area <= 0:
            return
        centre = geometry.representative_point()
        px = float(sx[b : b + size, a : a + size].sum() / count) if count else centre.x
        py = float(sy[b : b + size, a : a + size].sum() / count) if count else centre.y
        if not border.covers(shapely.Point(px, py)):
            px, py = centre.x, centre.y
        rows.append(
            {
                "cell_id": len(rows),
                "population": count,
                "cell_width_m": width,
                "x": px,
                "y": py,
                "area_km2": geometry.area / 1e6,
                "geometry": geometry,
            }
        )

    split(0, 0, n)
    cells = gpd.GeoDataFrame(rows, crs=2056)
    if int(cells.population.sum()) != len(indices):
        raise ValueError("Adaptive cells lost or duplicated population")
    cells["population_density"] = cells.population / cells.area_km2
    return cells, {
        "population_age_min": 6,
        "population_located": len(indices),
        "population_unlocated": int(raw.sum()) - len(indices),
        "cell_min_m": minimum,
        "cell_max_m": maximum,
    }


def kernel_surfaces(train, queries, neighbours=400, minimum=2000.0, chunk=256):
    """Pool nearby diaries with survey weights and a compact bisquare kernel.

    The bandwidth reaches the requested nearest neighbour, with a 2 km floor.
    If that floor contains additional respondents, all of them contribute.
    Sparse areas therefore borrow evidence across a wider physical radius.
    """
    if (
        isinstance(neighbours, bool)
        or not isinstance(neighbours, (int, np.integer))
        or neighbours < 2
    ):
        raise ValueError(
            "The adaptive neighbourhood must be an integer of at least two"
        )
    if not np.isfinite(minimum) or minimum <= 0 or len(train) < 2:
        raise ValueError("Positive bandwidth and at least two survey diaries required")
    coordinates = train[["home_x", "home_y"]].to_numpy()
    tree = cKDTree(coordinates)
    weights = train.person_weight.to_numpy(float)
    outcomes = train[list(FEATURES)].to_numpy(float)
    count = min(neighbours, len(train))
    rates = np.empty((len(queries), len(FEATURES)))
    support = np.empty(len(queries))
    bandwidth = np.empty(len(queries))
    with threadpool_limits(limits=1):
        for start in range(0, len(queries), chunk):
            query = queries[start : start + chunk]
            distances, indices = tree.query(query, k=count)
            radius = np.maximum(distances[:, -1], minimum) * 1.00000001
            kernel = (
                np.maximum(1 - (distances / radius[:, None]) ** 2, 0) ** 2
                * weights[indices]
            )
            total_weight = kernel.sum(axis=1)
            rates[start : start + len(query)] = (
                np.einsum("ij,ijf->if", kernel, outcomes[indices])
                / total_weight[:, None]
            )
            support[start : start + len(query)] = total_weight**2 / np.square(
                kernel
            ).sum(axis=1)
            bandwidth[start : start + len(query)] = radius

            # Include every diary within the minimum physical radius.
            for row in np.flatnonzero(distances[:, -1] < radius):
                selected = np.array(
                    tree.query_ball_point(query[row], radius[row]), dtype=int
                )
                distance = np.linalg.norm(coordinates[selected] - query[row], axis=1)
                local = (
                    np.maximum(1 - (distance / radius[row]) ** 2, 0) ** 2
                    * weights[selected]
                )
                rates[start + row] = np.average(
                    outcomes[selected], axis=0, weights=local
                )
                support[start + row] = local.sum() ** 2 / np.square(local).sum()
    return rates, support, bandwidth
