"""H3 hexagon helpers for the webmap_export stage (Schema v2).
Cell IDs are BIGINT (H3 never sets the high bit, so signed storage is lossless); geometry stays in LV95.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import h3
import numpy as np
import pyproj

# Resolutions used across the schema: 6 = coarse, 9 = mid, 12 = fine.
H3_RESOLUTIONS = (6, 9, 12)
H3_RES_COARSE, H3_RES_MID, H3_RES_FINE = H3_RESOLUTIONS

_LV95 = "EPSG:2056"
_WGS84 = "EPSG:4326"


@lru_cache(maxsize=2)
def _transformer(from_crs: str, to_crs: str) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)


def lv95_to_wgs84(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised reprojection. Returns (lon, lat) arrays."""
    lon, lat = _transformer(_LV95, _WGS84).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def wgs84_to_lv95(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised reprojection. Returns (x, y) in LV95."""
    x, y = _transformer(_WGS84, _LV95).transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def hex_for_xy_lv95(x: np.ndarray, y: np.ndarray, res: int) -> np.ndarray:
    """Bulk H3 indices (int64) for LV95 coordinate arrays at the given resolution."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    lon, lat = lv95_to_wgs84(x, y)
    out = np.empty(len(lon), dtype="int64")
    for i in range(len(lon)):
        out[i] = h3.str_to_int(h3.latlng_to_cell(float(lat[i]), float(lon[i]), res))
    return out


def hex_parent_int(h3_int_arr: np.ndarray, target_res: int) -> np.ndarray:
    """Bulk parent lookup. Input + output are int64 H3 indices."""
    arr = np.asarray(h3_int_arr, dtype="int64")
    out = np.empty(len(arr), dtype="int64")
    for i in range(len(arr)):
        parent_str = h3.cell_to_parent(h3.int_to_str(int(arr[i])), target_res)
        out[i] = h3.str_to_int(parent_str)
    return out


def hex_geom_wkt_lv95(h3_int: int) -> str:
    """Hex boundary as a closed POLYGON WKT in LV95 (EPSG:2056)."""
    cell_str = h3.int_to_str(int(h3_int))
    boundary_latlon = h3.cell_to_boundary(cell_str)
    lats = np.array([p[0] for p in boundary_latlon], dtype="float64")
    lons = np.array([p[1] for p in boundary_latlon], dtype="float64")
    xs, ys = wgs84_to_lv95(lons, lats)
    coords = [f"{x:.3f} {y:.3f}" for x, y in zip(xs, ys)]
    coords.append(coords[0])
    return f"POLYGON(({', '.join(coords)}))"


def hex_center_wkt_lv95(h3_int: int) -> str:
    """Hex center as a POINT WKT in LV95."""
    cell_str = h3.int_to_str(int(h3_int))
    lat, lon = h3.cell_to_latlng(cell_str)
    xs, ys = wgs84_to_lv95(np.array([lon]), np.array([lat]))
    return f"POINT({xs[0]:.3f} {ys[0]:.3f})"


def hex_geoms_wkt_lv95_bulk(h3_ints: Sequence[int]) -> tuple[list[str], list[str]]:
    """Bulk (geoms, centers) WKT in LV95 for many H3 ids, using batched reprojection."""
    h3_strs = [h3.int_to_str(int(i)) for i in h3_ints]

    boundaries = [h3.cell_to_boundary(s) for s in h3_strs]
    n = len(boundaries)
    bsizes = [len(b) for b in boundaries]
    flat_lat = np.empty(sum(bsizes), dtype="float64")
    flat_lon = np.empty(sum(bsizes), dtype="float64")
    pos = 0
    for b, sz in zip(boundaries, bsizes):
        for j in range(sz):
            flat_lat[pos + j] = b[j][0]
            flat_lon[pos + j] = b[j][1]
        pos += sz
    flat_x, flat_y = wgs84_to_lv95(flat_lon, flat_lat)

    geoms = []
    pos = 0
    for sz in bsizes:
        coords = [f"{flat_x[pos+j]:.3f} {flat_y[pos+j]:.3f}" for j in range(sz)]
        coords.append(coords[0])
        geoms.append(f"POLYGON(({', '.join(coords)}))")
        pos += sz

    centers_latlon = [h3.cell_to_latlng(s) for s in h3_strs]
    lat_arr = np.array([c[0] for c in centers_latlon], dtype="float64")
    lon_arr = np.array([c[1] for c in centers_latlon], dtype="float64")
    cx, cy = wgs84_to_lv95(lon_arr, lat_arr)
    centers = [f"POINT({cx[i]:.3f} {cy[i]:.3f})" for i in range(n)]

    return geoms, centers
