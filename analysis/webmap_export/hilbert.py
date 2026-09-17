"""2D Hilbert curve index for spatial-locality bulk loads.
Order = 16 -> 2^16 cells per axis, plenty for the CH bbox.
"""

from __future__ import annotations

import numpy as np

ORDER = 16
SIDE = 1 << ORDER


def _xy_to_d(n: int, x: int, y: int) -> int:
    """Reference scalar implementation (Wikipedia: Hilbert curve / d2xy)."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def hilbert_2d(xs: np.ndarray, ys: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Vectorised 2D Hilbert index (uint64) for EPSG:2056 coords; bbox
    (xmin, ymin, xmax, ymax) normalises coordinates into [0, SIDE)."""
    xmin, ymin, xmax, ymax = bbox
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    nx = np.clip(((xs - xmin) / (xmax - xmin) * SIDE).astype(np.int64), 0, SIDE - 1)
    ny = np.clip(((ys - ymin) / (ymax - ymin) * SIDE).astype(np.int64), 0, SIDE - 1)

    d = np.zeros_like(nx, dtype=np.uint64)
    x = nx.copy()
    y = ny.copy()
    s = SIDE >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += np.uint64(s * s) * ((3 * rx) ^ ry).astype(np.uint64)

        flip = ry == 0
        flip_rx1 = flip & (rx == 1)
        x = np.where(flip_rx1, s - 1 - x, x)
        y = np.where(flip_rx1, s - 1 - y, y)
        x_new = np.where(flip, y, x)
        y_new = np.where(flip, x, y)
        x, y = x_new, y_new
        s >>= 1
    return d


CH_BBOX_LV95 = (2_400_000.0, 1_050_000.0, 2_900_000.0, 1_300_000.0)
