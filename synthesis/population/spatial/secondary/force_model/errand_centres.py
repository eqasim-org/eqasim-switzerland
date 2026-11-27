"""Smooth, purpose-specific attraction around configured centres, in LV95 metres."""
import numpy as np

PREFIX = "secondary_activities.force_model."
CENTRE_KEYS = {"other_services": "errand_centres", "shop": "shop_centres", "leisure": "leisure_centres"}
OPTION = PREFIX + CENTRE_KEYS["other_services"]


def validate_centres(centres, option=OPTION):
    keys = {"x", "y", "radius_m", "strength_m"}
    if not isinstance(centres, (list, tuple)):
        raise ValueError(f"{option} must be a list of centre specifications")
    for centre in centres:
        if not isinstance(centre, dict) or set(centre) != keys:
            raise ValueError(f"Each centre in {option} needs {sorted(keys)}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
               for v in centre.values()):
            raise ValueError(f"{option} values must be finite numbers")
        if centre["radius_m"] <= 0 or centre["strength_m"] < 0:
            raise ValueError(f"{option} requires a positive radius and nonnegative strength")
    return centres


def adjust_field(field, grid, centres, ceiling):
    """Add radial forces to a normalized purpose grid, preserving the solver's cap."""
    if not centres:
        return field
    result = field.copy()
    cell = grid["cell_size"]
    x = grid["xmin"] + cell * np.arange(grid["nx"])
    y = grid["ymin"] + cell * np.arange(grid["ny"])
    touched = np.zeros(field.shape[:2], dtype=bool)
    for centre in centres:
        radius = centre["radius_m"]
        dx = centre["x"] - x
        dy = centre["y"] - y
        ix = np.flatnonzero(np.abs(dx) < radius)
        iy = np.flatnonzero(np.abs(dy) < radius)
        area = np.ix_(ix, iy)
        fraction = np.hypot(dx[ix, None], dy[None, iy]) / radius
        # Peak at one third of the radius, with zero force at centre and boundary.
        scale = 6.75 * centre["strength_m"] / radius * np.maximum(1. - fraction, 0.) ** 2
        result[:, :, 0][area] += dx[ix, None] * scale
        result[:, :, 1][area] += dy[None, iy] * scale
        touched[area] |= scale > 0
    vectors = result[touched]
    magnitude = np.linalg.norm(vectors, axis=1)
    above = magnitude > ceiling
    vectors[above] *= (ceiling / magnitude[above])[:, None]
    result[touched] = vectors
    return result
