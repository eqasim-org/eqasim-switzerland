"""One distance policy for diary targets and survey distance comparisons."""
import numpy as np
import pandas as pd

DETOUR_FACTOR = 1.3


def distance_observations(trips):
    """Keep endpoint displacement separate from a flagged travelled-distance proxy."""
    coordinates = trips[['origin_x', 'origin_y', 'destination_x', 'destination_y']].to_numpy(float)
    finite = np.isfinite(coordinates).all(axis=1)
    displacement = np.hypot(coordinates[:, 2]-coordinates[:, 0], coordinates[:, 3]-coordinates[:, 1])
    valid = finite & (displacement > 0)
    loop = finite & (displacement == 0) & trips['mode'].astype(str).str.endswith('_loop').to_numpy()
    reported = pd.to_numeric(trips['network_distance'], errors='coerce').to_numpy(float)
    proxy_valid = np.isfinite(reported) & (reported > 0)
    reference = np.where(valid, displacement, np.where(loop, 0., np.where(proxy_valid, reported/DETOUR_FACTOR, np.nan)))
    source = np.where(valid, 'coordinates', np.where(loop, 'loop', np.where(proxy_valid, 'reported_div_1.3', 'unavailable')))
    return pd.DataFrame(dict(coordinate_distance_m=displacement, reported_distance_m=reported,
        reference_distance_m=reference, distance_source=source, coordinate_valid=valid, loop_at_origin=loop), index=trips.index)


def spatial_training_trips(trips):
    """Never train or validate destination coordinates on coincident/missing endpoints."""
    coordinates = trips[['origin_x', 'origin_y', 'destination_x', 'destination_y']].to_numpy(float)
    valid = np.isfinite(coordinates).all(axis=1) & (np.hypot(coordinates[:, 2]-coordinates[:, 0], coordinates[:, 3]-coordinates[:, 1]) > 0)
    return trips.loc[valid].copy()
