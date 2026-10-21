import numpy as np

# mode: (detour factor, slow speed [km/h], fast speed [km/h], half distance [km], access and egress time [min])
PARAMETERS = {
    "car":           (1.3, 40.0, 90.0, 20.0, 5.0),
    "car_passenger": (1.3, 40.0, 90.0, 20.0, 5.0),
    "pt":            (1.3, 25.0, 80.0, 30.0, 10.0),
}

DEFAULT_MODE = "car"


def estimate_travel_time(distance, mode):
    """ Travel times [s] for arrays of beeline distances [m] and modes. """

    distance = np.nan_to_num(np.asarray(distance, dtype = float)) / 1000.0
    mode = np.broadcast_to(np.asarray(mode, dtype = object), distance.shape)
    mode = np.where(np.isin(mode, list(PARAMETERS)), mode, DEFAULT_MODE)

    travel_time = np.zeros(distance.shape)

    for name, (detour, slow, fast, half_distance, access) in PARAMETERS.items():
        selection = mode == name
        d = distance[selection]
        speed = slow + (fast - slow) * d / (d + half_distance)
        travel_time[selection] = (detour * d / speed + access / 60.0) * 3600.0

    return np.round(travel_time)
