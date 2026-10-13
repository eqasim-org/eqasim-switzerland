"""Balanced computational tour segments; vehicle state is not reset at a split."""
import numpy as np
from mode_choice.defaults import MAX_TRIPS_PER_SEGMENT, WORK_PURPOSES


def split_tour(num_trips, purposes=None, max_trips=MAX_TRIPS_PER_SEGMENT):
    if isinstance(max_trips, bool) or not isinstance(max_trips, (int, np.integer)) or max_trips < 1:
        raise ValueError("mode_choice_max_trips_per_tour must be a positive integer")
    if purposes is not None and len(purposes) != num_trips:
        raise ValueError("Expected one origin activity purpose per trip")
    if num_trips == 0:
        return ()
    pending, result = [(0, num_trips)], []
    while pending:
        start, end = pending.pop()
        if end - start <= max_trips:
            result.append((start, end))
            continue
        midpoint = (start + end) / 2
        # Prefer a work anchor in the central half, avoiding tiny fragments near
        # an endpoint. Otherwise split at the middle activity, never within a trip.
        margin = max(1, (end - start) // 4)
        candidates = [i for i in range(start + margin, end - margin + 1)
                      if purposes is not None and purposes[i] in WORK_PURPOSES]
        cut = min(candidates, key=lambda i: (abs(i - midpoint), i)) if candidates else (start + end) // 2
        pending.extend(((cut, end), (start, cut)))
    return tuple(result)
