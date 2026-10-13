"""Person-count batches preserve full days and parking-duration dependencies."""
import numpy as np


def person_batches(trips, batch_size):
    """Yield zero-copy slices of sorted trips, never splitting a person's day."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("mode_choice_person_batch_size must be a positive integer")
    if trips.is_empty():
        return
    persons = trips["person_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(persons[1:] != persons[:-1]) + 1]
    for index in range(0, len(starts), batch_size):
        start = int(starts[index])
        end = int(starts[index + batch_size]) if index + batch_size < len(starts) else trips.height
        yield trips.slice(start, end - start)

