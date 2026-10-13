"""Immutable batch inputs and reusable grouping for repeated tour choice."""
import numpy as np
import polars as pl

from mode_choice.defaults import MODES, CALIBRATION_MIN_DISTANCE_KM
from mode_choice.selection.random import RandomTape, generate
from mode_choice.tours.build import assign_tours


class PreparedBatch:
    """Prepare grouping, geometry, splits, count masks and random slices once.

    The retained arrays contain no sampled choices or utility-dependent weights.
    Every selection applies new alpha differences to the original base utilities.
    Use as a context manager to release the reserved random-buffer budget.
    """

    def __init__(self, trips, seed, sampler, *, assume_sorted=False):
        if "tour_index" not in trips.columns:
            trips = assign_tours(trips)
        elif not assume_sorted:
            trips = trips.sort("person_id", "trip_index")
        self.sampler = sampler
        self.keys = trips.select("person_id", "trip_index")
        self.base = trips.select([f"utility_{mode}" for mode in MODES]).to_numpy(order="c")
        if np.isnan(self.base).any() or np.isposinf(self.base).any():
            raise ValueError("Expected finite mode utilities or -inf for unavailable modes")
        self.base.flags.writeable = False
        persons, tours = trips["person_id"].to_numpy(), trips["tour_index"].to_numpy()
        self.boundaries = (np.r_[0, np.flatnonzero((persons[1:] != persons[:-1]) | (tours[1:] != tours[:-1])) + 1, trips.height]
                           if trips.height else np.array([0]))
        origins = trips.select("origin_x", "origin_y").to_numpy()
        destinations = trips.select("destination_x", "destination_y").to_numpy()
        purposes = trips["preceding_purpose"].to_numpy()
        self.plans, self.random_keys = [], []
        for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
            key = (seed, int(persons[start]), int(tours[start]))
            try:
                self.plans.append(sampler.prepare(origins[start:end], destinations[start:end], purposes[start:end]))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Mode choice failed for person {key[1]}, tour {key[2]}: {exc}") from exc
            self.random_keys.append(key)
        self.eligible = None
        if "euclidean_distance_km" in trips.columns:
            distances = trips["euclidean_distance_km"].to_numpy()
            self.eligible = np.flatnonzero(np.isfinite(distances) & (distances > CALIBRATION_MIN_DISTANCE_KM))
        sizes = np.array([plan.gumbel_values if sampler.method == "gumbel" else plan.num_trips
                          for plan in self.plans], dtype=np.int64)
        self.noise_offsets = np.r_[0, sizes.cumsum()]
        self.cached_tours, self.reserved_values = sampler.random_budget.reserve(self.noise_offsets[1:])
        self.closed = False
        try:
            self.noise = np.empty(self.reserved_values)
            for i in range(self.cached_tours):
                start, end = self.noise_offsets[i:i+2]
                self.noise[start:end] = generate(self.random_keys[i], sampler.method, int(end-start))
            self.noise.flags.writeable = False
        except BaseException:
            sampler.random_budget.release(self.reserved_values)
            raise

    def sample_codes(self, delta=None, *, draw_index=0):
        """Return int8 mode codes; retain grouping across all calibration passes."""
        if self.closed:
            raise RuntimeError("Prepared batch is closed")
        if isinstance(draw_index, bool) or not isinstance(draw_index, (int, np.integer)) or draw_index < 0:
            raise ValueError("draw_index must be a nonnegative integer")
        if delta is None:
            values = self.base
        else:
            delta = np.asarray(delta, dtype=float)
            if delta.shape != (len(MODES),) or not np.isfinite(delta).all():
                raise ValueError("Expected one finite alpha adjustment per mode")
            values = self.base + delta
        result = np.empty(self.keys.height, dtype=np.int8)
        for i, (start, end) in enumerate(zip(self.boundaries[:-1], self.boundaries[1:])):
            left, right = self.noise_offsets[i:i+2]
            noise = (self.noise[left:right] if i < self.cached_tours and draw_index == 0 else
                     generate(self.random_keys[i], self.sampler.method, int(right-left), draw_index))
            try:
                self.sampler.sample_prepared(values[start:end], self.plans[i], RandomTape(noise), out=result[start:end])
            except (ValueError, TypeError) as exc:
                key = self.random_keys[i]
                raise ValueError(f"Mode choice failed for person {key[1]}, tour {key[2]}: {exc}") from exc
        return result

    def counts(self, codes):
        if self.eligible is None:
            raise ValueError("Trip distances are required for calibration shares")
        return np.bincount(codes[self.eligible], minlength=len(MODES))

    def materialize(self, codes):
        return self.keys.with_columns(pl.Series("mode", np.asarray(MODES)[codes]))

    def close(self):
        if not self.closed:
            self.sampler.random_budget.release(self.reserved_values)
            self.noise = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
