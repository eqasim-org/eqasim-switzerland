"""Bulk per-tour random streams with bounded, in-memory calibration retention."""
import numpy as np

from mode_choice.defaults import RANDOM_CACHE_MAX_VALUES
from mode_choice.selection.cache import nonnegative_integer


class RandomBudget:
    def __init__(self, max_values=RANDOM_CACHE_MAX_VALUES):
        self.max_values = nonnegative_integer(max_values, "mode_choice_random_cache_max_values")
        self.retained = self.peak = 0

    def reserve(self, cumulative_sizes):
        count = int(np.searchsorted(cumulative_sizes, self.max_values - self.retained, side="right"))
        values = int(cumulative_sizes[count - 1]) if count else 0
        self.retained += values
        self.peak = max(self.peak, self.retained)
        return count, values

    def release(self, values):
        self.retained -= values


def generate(key, method, size, draw_index=0):
    # Index zero exactly reproduces the existing person/tour SeedSequence.
    # A separate draw index gives an independent, reproducible replicate.
    key = key if draw_index == 0 else (*key, draw_index)
    rng = np.random.default_rng(np.random.SeedSequence(key))
    return rng.gumbel(size=size) if method == "gumbel" else rng.random(size)


class RandomTape:
    """A cursor over a tour's own draws; slices never wrap or overlap tours."""
    __slots__ = ("values", "position")

    def __init__(self, values):
        self.values, self.position = values, 0

    def gumbel(self, size):
        start = self.position
        self.position += size
        if self.position > len(self.values):
            raise RuntimeError("Prepared Gumbel buffer exhausted")
        return self.values[start:self.position]

    def random(self):
        if self.position >= len(self.values):
            raise RuntimeError("Prepared uniform buffer exhausted")
        value = self.values[self.position]
        self.position += 1
        return value
