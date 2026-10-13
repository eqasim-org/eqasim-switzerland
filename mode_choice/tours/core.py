"""Convenience interface for standalone tour sampling (process-local cache)."""
from mode_choice.selection.sampler import TourSampler

_DEFAULT_SAMPLER = TourSampler()


def sample_tour(utilities, origins, destinations, rng):
    return _DEFAULT_SAMPLER.sample(utilities, list(origins), list(destinations), rng)
