"""Compatibility facade for the central matching preprocessing workflow."""
from ..workflow import preprocess


def prepare_inputs(context, const):
    prepared = preprocess(context, "v2", const, include_regions=False)
    return prepared.source, prepared.population, prepared.trips
