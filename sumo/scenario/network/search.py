"""Finite-network spatial search without an arbitrary distance cutoff."""

import math

import numpy as np
from shapely.geometry import box

SEARCH_STEP = 500.


def distance_limit(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("sumo_mapmatch_distance must be positive or null (unlimited)")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("sumo_mapmatch_distance must be positive or null (unlimited)")
    return value


def candidate_windows(tree, point, bounds, best_score, maximum=None, seen=()):
    """Search 500, 1000, ... metres, evaluating each edge at most once.

    A nonnegative distance-plus-heading score safely narrows each spatial query.
    Stop when no unseen edge can beat the best match. With no valid match, stop
    after covering the finite network, rather than expanding forever.
    """
    if len(tree.geometries) == 0:
        return
    x, y = point.x, point.y
    if not np.isfinite([x, y]).all():
        raise ValueError("Endpoint coordinates must be finite")
    xmin, ymin, xmax, ymax = bounds
    extent = math.hypot(max(abs(x-xmin), abs(x-xmax)), max(abs(y-ymin), abs(y-ymax))) + 20.
    end = maximum if maximum is not None else max(SEARCH_STEP, math.ceil(extent/SEARCH_STEP)*SEARCH_STEP)
    radius = min(SEARCH_STEP, end)
    visited = set(seen)
    while True:
        score = best_score()
        query_radius = radius if score is None else min(radius, max(0., score) + 1e-6)
        candidates = tree.query(box(x-query_radius, y-query_radius, x+query_radius, y+query_radius))
        fresh = [int(index) for index in candidates if index not in visited]
        visited.update(fresh)
        if fresh:
            yield fresh
        score = best_score()
        if radius >= end or (score is not None and score + 1e-6 < radius):
            return
        radius = min(end, radius + SEARCH_STEP)
