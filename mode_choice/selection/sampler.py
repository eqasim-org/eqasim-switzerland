"""Exact whole-tour logit choice in bounded-length computational segments.

Forward messages carry weights for BOTH vehicle locations across split points.
Backward draws are conditioned on returning both vehicles to the tour origin.
A split therefore neither strands a vehicle nor creates one at a work activity.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from mode_choice.defaults import MODES, SAMPLING_METHOD, SAMPLING_METHODS, MAX_TRIPS_PER_SEGMENT, RANDOM_CACHE_MAX_VALUES
from mode_choice.selection.cache import TopologyCache
from mode_choice.selection.methods import select
from mode_choice.selection.random import RandomBudget
from mode_choice.tours.split import split_tour


def canonical_locations(origins, destinations):
    """Same visit pattern gets the same cache key, regardless of coordinates."""
    origins, destinations = np.asarray(origins), np.asarray(destinations)
    if origins.shape != destinations.shape or origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError("Expected matching origin/destination coordinate pairs")
    if not np.isfinite(origins).all() or not np.isfinite(destinations).all():
        raise ValueError("Tour coordinates must be finite")
    if len(origins) > 1 and not np.array_equal(origins[1:], destinations[:-1]):
        raise ValueError("A tour contains disconnected activity locations")
    identifiers = {}
    def encode(point):
        key = tuple(point)
        return identifiers.setdefault(key, len(identifiers))
    origin_ids, destination_ids = [], []
    for origin, destination in zip(origins, destinations):
        origin_ids.append(encode(origin))
        destination_ids.append(encode(destination))
    return tuple(origin_ids), tuple(destination_ids)


def forward(graph, values, initial_weights, keep_layers=False):
    weights = initial_weights
    history = []
    for row, layer in zip(values, graph.layers):
        if keep_layers:
            history.append(weights)
        edge_weights = weights[layer.sources] + row[layer.modes]
        weights = np.logaddexp.reduceat(edge_weights, layer.offsets[:-1])
    return weights, history


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    start: int
    end: int
    key: tuple


@dataclass(frozen=True, slots=True)
class TourPlan:
    segments: tuple
    final_target: int
    num_trips: int
    gumbel_values: int


class TourSampler:
    def __init__(self, method=SAMPLING_METHOD, max_trips=MAX_TRIPS_PER_SEGMENT, cache=None,
                 random_cache_max_values=RANDOM_CACHE_MAX_VALUES):
        if method not in SAMPLING_METHODS:
            raise ValueError(f"Sampling method must be one of {SAMPLING_METHODS}")
        split_tour(0, max_trips=max_trips)
        self.method, self.max_trips = method, max_trips
        self.cache = cache if cache is not None else TopologyCache()
        self.random_budget = RandomBudget(random_cache_max_values)
        self._plans = OrderedDict()

    def prepare(self, origins, destinations, purposes=None):
        """Validate geometry and retain utility-independent segment/group indices."""
        origins, destinations = canonical_locations(origins, destinations)
        segments = split_tour(len(origins), purposes, self.max_trips)
        key = (origins, destinations, segments)
        if key in self._plans:
            self._plans.move_to_end(key)
            return self._plans[key]
        initial_state = (origins[0], origins[0])
        states, plans, noise_size = (initial_state,), [], 0
        for start, end in segments:
            graph_key = (states, origins[start:end], destinations[start:end])
            graph = self.cache.get(*graph_key)
            plans.append(SegmentPlan(start, end, graph_key))
            # Upper bound for any backward path, independent of utilities and
            # availability. One bulk stream preserves legacy sequential draws.
            noise_size += sum(int(np.diff(layer.offsets).max()) for layer in graph.layers)
            states = graph.final_states
        plan = TourPlan(tuple(plans), states.index(initial_state), len(origins), noise_size)
        if self.cache.max_entries and self.cache.max_edges:
            if len(self._plans) >= self.cache.max_entries:
                self._plans.popitem(last=False)
            self._plans[key] = plan
        return plan

    def sample(self, values, origins, destinations, rng, purposes=None):
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return []
        if values.ndim != 2 or values.shape[1] != len(MODES) or np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("Expected finite mode utilities or -inf for unavailable modes")
        plan = self.prepare(origins, destinations, purposes)
        if plan.num_trips != len(values):
            raise ValueError("Utility and location counts differ")
        return [MODES[index] for index in self.sample_prepared(values, plan, rng)]

    def sample_prepared(self, values, plan, rng, out=None):
        """Draw using prevalidated utilities and retained tour structure.

        Only utility-dependent forward weights and backward choices change.
        Prepared plans hold graph keys, not graph arrays, so graph-cache eviction
        continues to release transition arrays even while tours are retained.
        """
        weights = np.array([0.0])
        boundaries = []
        last_history = None
        for segment in plan.segments:
            start, end = segment.start, segment.end
            graph = self.cache.get(*segment.key)
            boundaries.append(weights)
            weights, history = forward(graph, values[start:end], weights, keep_layers=end == plan.num_trips)
            if end == plan.num_trips:
                last_history = history
        if not np.isfinite(weights[plan.final_target]):
            raise ValueError("Tour has no feasible mode sequence with vehicles returned to the start")
        last_graph = graph
        target = plan.final_target
        choices = np.empty(plan.num_trips, dtype=np.int8) if out is None else out
        for segment, weights in zip(reversed(plan.segments), reversed(boundaries)):
            start, end = segment.start, segment.end
            if end == plan.num_trips:
                graph, history = last_graph, last_history
            else:
                graph = self.cache.get(*segment.key)
                _, history = forward(graph, values[start:end], weights, keep_layers=True)
            for offset in range(end - start - 1, -1, -1):
                layer = graph.layers[offset]
                left, right = layer.offsets[target:target + 2]
                sources, modes = layer.sources[left:right], layer.modes[left:right]
                scores = history[offset][sources] + values[start + offset, modes]
                selected = select(scores, rng, self.method)
                choices[start + offset] = modes[selected]
                target = int(sources[selected])
        return choices
