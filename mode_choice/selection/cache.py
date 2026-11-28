"""Bounded, process-local LRU cache for immutable vehicle-state graphs."""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from mode_choice.defaults import CACHE_MAX_ENTRIES
from mode_choice.defaults import CACHE_MAX_EDGES
from mode_choice.defaults import MODES
from mode_choice.defaults import VEHICLE_MODES


def nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class Layer:
    # Edges grouped by destination state; offsets index each incoming group.
    sources: np.ndarray
    modes: np.ndarray
    offsets: np.ndarray
    history: np.ndarray | None = None


@dataclass(frozen=True)
class Graph:
    layers: tuple
    final_states: tuple
    edge_count: int


def build_graph(initial_states, origins, destinations):
    """Compile vehicle/history transitions; availability/utilities are never cached."""
    states, layers, edge_count = initial_states, [], 0
    with_history = len(initial_states[0]) == 4
    passenger_mode = MODES.index('car_passenger')
    for origin, destination in zip(origins, destinations):
        incoming = {}
        for source, state in enumerate(states):
            for mode in range(len(MODES)):
                vehicle = VEHICLE_MODES.get(mode)
                target = state
                if vehicle is not None:
                    if state[vehicle] != origin:
                        continue
                    target = list(state)
                    target[vehicle] = destination
                    target = tuple(target)
                if with_history:
                    passenger = mode == passenger_mode
                    target = (*target[:2], passenger, state[3] or passenger)
                    edge = (source, mode, passenger and state[2], passenger and state[3])
                else:
                    edge = (source, mode)
                incoming.setdefault(target, []).append(edge)
        states = tuple(incoming)
        groups = list(incoming.values())
        edges = np.asarray([edge for group in groups for edge in group], dtype=np.int32)
        sources, modes = edges[:, 0].copy(), edges[:, 1].astype(np.int8)
        offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
        for array in (sources, modes, offsets):
            array.flags.writeable = False
        history = edges[:, 2:].astype(np.float64) if edges.shape[1] == 4 else None
        if history is not None:
            history.flags.writeable = False
        layers.append(Layer(sources, modes, offsets, history))
        edge_count += len(edges)
    return Graph(tuple(layers), states, edge_count)


class TopologyCache:
    """Evict by both graph count and edge count; zero disables retention."""
    def __init__(self, max_entries=CACHE_MAX_ENTRIES, max_edges=CACHE_MAX_EDGES):
        self.max_entries = nonnegative_integer(max_entries, "mode_choice_cache_max_entries")
        self.max_edges = nonnegative_integer(max_edges, "mode_choice_cache_max_edges")
        self._graphs = OrderedDict()
        self.hits = self.misses = self.edges = 0

    def get(self, initial_states, origins, destinations):
        key = (initial_states, origins, destinations)
        if key in self._graphs:
            self.hits += 1
            self._graphs.move_to_end(key)
            return self._graphs[key]
        self.misses += 1
        graph = build_graph(*key)
        if self.max_entries and graph.edge_count <= self.max_edges:
            while self._graphs and (len(self._graphs) >= self.max_entries or self.edges + graph.edge_count > self.max_edges):
                _, evicted = self._graphs.popitem(last=False)
                self.edges -= evicted.edge_count
            self._graphs[key] = graph
            self.edges += graph.edge_count
        return graph

    def info(self):
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._graphs), "edges": self.edges}
