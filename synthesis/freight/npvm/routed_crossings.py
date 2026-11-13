"""Choose OD-specific border crossings via sampled foreign road access points."""

import logging
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

logger = logging.getLogger("synpp")


def configure(context):
    context.config("data_path")
    context.config("random_seed")
    context.config("npvm_freight_lw_file", default="npvm/LW_NPVM_2023.mtx")
    context.config("npvm_freight_routed_crossings_file", default=None)
    overwrite = context.config("npvm_freight_routing_overwrite", default=False)
    context.stage("synthesis.freight.npvm.od")
    context.stage("synthesis.freight.npvm.zone_locations")
    if overwrite:
        context.stage("synthesis.freight.npvm.network_access")


def execute(context):

    # Reuse the persistent NPVM assignments unless rerouting was explicitly requested.

    path = saved_crossings_path(context)
    od = context.stage("synthesis.freight.npvm.od")
    locations = context.stage("synthesis.freight.npvm.zone_locations")
    foreign = ~od.origin_zone.isin(locations["points"]) | ~od.destination_zone.isin(locations["points"])
    if not foreign.any():
        return pd.DataFrame(columns=["origin_zone", "destination_zone", "origin_x", "origin_y",
                                     "destination_x", "destination_y", "entry_link_id", "exit_link_id", "probability"])
    if not context.config("npvm_freight_routing_overwrite"):
        if not path.is_file():
            raise FileNotFoundError(f"Saved NPVM crossings not found: {path}. "
                                    "Set npvm_freight_routing_overwrite: true once to create them.")
        result = pd.read_parquet(path)
        # A saved larger sample can serve a smaller one without changing its file.
        columns = ["origin_zone", "destination_zone"]
        if set(columns).issubset(result):
            requested = pd.MultiIndex.from_frame(od[columns].drop_duplicates())
            result = result.loc[pd.MultiIndex.from_frame(result[columns]).isin(requested)].reset_index(drop=True)
        validate_saved_crossings(result, od, locations["points"])
        logger.info("Reused %d saved NPVM crossing candidates from %s; no routing", len(result), path)
        return result

    # Sample foreign road access and route each OD once using the converted network pickle.

    from synthesis.freight.npvm.network_access import sample_access_points
    network = context.stage("synthesis.freight.npvm.network_access")
    foreign = locations["zones"].loc[lambda frame: ~frame.is_swiss]
    access = sample_access_points(foreign, network["roads"],
                                  np.random.default_rng(context.config("random_seed")))
    pairs = od[["origin_zone", "destination_zone"]].drop_duplicates()
    result = route_zone_pairs(pairs, locations["points"], access, network)
    validate_saved_crossings(result, od, locations["points"])

    # Replace the saved assignments only after the complete routing result is valid.

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    logger.info("Saved %d routed NPVM crossing candidates to %s", len(result), path)
    return result


def validate(context):
    path = saved_crossings_path(context)
    stat = path.stat() if path.is_file() else None
    # Explicit overwrite must reroute even if a previous Synpp result exists.
    return [str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None,
            time.time_ns() if context.config("npvm_freight_routing_overwrite") else None]


def saved_crossings_path(context):
    configured = context.config("npvm_freight_routed_crossings_file")
    base = Path(context.config("data_path"))
    return (base / configured if configured else
            (base / context.config("npvm_freight_lw_file")).parent / "freight_routed_crossings.parquet")


def validate_saved_crossings(result, od, domestic):
    required = {"origin_zone", "destination_zone", "origin_x", "origin_y", "destination_x",
                "destination_y", "entry_link_id", "exit_link_id", "probability"}
    if not required.issubset(result):
        raise ValueError(f"Saved NPVM crossings missing columns: {sorted(required - set(result))}")
    foreign = ~od.origin_zone.isin(domestic) | ~od.destination_zone.isin(domestic)
    wanted = set(od.loc[foreign, ["origin_zone", "destination_zone"]].itertuples(index=False, name=None))
    found = set(result[["origin_zone", "destination_zone"]].itertuples(index=False, name=None))
    missing = wanted - found
    if missing:
        raise ValueError(f"Saved NPVM crossings lack {len(missing)} requested OD pairs (e.g. {sorted(missing)[:3]}). "
                         "Set npvm_freight_routing_overwrite: true to rebuild them for this configuration.")
    probabilities = result.probability.to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(result.groupby(["origin_zone", "destination_zone"]).probability.sum(), 1)):
        raise ValueError("Saved NPVM crossing probabilities must be finite, nonnegative and sum to one per OD")
    for end in ("origin", "destination"):
        coordinates = result.loc[~result[end + "_zone"].isin(domestic), [end + "_x", end + "_y"]]
        if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
            raise ValueError(f"Saved NPVM crossings have invalid foreign {end} coordinates")


def route_zone_pairs(pairs, domestic, access, network):
    graph, nodes, links = network["graph"], network["nodes"], network["links"]
    tree = cKDTree(nodes[["x", "y"]].to_numpy())
    access_by_zone = {int(zone): group.to_dict("records") for zone, group in access.groupby("zone_id")}
    events = {str(link): group.sort_values("crossing_index").to_dict("records")
              for link, group in network["crossings"].groupby("link_id")}
    inside_indices = np.flatnonzero(network["inside"])
    inside_tree = cKDTree(nodes.iloc[inside_indices][["x", "y"]].to_numpy())
    swiss_nodes, physical_inside = {}, {}
    for zone in set(pairs.origin_zone) | set(pairs.destination_zone):
        if zone in domestic:
            points = np.asarray(domestic[zone])
            representative = points[np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1))]
            physical_inside[zone] = bool(network["country"].covers(Point(representative)))
            swiss_nodes[zone] = (int(inside_indices[inside_tree.query(representative)[1]]) if physical_inside[zone]
                                 else int(tree.query(representative)[1]))
    edge_links = np.asarray(graph.es["link_index"], dtype=int)
    edge_times = np.asarray(graph.es["time"], dtype=float)
    routing_weights = edge_times.tolist()
    transit = TransitRouter(network)
    link_ids = links.link_id.astype(str).to_numpy()
    path_cache = {}
    # Batch shortest-path searches by source node across all repeated OD requests.
    requests = {False: {}, True: {}}
    for pair in pairs.itertuples(index=False):
        if pair.origin_zone in access_by_zone and pair.destination_zone in access_by_zone:
            continue
        if pair.origin_zone not in access_by_zone and pair.destination_zone not in access_by_zone:
            continue
        starts = [int(a["to_index"]) for a in access_by_zone[pair.origin_zone]] if pair.origin_zone in access_by_zone else [swiss_nodes[pair.origin_zone]]
        targets = [int(a["from_index"]) for a in access_by_zone[pair.destination_zone]] if pair.destination_zone in access_by_zone else [swiss_nodes[pair.destination_zone]]
        reverse = pair.destination_zone in access_by_zone
        for start in starts:
            for target in targets:
                root, goal = (target, start) if reverse else (start, target)
                requests[reverse].setdefault(root, set()).add(goal)
    reverse_graph = graph.copy()
    reverse_graph.reverse_edges()
    for reverse, grouped in requests.items():
        search = reverse_graph if reverse else graph
        for i, (start, targets) in enumerate(grouped.items()):
            targets = sorted(targets)
            paths = search.get_shortest_paths(start, to=targets, weights=routing_weights, output="epath")
            for target, path in zip(targets, paths):
                path_cache[(target, start) if reverse else (start, target)] = path[::-1] if reverse else path
            if i % 25 == 0:
                logger.info("Freight routing trees (%s): %d/%d foreign access nodes",
                            "out" if reverse else "in", i + 1, len(grouped))
    del reverse_graph
    records = []
    for i, pair in enumerate(pairs.itertuples(index=False)):
        origin_foreign = pair.origin_zone in access_by_zone
        destination_foreign = pair.destination_zone in access_by_zone
        if not origin_foreign and not destination_foreign:
            continue
        origins = access_by_zone[pair.origin_zone] if origin_foreign else [None]
        destinations = access_by_zone[pair.destination_zone] if destination_foreign else [None]
        candidates = []
        for origin in origins:
            start = int(origin["to_index"]) if origin_foreign else swiss_nodes[pair.origin_zone]
            targets = [int(d["from_index"]) if destination_foreign else swiss_nodes[pair.destination_zone]
                       for d in destinations]
            paths = ([None] * len(targets) if origin_foreign and destination_foreign else
                     [path_cache[start, t] for t in targets])
            for destination, target, path in zip(destinations, targets, paths):
                if not (origin_foreign and destination_foreign) and not path and start != target:
                    continue
                if origin_foreign and destination_foreign:
                    route = transit.route(origin, destination)
                    if route is None:
                        continue
                    indices, route_time = route
                else:
                    indices, route_time = edge_links[path], float(edge_times[path].sum())
                crossings = [event for link in link_ids[indices] for event in events.get(link, [])]
                entering = next((event for event in crossings if event["direction"] == "in"), None)
                leaving = next((event for event in reversed(crossings) if event["direction"] == "out"), None)
                outside_model_domestic = ((not origin_foreign and not physical_inside[pair.origin_zone]) or
                                          (not destination_foreign and not physical_inside[pair.destination_zone]))
                if ((origin_foreign and entering is None) or (destination_foreign and leaving is None)) and not outside_model_domestic:
                    continue
                travel_time = route_time
                if origin_foreign:
                    travel_time += (1 - origin["fraction"]) * origin["link_time"]
                if destination_foreign:
                    travel_time += destination["fraction"] * destination["link_time"]
                weight = (origin["weight"] if origin_foreign else 1) * (destination["weight"] if destination_foreign else 1)
                candidates.append({"origin_zone": pair.origin_zone, "destination_zone": pair.destination_zone,
                    "origin_x": (entering["crossing_x"] if entering else origin["x"]) if origin_foreign else np.nan,
                    "origin_y": (entering["crossing_y"] if entering else origin["y"]) if origin_foreign else np.nan,
                    "destination_x": (leaving["crossing_x"] if leaving else destination["x"]) if destination_foreign else np.nan,
                    "destination_y": (leaving["crossing_y"] if leaving else destination["y"]) if destination_foreign else np.nan,
                    "entry_link_id": entering["link_id"] if origin_foreign and entering else None,
                    "exit_link_id": leaving["link_id"] if destination_foreign and leaving else None,
                    "uses_swiss_crossing": bool(entering or leaving),
                    "outside_model_domestic": outside_model_domestic,
                    "origin_access_link_id": origin["link_id"] if origin_foreign else None,
                    "destination_access_link_id": destination["link_id"] if destination_foreign else None,
                    "origin_access_x": origin["x"] if origin_foreign else np.nan,
                    "origin_access_y": origin["y"] if origin_foreign else np.nan,
                    "destination_access_x": destination["x"] if destination_foreign else np.nan,
                    "destination_access_y": destination["y"] if destination_foreign else np.nan,
                    "travel_time_seconds": travel_time, "weight": weight / (1 + travel_time)})
        if not candidates:
            raise ValueError(f"No Swiss-crossing truck route for NPVM OD {pair.origin_zone} → {pair.destination_zone}; no demand was dropped")
        total = sum(row["weight"] for row in candidates)
        for row in candidates:
            row["probability"] = row.pop("weight") / total
        records.extend(candidates)
        if i % 100 == 0:
            logger.info("Freight border routing: processed %d/%d distinct zone pairs", i + 1, len(pairs))
    return pd.DataFrame(records)


class TransitRouter:
    """Join exterior approaches with a positive-length route wholly inside Switzerland."""
    def __init__(self, network):
        self.network = network
        self.cache = {}
        self.graphs = {}
        for name in ("inside_graph", "outside_graph"):
            graph = network[name]
            weights = graph.es["time"]
            self.graphs[name] = (graph, weights, np.asarray(graph.es["link_index"], dtype=int), np.asarray(weights))
        self.link_lookup = {str(link): i for i, link in enumerate(network["links"].link_id)}
        self.outside_components = np.asarray(network["outside_graph"].connected_components(mode="weak").membership)
        links = network["links"].set_index("link_id")
        c = network["crossings"].copy()
        c["from_index"] = c.link_id.map(links.from_index)
        c["to_index"] = c.link_id.map(links.to_index)
        inside = network["inside"]
        self.entries = c[c.direction.eq("in") & ~inside[c.from_index] & inside[c.to_index]]
        self.exits = c[c.direction.eq("out") & inside[c.from_index] & ~inside[c.to_index]]

    def path(self, name, source, target):
        key = name, int(source), int(target)
        if key not in self.cache:
            graph, weights, indices, times = self.graphs[name]
            # Disconnected candidate approaches are expected and receive no probability.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Couldn't reach some vertices.*", category=RuntimeWarning)
                path = graph.get_shortest_paths(int(source), to=int(target), weights=weights, output="epath")[0]
            self.cache[key] = (indices[path], float(times[path].sum())) if path or source == target else None
        return self.cache[key]

    @staticmethod
    def nearest(frame, access):
        distance = np.hypot(frame.border_x - access["x"], frame.border_y - access["y"])
        return (frame.assign(distance=distance).sort_values(["distance", "link_id"], kind="stable")
                .drop_duplicates(["border_x", "border_y"]).head(3))

    def route(self, origin, destination):
        best = None
        entries = self.entries[self.outside_components[self.entries.from_index] == self.outside_components[int(origin["to_index"])]]
        exits = self.exits[self.outside_components[self.exits.to_index] == self.outside_components[int(destination["from_index"])]]
        for entry in self.nearest(entries, origin).itertuples():
            approach = self.path("outside_graph", origin["to_index"], entry.from_index)
            if approach is None:
                continue
            for exit in self.nearest(exits, destination).itertuples():
                if (round(entry.border_x, 3), round(entry.border_y, 3)) == (round(exit.border_x, 3), round(exit.border_y, 3)):
                    continue
                middle = self.path("inside_graph", entry.to_index, exit.from_index)
                onward = self.path("outside_graph", exit.to_index, destination["from_index"])
                if middle is None or not len(middle[0]) or onward is None:
                    continue
                links = self.network["links"]
                entry_index = self.link_lookup[str(entry.link_id)]
                exit_index = self.link_lookup[str(exit.link_id)]
                cost = approach[1] + middle[1] + onward[1] + links.time.iloc[entry_index] + links.time.iloc[exit_index]
                if best is None or cost < best[1]:
                    best = np.r_[approach[0], entry_index, middle[0], exit_index, onward[0]], float(cost)
        return best
