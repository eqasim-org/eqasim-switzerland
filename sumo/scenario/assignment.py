"""Realize native macroscopic route distributions without another route search."""

from collections import defaultdict
import copy
import math
import random

from sumo.scenario.network.tools import position, read_network, validate_routes
from sumo.scenario.tools import elements, write_routes


def materialize_assignment(assigned, weights, trips, network, output, seed, begin=0, interval=3600):
    """Keep assigned paths, balance integer route shares, and estimate edge clocks.

    marouter's probabilities can be flow counts rather than normalized fractions.
    Integer realization differs by less than one vehicle per route/distribution.
    Travel times use the loaded cost slice containing the requested departure:
    macroscopic assignment does not simulate a vehicle's progress between slices.
    """
    definitions, types = {}, {}
    for element in elements(trips, ("vType", "trip")):
        if element.tag == "vType":
            types[element.get("id")] = copy.deepcopy(element)
        else:
            definitions[element.get("id")] = dict(element.attrib)
    groups = defaultdict(list)
    for element in elements(assigned, "vehicle"):
        attributes = dict(element.attrib)
        source_id = source_trip_id(attributes["id"], definitions)
        # Geographic and edge endpoints belong to trips, not explicit vehicles.
        source = {key: value for key, value in definitions[source_id].items()
                  if key not in {"from", "to", "via", "fromLonLat", "toLonLat", "viaLonLat",
                                 "fromXY", "toXY", "fromJunction", "toJunction", "fromTaz", "toTaz"}}
        attributes = {**source, **attributes}
        alternatives = element.findall("routeDistribution/route") or element.findall("route")
        choices = []
        for route in alternatives:
            probability = float(route.get("probability", "1"))
            edges = tuple(route.get("edges", "").split())
            if not math.isfinite(probability) or probability < 0 or not edges:
                raise ValueError(f"Invalid assigned route distribution for {attributes['id']}")
            if probability:
                choices.append((edges, probability))
        if not choices:
            raise ValueError(f"No positive assigned route probability for {attributes['id']}")
        period = math.floor((float(attributes["depart"]) - float(begin)) / float(interval))
        groups[(period, attributes.get("type"), tuple(choices))].append(attributes)

    # Realize every distribution, including native scale clones, with a seeded shuffle.

    rng = random.Random(seed)
    vehicles = []
    for (_, _, choices), members in groups.items():
        total = math.fsum(probability for _, probability in choices)
        targets = [len(members) * probability / total for _, probability in choices]
        counts = [math.floor(target) for target in targets]
        ranking = sorted(range(len(choices)), key=lambda i: (targets[i] - counts[i], rng.random()), reverse=True)
        for index in ranking[:len(members) - sum(counts)]:
            counts[index] += 1
        rng.shuffle(members)
        offset = 0
        for (edges, _), count in zip(choices, counts):
            for attributes in members[offset:offset + count]:
                vehicles.append({"attributes": attributes, "edges": list(edges), "exit_times": [],
                                 "route_attributes": {}, "parameters": {}})
            offset += count
    if not vehicles:
        raise ValueError("Macroscopic assignment produced no individual vehicles")
    data = read_network(network)
    try:
        validate_routes(vehicles, data, types)
    except ValueError as error:
        raise ValueError(f"Invalid native marouter assignment: {error}. SUMO marouter may ignore vehicle-class "
                         "restrictions; use sumo_loading_method: dua for class-aware assignment. "
                         "Assigned paths are never silently replaced.") from error

    # Accumulate loaded costs on the selected paths, correcting partial endpoint edges.

    intervals = iter(_weight_intervals(weights))
    current = None
    for vehicle in sorted(vehicles, key=lambda row: float(row["attributes"]["depart"])):
        attributes, edges = vehicle["attributes"], vehicle["edges"]
        depart = float(attributes["depart"])
        while current is None or depart >= current[1]:
            current = next(intervals, None)
            if current is None:
                raise ValueError(f"Loaded edge costs do not cover departure {depart}")
        begin, end, costs = current
        if depart < begin:
            raise ValueError(f"Loaded edge costs do not cover departure {depart}")
        clock = depart
        for index, edge_id in enumerate(edges):
            length = data["edges"][edge_id]["length"]
            start = position(attributes.get("departPos", "0"), length) if index == 0 else 0
            stop = position(attributes.get("arrivalPos", "max"), length) if index == len(edges) - 1 else length
            if stop < start:
                raise ValueError(f"Arrival precedes departure position on route {attributes['id']}")
            if edge_id not in costs:
                raise ValueError(f"No loaded travel time for assigned edge {edge_id} in [{begin}, {end})")
            clock += costs[edge_id] * (stop - start) / length
            vehicle["exit_times"].append(clock)
        vehicle["parameters"]["sumo_timing"] = "macroscopic_departure_slice"
    write_routes(output, types, vehicles)
    return {"vehicles": len(vehicles), "distributions": len(groups),
            "route_choice": "seeded balanced integer realization of assigned distributions",
            "timing": "loaded edge costs from each vehicle's departure interval",
            "post_assignment_routing": False}


def source_trip_id(vehicle_id, originals):
    """Match native scale clones while allowing original IDs containing dots."""
    if vehicle_id in originals:
        return vehicle_id
    parent, separator, clone = vehicle_id.rpartition(".")
    if separator and clone.isdigit() and parent in originals:
        return parent
    raise ValueError(f"Cannot match loaded vehicle {vehicle_id} to its source trip")


def _weight_intervals(path):
    previous_end = -math.inf
    for interval in elements(path, "interval"):
        begin, end = float(interval.get("begin")), float(interval.get("end"))
        if not math.isfinite(begin) or not math.isfinite(end) or not previous_end <= begin < end:
            raise ValueError("Loaded edge-cost intervals must be finite, ordered and nonoverlapping")
        costs = {}
        for edge in interval.findall("edge"):
            value = float(edge.get("traveltime"))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid loaded travel time for edge {edge.get('id')}")
            costs[edge.get("id")] = value
        yield begin, end, costs
        previous_end = end
