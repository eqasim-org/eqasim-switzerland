"""Realize native macroscopic route distributions without another route search."""

import math

from sumo.scenario.network.tools import position, read_network, validate_routes
from sumo.scenario.tools import elements


def materialize_assignment(assigned, weights, trips, network, output, seed, begin=0, interval=3600, context=None, kernel=None):
    """Realize shares on disk and stream validated, departure-sorted vehicles."""
    import logging
    from pathlib import Path
    import tempfile
    import time
    from sumo.scenario.assignment_spool import AssignmentSpool
    from sumo.scenario.population.writer import PopulationWriter

    logger = logging.getLogger("synpp")
    started = time.monotonic()
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix="realization_", dir=output.parent) as temporary:
        spool = AssignmentSpool(Path(temporary) / "assignment.sqlite")
        try:
            # Keep source lookup, distribution groups and sorting on disk.

            logger.info("SUMO realization: indexing source trip metadata in bounded batches")
            types = spool.read_sources(trips)
            logger.info("SUMO realization: reading assigned route distributions")
            spool.read_assignments(assigned, begin, interval)
            distributions = spool.allocate(seed)
            logger.info("SUMO realization: reading network for route validation and clocks")
            data = read_network(network, context=context)
            state = {}
            intervals = iter(_weight_intervals(weights))
            current = None
            completed = 0
            target = Path(temporary) / output.name

            # One loaded cost interval and one vehicle are live during final export.

            with PopulationWriter(target) as writer:
                for definition in types.values(): writer.element(definition)
                for attributes, edges in spool.selected():
                    depart = float(attributes["depart"])
                    while current is None or depart >= current[1]:
                        current = None  # Release the preceding cost dictionary before parsing the next interval.
                        current = next(intervals, None)
                        if current is None:
                            raise ValueError(f"Loaded edge costs do not cover departure {depart}")
                        logger.info("SUMO realization: loaded costs [%.0f, %.0f); %d/%d vehicles written",
                                    current[0], current[1], completed, spool.count)
                    if depart < current[0]:
                        raise ValueError(f"Loaded edge costs do not cover departure {depart}")
                    vehicle = {"attributes": attributes, "edges": edges}
                    try:
                        validate_routes([vehicle], data, types, state=state)
                    except ValueError as error:
                        raise ValueError(f"Invalid native marouter assignment: {error}. Check class permissions and turn "
                                         "connectivity in the assignment network. Assigned paths are never silently replaced.") from error
                    exits, route_attrs = _edge_clocks(attributes, edges, data, current, kernel)
                    writer.vehicle(attributes, edges, exits,
                                   {"sumo_timing": "macroscopic_departure_slice"}, route_attrs)
                    completed += 1
                    if completed % 10000 == 0:
                        elapsed = time.monotonic()-started
                        logger.info("SUMO realization: timed/written %d/%d vehicles (%.1f%%); %.1fs elapsed",
                                    completed, spool.count, 100*completed/spool.count, elapsed)
            target.replace(output)
            logger.info("SUMO realization complete: %d vehicles, %d distributions; %.1fs", completed, distributions, time.monotonic()-started)
            return {"vehicles": completed, "distributions": distributions,
                    "route_choice": "seeded balanced integer realization of assigned distributions",
                    "timing": "loaded edge costs from each vehicle's departure interval",
                    "post_assignment_routing": False}
        finally:
            spool.close()


def _edge_clocks(attributes, edges, data, current, kernel):
    begin, end, costs = current
    depart = float(attributes["depart"])
    lengths = [data["edges"][edge]["length"] for edge in edges]
    start = position(attributes.get("departPos", "0"), lengths[0])
    stop = position(attributes.get("arrivalPos", "max"), lengths[-1])
    missing = next((edge for edge in edges if edge not in costs), None)
    if missing is not None:
        raise ValueError(f"No loaded travel time for assigned edge {missing} in [{begin}, {end})")
    if kernel is not None:
        exits, distance = kernel.clocks(lengths, [costs[edge] for edge in edges], depart, start, stop)
        return exits, {"routeLength": format(distance, ".12g")}
    clock, exits = depart, []
    for index, (edge_id, length) in enumerate(zip(edges, lengths)):
        low = start if index == 0 else 0
        high = stop if index == len(edges)-1 else length
        if high < low:
            raise ValueError(f"Arrival precedes departure position on route {attributes['id']}")
        clock += costs[edge_id] * (high-low) / length
        exits.append(clock)
    return exits, {}


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
    import logging
    logger = logging.getLogger("synpp")
    costs = {}
    for element in elements(path, ("edge", "interval")):
        if element.tag == "edge":
            value = float(element.get("traveltime"))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid loaded travel time for edge {element.get('id')}")
            costs[element.get("id")] = value
            if len(costs) % 250000 == 0:
                logger.info("SUMO realization: reading loaded costs: %d edges in current interval", len(costs))
            continue
        begin, end = float(element.get("begin")), float(element.get("end"))
        if not math.isfinite(begin) or not math.isfinite(end) or not previous_end <= begin < end:
            raise ValueError("Loaded edge-cost intervals must be finite, ordered and nonoverlapping")
        yield begin, end, costs
        costs = {}
        previous_end = end
