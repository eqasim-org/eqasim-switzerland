"""Load selected demand with iterative simulation or native macroscopic assignment."""

import json
import logging
from pathlib import Path
import shutil

from sumo.config import macroscopic, tools
from sumo.scenario import modes
from sumo.scenario.network.tools import read_network, route_length, validate_routes
from sumo.scenario.tools import elements, filter_loading_trips, read_routes, write_routes
from sumo.runtime import sumo
from sumo.router.tools import write_trip_metrics
from sumo.simulation.loading import run_iterations, run_macroscopic

logger = logging.getLogger("synpp")


def configure(context):
    modes.configure(context)
    context.stage("sumo.runtime.sumo")
    context.stage("sumo.scenario.demand")
    tools.configure_network(context)
    context.config("sumo_region_geometry", None)
    if context.config("sumo_region_geometry") is not None:
        context.stage("sumo.cutter.split_network")
    method = context.config("sumo_loading_method", "dua")
    if method not in {"dua", "incremental", "SUE"}:
        raise ValueError("sumo_loading_method must be 'dua', 'incremental' or 'SUE'")
    if method != "dua":
        macroscopic.configure_profile(context)
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_end", 108000)
    context.config("sumo_assignment_iterations", 50)
    context.config("sumo_assignment_interval", 3600)
    context.config("sumo_assignment_scale", 1.0)
    context.config("sumo_mapmatch_distance", 1000)
    context.config("sumo_loading_mesoscopic", True)
    context.config("sumo_loading_route_choice", "msa")
    context.config("sumo_loading_convergence_window", 5)
    context.config("sumo_loading_convergence_tolerance", 0.01)
    context.config("sumo_loading_require_convergence", False)


def execute(context):

    # Resolve the original or boundary-split full network and selected scenario demand.

    cache = Path(context.path()).resolve()
    source = context.stage("sumo.scenario.demand")
    network = tools.network_path(context)
    split = None
    if context.config("sumo_region_geometry") is not None:
        split = context.stage("sumo.cutter.split_network")
        network = split["network"]
    loading = cache / "loading.trips.xml.gz"
    count = filter_loading_trips(source["queries"], loading)
    if not count:
        raise ValueError("No selected car/truck/LCV or other road vehicle demand to load")
    if split is not None:
        from sumo.cutter.tools import remap_trip_endpoints
        remap_trip_endpoints(loading, split["mapping"], cache / "split.trips.xml.gz")
        loading = cache / "split.trips.xml.gz"

    # Alternate route choice and traffic loading, or explicitly use a macroscopic method.

    method = context.config("sumo_loading_method")
    if method == "dua":
        raw_routes, weights, convergence = run_iterations(context, network, loading)
        timing = "experienced edge exit times"
    else:
        raw_routes, weights = run_macroscopic(context, network, loading, method)
        convergence = {"method": method, "equilibrium_guaranteed": False}
        timing = "macroscopic estimated edge exit times"

    # Preserve source IDs and check completion and turn connectivity before publishing routes.

    definitions = {element.get("id"): dict(element.attrib) for element in elements(loading, "trip")}
    parameters = {element.get("id"): {p.get("key"): p.get("value") for p in element.findall("param")}
                  for element in elements(loading, "trip")}
    types = {element.get("id"): _copy(element) for element in elements(loading, "vType")}
    routed_types, vehicles = read_routes(raw_routes)
    types.update(routed_types)
    seen = set()
    data = read_network(network)
    for vehicle in vehicles:
        source_id = _source_id(vehicle["attributes"]["id"], definitions)
        seen.add(source_id)
        vehicle["parameters"] = {**parameters[source_id], **vehicle["parameters"], "sumo_original_trip_id": source_id,
                                 "sumo_requested_depart": definitions[source_id]["depart"]}
        vehicle["route_attributes"]["cost"] = format(vehicle["exit_times"][-1] - float(vehicle["attributes"]["depart"]), ".12g")
        distance = (float(vehicle["attributes"]["routeLength"]) if method == "dua" and "routeLength" in vehicle["attributes"]
                    else route_length(vehicle["edges"], data, vehicle["attributes"]))
        vehicle["route_attributes"]["routeLength"] = format(distance, ".12g")
    if seen != set(definitions):
        raise ValueError("Loading did not complete every source trip; extend sumo_end or repair network deadlocks")
    validate_routes(vehicles, data, types)
    output = cache / "loaded.routes.xml.gz"
    write_routes(output, types, vehicles)
    metrics = cache / "trip_metrics.parquet"
    write_trip_metrics(output, metrics, len(vehicles))
    final_weights = cache / "loaded.edges.xml.gz"
    if Path(weights).resolve() != final_weights:
        shutil.copyfile(weights, final_weights)

    # Return the loaded full scenario and an explicit stability report for the cutter.

    result = {"network": str(network), "routes": str(output), "weights": str(final_weights), "metrics": str(metrics),
              "count": len(vehicles), "source_count": count, "timing": timing,
              "boundary_split": split is not None, "convergence": convergence}
    (cache / "loading_report.json").write_text(json.dumps(result, indent=2) + "\n")
    if convergence.get("stable") is False:
        message = "Loading reached its iteration limit before the requested travel-time stability threshold"
        if context.config("sumo_loading_require_convergence"):
            raise RuntimeError(message)
        logger.warning(message)
    return result


def validate(context):
    options = []
    if context.config("sumo_loading_method") != "dua":
        options.append("sumo_assignment_previous_weights")
    return tools.external_token(context, options)


def _copy(element):
    import copy
    return copy.deepcopy(element)


def _source_id(vehicle_id, originals):
    if vehicle_id in originals:
        return vehicle_id
    parent, separator, clone = vehicle_id.rpartition(".")
    if separator and clone.isdigit() and parent in originals:
        return parent
    raise ValueError(f"Cannot match loaded vehicle {vehicle_id} to its source trip")
