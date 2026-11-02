"""Native macroscopic assignment and opt-in simulation-based DUA loading."""

from pathlib import Path
import math
import statistics

from sumo.config import macroscopic, tools
from sumo.scenario import modes
from sumo.config.tools import positive, time_window
from sumo.runtime import sumo
from sumo.scenario.tools import elements
from sumo.scenario.assignment import materialize_assignment


def run_macroscopic(context, network, trips, method, realize=True):
    cache = Path(context.path()).resolve()
    if positive(context.config("sumo_assignment_scale"), "sumo_assignment_scale") < 1:
        raise ValueError("Loading requires scale >= 1 so every original trip retains a route")
    # Read departure bounds once; mapped endpoints go directly into assignment.

    departures = [float(trip.get("depart")) for trip in elements(trips, "trip")]
    if not departures:
        raise ValueError("No trips available for macroscopic assignment")
    demand = {"network": network, "loading": str(trips), "queries": str(trips), "count": len(departures),
              "first_departure": min(departures), "last_departure": max(departures)}

    # Validate the time window and run class- and position-aware marouter.

    assignment = macroscopic.write(context, demand, method=method)
    sumo.run(context, ["-c", assignment], binary="marouter",
             executable=context.stage("sumo.native.marouter"))
    routes, weights = cache / "assigned.routes.xml.gz", cache / "loaded.edges.xml.gz"
    if not realize:
        return cache / "assigned.flows.xml.gz", weights, None
    report = materialize_assignment(cache / "assigned.flows.xml.gz", weights, trips, network,
                                    routes, context.config("random_seed"), context.config("sumo_begin"),
                                    context.config("sumo_assignment_interval"))
    return routes, weights, report


def run_iterations(context, network, trips):
    begin, end = time_window(context)
    departures = [float(trip.get("depart")) for trip in elements(trips, "trip")]
    if min(departures) < begin or (end is not None and max(departures) >= end):
        raise ValueError("sumo_begin/sumo_end must include every selected trip departure")
    iterations = positive(context.config("sumo_assignment_iterations"), "sumo_assignment_iterations", integer=True)
    window = positive(context.config("sumo_loading_convergence_window"), "sumo_loading_convergence_window", integer=True)
    if window < 2:
        raise ValueError("sumo_loading_convergence_window must be at least two")
    tolerance = float(context.config("sumo_loading_convergence_tolerance"))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("sumo_loading_convergence_tolerance must be finite and nonnegative")
    scale = positive(context.config("sumo_assignment_scale"), "sumo_assignment_scale")
    if scale < 1:
        raise ValueError("Loading requires scale >= 1 so every original trip retains a route")
    settings = context.stage("sumo.runtime.sumo")
    choice = context.config("sumo_loading_route_choice")
    if choice not in {"msa", "gawron", "logit"}:
        raise ValueError("sumo_loading_route_choice must be 'msa', 'gawron', or 'logit'")
    arguments = ["--net-file", network, "--trips", trips, "--path", str(Path(settings["home"]) / "bin"),
                 "--begin", begin,
                 "--last-step", iterations, "--aggregation", context.config("sumo_assignment_interval"),
                 "--vehroute-file", "detailed", "--output-lastRoute", "--time-to-teleport=-1",
                 "--convergence-iterations", window, "--max-convergence-deviation", tolerance]
    if end is not None:
        arguments += ["--end", end]
    if choice in {"msa", "logit"}:
        arguments += ["--logit"]
    if choice == "msa":
        arguments += ["--method-of-successive-average"]
    if context.config("sumo_loading_mesoscopic"):
        arguments += ["--mesosim", "--meso-junctioncontrol", "--meso-multiqueue"]
    arguments += ["sumo--seed", context.config("random_seed"), "duarouter--seed", context.config("random_seed"),
                  "sumo--scale", scale, "sumo--vehroute-output.intended-depart", "false",
                  "sumo--vehroute-output.write-unfinished", "true", "sumo--tripinfo-output.write-unfinished", "true",
                  "sumo--tripinfo-output.write-undeparted", "true"]
    if context.config("sumo_loading_mesoscopic"):
        arguments += ["sumo--meso-ltm", "true", "sumo--meso-lane-queue", "true"]
    sumo.run_python(context, "assign/duaIterate.py", arguments)
    cache = Path(context.path())
    steps = sorted((p for p in cache.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    means = []
    for step in steps:
        durations = []
        for trip in elements(step / f"tripinfo_{int(step.name):03d}.xml", "tripinfo"):
            if float(trip.get("arrival", "-1")) < 0:
                raise ValueError(f"Iteration {step.name} has unfinished/undeparted vehicles; extend sumo_end")
            durations.append(float(trip.get("duration")))
        if not durations:
            raise ValueError(f"No completed trips in iteration {step.name}")
        means.append(statistics.fmean(durations))
    recent = means[-window:]
    deviation = statistics.stdev(recent) / statistics.fmean(recent) if len(recent) > 1 else None
    stable = tolerance > 0 and len(recent) >= window and deviation is not None and deviation <= tolerance
    last = steps[-1]
    routes = last / f"vehroute_{int(last.name):03d}.xml"
    weights = list(last.glob("dump_*.xml.gz"))
    if not routes.is_file() or len(weights) != 1:
        raise RuntimeError("Native DUA did not produce the expected final route and edge-cost files")
    return routes, weights[0], {"method": "simulation-based DUA", "route_choice": choice,
        "iterations": len(means), "mean_travel_times_s": means, "relative_stddev": deviation,
        "stable": stable, "stability_tolerance": tolerance, "equilibrium_guaranteed": False}


def configure_loading(context):
    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("sumo.runtime.sumo")
    context.stage("sumo.scenario.demand")
    context.stage("sumo.scenario.network.routing")
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo"):
        context.stage("sumo.scenario.border")
    tools.configure_network(context)
    context.config("sumo_region_geometry", None)
    if context.config("sumo_region_geometry") is not None:
        context.stage("sumo.cutter.split_network")
    method = context.config("sumo_loading_method", "incremental")
    if method not in {"dua", "incremental", "SUE"}:
        raise ValueError("sumo_loading_method must be 'dua', 'incremental' or 'SUE'")
    if method != "dua":
        context.stage("sumo.native.marouter")
        macroscopic.configure_profile(context)
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_end", None)
    context.config("sumo_assignment_iterations", 20)
    context.config("sumo_assignment_interval", 3600)
    context.config("sumo_assignment_scale", 1.0)
    context.config("sumo_mapmatch_distance", None)
    context.config("sumo_loading_mesoscopic", True)
    context.config("sumo_loading_route_choice", "msa")
    context.config("sumo_loading_convergence_window", 5)
    context.config("sumo_loading_convergence_tolerance", 0.01)
    context.config("sumo_loading_require_convergence", False)
