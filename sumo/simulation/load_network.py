"""Load selected demand with iterative simulation or native macroscopic assignment."""

import json
import hashlib
import logging
from pathlib import Path
import shutil

from sumo.config import tools
from sumo.router.tools import write_trip_metrics

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("sumo.simulation.realize_routes")
    context.config("sumo_loading_method", "incremental")
    context.config("sumo_loading_require_convergence", False)


def execute(context):

    # Publish independently cached prepared, assigned and realized demand.

    cache = Path(context.path()).resolve()
    realized = context.stage("sumo.simulation.realize_routes")
    network, loading = realized["network"], realized["trips"]
    count, endpoint_mapping = realized["source_count"], realized["endpoint_mapping"]
    raw_routes, weights = realized["routes"], realized["weights"]
    method = realized["method"]
    convergence, timing = realized["convergence"], realized["timing"]

    # Preserve source IDs and check completion and turn connectivity before publishing routes.

    from sumo.simulation.publication import publish
    output, vehicle_count = publish(context, loading, raw_routes, network, method)
    metrics = cache / "trip_metrics.arrow"
    write_trip_metrics(output, metrics, vehicle_count)
    final_weights = cache / "loaded.edges.xml.gz"
    if Path(weights).resolve() != final_weights:
        shutil.copyfile(weights, final_weights)

    # Return the loaded full scenario and an explicit stability report for the cutter.

    result = {"network": str(network), "routes": str(output), "weights": str(final_weights), "metrics": str(metrics),
              "count": vehicle_count, "source_count": count, "timing": timing,
              "boundary_split": realized["boundary_split"], "convergence": convergence,
              "endpoint_mapping": endpoint_mapping}
    (cache / "loading_report.json").write_text(json.dumps(result, indent=2) + "\n")
    if convergence.get("stable") is False:
        message = "Loading reached its iteration limit before the requested travel-time stability threshold"
        if context.config("sumo_loading_require_convergence"):
            raise RuntimeError(message)
        logger.warning(message)
    return result


def validate(context):
    options = []
    fingerprint = hashlib.sha256()
    from sumo.simulation import publication
    for path in (publication.__file__,):
        fingerprint.update(Path(path).read_bytes())
    return {"external_inputs": tools.external_token(context, options), "helpers": fingerprint.hexdigest()}


def _copy(element):
    import copy
    return copy.deepcopy(element)
