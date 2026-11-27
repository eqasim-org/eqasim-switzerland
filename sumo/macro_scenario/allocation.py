"""Allocate route shares reproducibly and cache compact native processing inputs."""

import hashlib
import json
import tempfile
from pathlib import Path

from sumo.functions.config.tools import positive
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.native import processing
from sumo.functions.scenario import assignment_spool, route_io


def configure(context):
    context.stage("sumo.macro_scenario.indexed")
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_assignment_interval", 3600)
    processing.configure_options(context)


@monitored
def execute(context):

    # Keep lookup tables and integer allocation in a temporary, bounded disk spool.

    indexed = context.stage("sumo.macro_scenario.indexed")
    cache = Path(context.path()).resolve()
    with tempfile.TemporaryDirectory(prefix="allocation_", dir=cache) as temporary:
        spool = assignment_spool.AssignmentSpool(Path(temporary) / "allocation.sqlite")
        try:
            spool.read_sources_arrow(indexed["sources"], indexed["types"])
            spool.read_assignments_arrow(
                indexed["assignments"],
                context.config("sumo_begin"),
                context.config("sumo_assignment_interval"),
            )
            groups = spool.allocate(context.config("random_seed"))

            # Write departure-ordered selections and shared paths with row/byte limits.

            selected, paths = cache / "selected.arrow", cache / "groups.arrow"
            spool.write_selection(
                Path(temporary) / selected.name,
                Path(temporary) / paths.name,
                positive(
                    context.config("sumo_route_batch_size"),
                    "sumo_route_batch_size",
                    integer=True,
                ),
                positive(
                    context.config("sumo_route_batch_bytes"),
                    "sumo_route_batch_bytes",
                    integer=True,
                ),
            )
            (Path(temporary) / selected.name).replace(selected)
            (Path(temporary) / paths.name).replace(paths)
            result = {
                "selected": str(selected),
                "groups": str(paths),
                "types": indexed["types"],
                "vehicles": spool.count,
                "distributions": groups,
            }
        finally:
            spool.close()

    # Retain a small manifest for independent realization reruns.

    (cache / "allocation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate(context):
    return [
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (assignment_spool, route_io)
    ]
