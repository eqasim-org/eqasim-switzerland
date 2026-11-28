"""Parse macroscopic XML outputs into bounded IPC batches with native SAX."""

import tempfile
from pathlib import Path

from scripts.monitoring import monitored
from sumo.functions.native import processing


def configure(context):
    context.stage("sumo.macro_scenario.assignment")
    processing.configure_stage(context, "index")


@monitored
def execute(context):

    # Read assignment inputs without modifying their caches.

    assigned = context.stage("sumo.macro_scenario.assignment")
    if assigned["method"] == "dua":
        raise ValueError(
            "Native assignment indexing applies to incremental and SUE loading"
        )
    cache = Path(context.path()).resolve()
    paths = {
        name: str(cache / filename)
        for name, filename in [
            ("sources", "sources.arrow"),
            ("assignments", "assignments.arrow"),
            ("types", "types.xml"),
        ]
    }

    # Convert source metadata and assignment paths in bounded native batches.

    with tempfile.TemporaryDirectory(prefix="index_", dir=cache) as temporary:
        temporary = Path(temporary)
        pending = {
            name: str(temporary / Path(path).name) for name, path in paths.items()
        }
        processing.run(
            context,
            "index",
            trips=assigned["trips"],
            assigned=assigned["routes"],
            **pending,
        )
        for name, path in paths.items():
            Path(pending[name]).replace(path)
    return paths


def validate(context):
    return processing.source_token("index")
