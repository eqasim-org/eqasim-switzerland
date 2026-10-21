"""Execute a generated micro/mesoscopic config in an independent stage cache."""

from pathlib import Path
import shutil

from sumo.config import tools
from sumo.runtime import sumo


def configure(context):
    context.config("sumo_simulation_model", "microscopic")
    model = context.config("sumo_simulation_model")
    if model not in {"microscopic", "mesoscopic"}:
        raise ValueError("sumo_simulation_model must be 'microscopic' or 'mesoscopic'")
    context.stage("sumo.config.config")
    context.stage("sumo.runtime.sumo")


def execute(context):

    # Copy configuration and generated detectors into this execution's directory.
    configs = context.stage("sumo.config.config")
    simulation_model = context.config("sumo_simulation_model")
    source = Path(configs[simulation_model])
    cache = Path(context.path()).resolve()
    statistics = source.parent / "edge_statistics.add.xml"
    shutil.copyfile(statistics, cache / statistics.name)
    config = tools.relocate_config(source, cache / source.name, {statistics.resolve(): cache / statistics.name})

    # Run with independent outputs while preserving all upstream inputs.
    sumo.run(context, ["--configuration-file", config])
    return {"config": config, "tripinfo": str(cache / "tripinfo.xml.gz"),
            "edge_data": str(cache / "edge_data.xml.gz"), "summary": str(cache / "summary.xml.gz")}
