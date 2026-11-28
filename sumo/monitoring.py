"""Build a quality report from the completed regional simulation outputs."""

import hashlib
from pathlib import Path

from scripts.monitoring import monitored, plots
from scripts.monitoring.quality import sumo as quality
from scripts.monitoring.quality.sumo import generate_quality


def configure(context):
    context.stage("sumo.run")


@monitored
def execute(context):

    # Summarize structured outputs without loading routes or changing the model.

    return generate_quality(context.stage("sumo.run"), Path(context.path()) / "quality")


def validate(context):
    return [
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (quality, plots)
    ]
