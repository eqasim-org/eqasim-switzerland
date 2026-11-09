"""Compare RDA assignments with and without typical business opening hours."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rda import plots
from synthesis.population.spatial.secondary.locations_rda import prepare_locations


def configure(context):
    context.config("analysis_path")
    context.config("analysis.rda.city_radius_m", default=15000.)
    context.config("opening_hours_tolerance_minutes", default=10.)
    context.stage("synthesis.population.spatial.secondary.locations_rda",
                  config={"consider_opening_hours": False}, alias="without_hours")
    context.stage("synthesis.population.spatial.secondary.locations_rda",
                  config={"consider_opening_hours": True}, alias="with_hours")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.destinations_opening_hours")
    for stage in ("synthesis.population.sampled", "synthesis.population.spatial.home.locations",
                  "synthesis.population.spatial.primary.locations"):
        context.stage(stage)


def execute(context):

    # Resolve the shared trips, anchors and schedules for both completed runs.

    output = Path(context.config("analysis_path")) / "rda"
    output.mkdir(parents=True, exist_ok=True)
    trips = context.stage("synthesis.population.trips").sort_values(["person_id", "trip_index"])
    primary = prepare_locations(context)
    hours = context.stage("synthesis.population.destinations_opening_hours")
    radius = float(context.config("analysis.rda.city_radius_m"))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("analysis.rda.city_radius_m must be positive")

    # Reconstruct activity locations and incoming distances on identical keys.

    variants, convergence = {}, []
    for variant in ("without_hours", "with_hours"):
        secondary, checks = context.stage(variant)
        variants[variant] = plots.activity_metrics(trips, primary, secondary, hours, radius,
            float(context.config("opening_hours_tolerance_minutes")) * 60.)
        convergence.append(dict(variant=variant, problems=len(checks),
                                valid_problems=int(checks.valid.sum()), valid_share=float(checks.valid.mean()),
                                variable_activities=int(checks["size"].sum())))

    # Export paired changes, city summaries and distributions.

    return plots.generate(variants, convergence, radius, output,
        float(context.config("opening_hours_tolerance_minutes")) * 60.)


def validate(context):
    return hashlib.sha256(Path(plots.__file__).read_bytes()).hexdigest()
