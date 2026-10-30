"""Pooled domestic tertiary flows with geographic and parent-origin smoothing."""

import hashlib
from pathlib import Path

from data.od.education_matrix import prepare_observations
from data.od.tertiary_education_smoothing import (
    build_smoothed_matrix, DEFAULT_PRIOR_STRENGTH, DEFAULT_NEIGHBORS, DEFAULT_BANDWIDTH, DEFAULT_DISTANCE_DECAY)


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.config("tertiary_od_prior_strength", DEFAULT_PRIOR_STRENGTH)
    context.config("tertiary_od_neighbors", DEFAULT_NEIGHBORS)
    context.config("tertiary_od_bandwidth", DEFAULT_BANDWIDTH)
    context.config("tertiary_od_distance_decay", DEFAULT_DISTANCE_DECAY)


def validate(context):
    # The preparation functions are imported rather than separate dependencies.
    # Detect changes to the shared origin policy as well as this wrapper's code.
    files = [Path(__file__).with_name("education_matrix.py"),
             Path(__file__).with_name("tertiary_education_smoothing.py"),
             Path(__file__).parents[1] / "structural_survey/education_origins.py"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def execute(context):
    df_od = prepare_observations(
        context.stage("data.structural_survey.structural_survey"), tertiary_only=True)
    municipalities, mapping = context.stage("data.spatial.municipalities")
    pdf, cdf, diagnostics = build_smoothed_matrix(
        df_od, context.stage("data.spatial.zones"), municipalities,
        context.stage("data.spatial.quarters"), mapping,
        context.config("tertiary_od_prior_strength"), context.config("tertiary_od_neighbors"),
        context.config("tertiary_od_bandwidth"), context.config("tertiary_od_distance_decay"))
    return pdf, cdf
