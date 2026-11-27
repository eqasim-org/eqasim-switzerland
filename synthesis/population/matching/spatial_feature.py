"""Attach the same home activity zone to survey donors and synthetic recipients.

V1/V3 replace their density preference with this category. V2 retains density
and adds the zone to its categorical geographic inputs.
"""

import logging
from . import opening_hours

OPTION = "matching_spatial_feature"
STAGE = "data.microcensus.activity_chains_clusters"
DENSITY = "population_density_class"
ZONE = "activity_chain_cluster"
logger = logging.getLogger("synpp")


def configure(context):
    feature = context.config(OPTION, default="activity_chain_cluster")
    if feature not in ("density", "activity_chain_cluster"):
        raise ValueError(f"{OPTION} must be density or activity_chain_cluster")
    if feature == "activity_chain_cluster":
        context.stage(STAGE)


def add_regions(context, source, population):
    if context.config(OPTION) != "activity_chain_cluster":
        return source.drop(columns=ZONE, errors="ignore"), population.drop(
            columns=ZONE, errors="ignore"
        )
    municipalities = context.stage(STAGE)["municipalities"]
    return assign_regions(source, municipalities), assign_regions(
        population, municipalities
    )


def assign_regions(frame, municipalities):
    """Preserve rows and resolve municipal boundary ties by municipality ID.

    As with opening-hours geography, finite homes outside polygon coverage use
    the nearest zone, with deterministic boundary ties. This fallback is logged
    by the common helper rather than introducing a permissive missing category.
    """
    if not municipalities[ZONE].isin([1, 2, 3, 4]).all():
        raise ValueError("Municipal matching requires labels in 1..4")
    labelled = municipalities[["municipality_id", "geometry"]].rename(
        columns={"municipality_id": opening_hours.REGION}
    )
    assigned = opening_hours.assign_regions(frame, labelled)
    lookup = municipalities.set_index("municipality_id")[ZONE]
    result = frame.copy()
    result[ZONE] = assigned[opening_hours.REGION].map(lookup).to_numpy(dtype="int8")
    if not result[ZONE].isin([1, 2, 3, 4]).all():
        raise ValueError("Daily-trip matching requires four valid home zones")
    return result


def columns(names, source, population):
    """Replace one preference in place, retaining its priority and relaxation rule."""
    if names is None:
        return None
    if ZONE not in population:
        return list(names)
    if ZONE not in source:
        raise ValueError("Both matching frames require activity-chain zones")
    return [ZONE if name == DENSITY else name for name in names]
