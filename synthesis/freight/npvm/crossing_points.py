"""Return ready-to-use foreign crossing assignments from one selected source."""

import logging

import pandas as pd

from synthesis.freight.npvm.crossings import select_crossings

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("synthesis.freight.npvm.zone_locations")
    if not context.config("npvm_freight_use_border_crossings", default=True):
        return
    source = context.config("npvm_freight_crossing_source", default=None)
    interview = context.config("npvm_freight_use_interview_places", default=False)
    routed = context.config("npvm_freight_route_border_crossings", default=True)
    source = source or ("interview" if interview else "route" if routed else "network")
    if source not in {"interview", "network", "route"}:
        raise ValueError("npvm_freight_crossing_source must be interview, network or route")
    if source == "route":
        context.stage("synthesis.freight.npvm.routed_crossings")
    else:
        context.stage("synthesis.freight.npvm.crossings", config={
            "npvm_freight_use_interview_places": source == "interview"}, alias="source_crossings")


def execute(context):

    # Copy the base geography; upstream results remain unchanged.

    result = dict(context.stage("synthesis.freight.npvm.zone_locations"))
    result.update(border_assignments=pd.DataFrame(), border_points={"in": {}, "out": {}},
                  border_probabilities={"in": {}, "out": {}}, foreign_access=pd.DataFrame(),
                  routed_crossings=None, crossing_source="geometric")
    foreign = result["zones"].loc[lambda frame: ~frame.is_swiss]
    if not result["use_border_crossings"]:
        return result
    source = context.config("npvm_freight_crossing_source") or (
        "interview" if context.config("npvm_freight_use_interview_places") else
        "route" if context.config("npvm_freight_route_border_crossings") else "network")

    # Routed assignments already contain paired entry/exit points and probabilities.

    if source == "route":
        assignments = context.stage("synthesis.freight.npvm.routed_crossings")
        result.update(crossing_source="network_routed", border_assignments=assignments,
                      routed_crossings=assignments)
        return result

    # Adapt interview or network points into the same directional zone pools.

    crossings = context.stage("source_crossings")
    assignments = (pd.concat([select_crossings(foreign, crossings, direction)
                              for direction in ("in", "out")], ignore_index=True)
                   if not foreign.empty else pd.DataFrame())
    if not assignments.empty:
        for (direction, zone), candidates in assignments.groupby(["direction", "zone_id"], sort=False):
            result["border_points"][direction][zone] = candidates[["crossing_x", "crossing_y"]].to_numpy(dtype=float)
            result["border_probabilities"][direction][zone] = candidates.probability.to_numpy(dtype=float)
    result.update(crossing_source="interview_places" if source == "interview" else "network",
                  border_assignments=assignments)
    logger.info("Prepared %d %s crossing candidates", len(assignments), source)
    return result
