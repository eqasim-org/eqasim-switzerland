"""Capacity-weighted physical road choices within surveyed interview places."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from matsim.simulation.cross_border_links import build_directional_border_link_table

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.cross_border.interview_places")
    context.stage("data.spatial.swiss_border")
    context.stage("matsim.scenario.network.pt")


def execute(context):

    # Match road references before either resident or foreign demand is sampled.

    points = context.stage("data.cross_border.interview_places").to_crs(2056).copy()
    roads = points[points.label.eq("road")].copy()
    network = context.stage("matsim.scenario.network.pt")
    facilities = roads.assign(
        entry_interview_point_id=roads.border_crossing_point_id + "_entry",
        exit_interview_point_id=roads.border_crossing_point_id + "_exit",
        entry_interview_geometry_point=roads.geometry,
        exit_interview_geometry_point=roads.geometry,
        trip_mode="car",
    )
    assignments = build_directional_border_link_table(
        facilities, None, network["network"], context.stage("data.spatial.swiss_border"),
        geometry_path=network["geometry"], context=context)

    # Collapse aliases only within each interview group, never across survey totals.

    candidates, audit = weight_candidates(points, assignments)
    audit.drop(columns="geometry").to_csv(Path(context.path()) / "road_candidate_weights.csv", index=False)
    assignments.to_csv(Path(context.path()) / "directional_links.csv", index=False)
    logger.info("Road choices: %d references, %d physical candidates; weighting methods: %s",
                len(roads), int(audit.is_candidate.sum()),
                audit.weight_method.value_counts().to_dict())

    # PT keeps the existing importance weights and all its original points.

    return candidates


def weight_candidates(points, assignments):
    """Keep original importance/IDs; expose sampling weights and alias diagnostics."""
    points = points.copy()
    points["selection_weight"] = points.importance
    roads = points[points.label.eq("road")].copy()
    for direction in ["entry", "exit"]:
        lookup = assignments[assignments.direction.eq(direction)].set_index("facility_id")
        ids = roads.border_crossing_point_id + "_" + direction
        for column in ["link_id", "selection_capacity", "match_type", "anchor_link_id", "road_class"]:
            roads[direction + "_" + column] = ids.map(lookup[column]).to_numpy()
    # The road survey observes inbound traffic. Use the inbound corridor's total
    # capacity (already including lanes), not the sum of the two carriageways.
    roads["physical_crossing"] = roads.entry_link_id.astype(str) + "|" + roads.exit_link_id.astype(str)
    unresolved = roads.entry_link_id.isna() | roads.exit_link_id.isna() | roads.entry_link_id.eq(roads.exit_link_id)
    roads.loc[unresolved, "physical_crossing"] = roads.loc[unresolved, "border_crossing_point_id"]
    keys = ["interview_place", "physical_crossing"]
    # The strongest original importance represents duplicate aliases. Adding them
    # would give a road more demand just because it had more reference points.
    ordered = roads.sort_values(["importance", "border_crossing_point_id"], ascending=[False, True])
    canonical = ordered.drop_duplicates(keys).set_index(keys).border_crossing_point_id
    roads["canonical_point_id"] = pd.MultiIndex.from_frame(roads[keys]).map(canonical)
    roads["is_candidate"] = roads.border_crossing_point_id.eq(roads.canonical_point_id)
    roads["weight_method"] = "capacity_times_importance"
    roads["selection_weight"] = 0.0
    for place, group in roads[roads.is_candidate].groupby("interview_place", sort=False):
        capacity = pd.to_numeric(group.entry_selection_capacity, errors="coerce")
        importance = pd.to_numeric(group.importance, errors="coerce")
        if not (np.isfinite(importance) & importance.ge(0)).all():
            raise ValueError("Invalid road importance in interview place %s" % place)
        if not (np.isfinite(capacity) & capacity.gt(0)).all():
            # Do not mix arbitrary substitute capacities with real capacities:
            # retain the old importance rule for this entire group and flag it.
            weights = importance.copy()
            method = "importance_missing_capacity"
            logger.warning("Missing road capacity at %s; retaining importance-only weights for that group.", place)
        else:
            weights = importance * capacity
            method = "capacity_times_importance"
        if weights.sum() <= 0:
            weights = pd.Series(1.0, index=group.index)
            method = "uniform_zero_importance"
        roads.loc[group.index, "selection_weight"] = weights
        roads.loc[roads.interview_place.eq(place), "weight_method"] = method
    totals = roads.groupby("interview_place").selection_weight.transform("sum")
    physical_weights = roads[roads.is_candidate].set_index("border_crossing_point_id").selection_weight
    roads["selection_weight"] = roads.canonical_point_id.map(physical_weights)
    alias_count = roads.groupby("canonical_point_id").canonical_point_id.transform("size")
    roads["group_share_before_detour"] = roads.selection_weight / alias_count / totals
    # Keep all coordinates eligible for the existing detour test. The sampling
    # helper divides each physical road's weight over its *eligible* aliases,
    # preventing duplicates from gaining weight without dropping useful anchors.
    result = pd.concat([roads, points[~points.label.eq("road")]], ignore_index=True)
    return result, roads
