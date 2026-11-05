"""Shared count-reference paths and checked physical crossing identities."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.cross_border.reference_files import reference_directory


def counts_path(context):
    configured = context.config("cross_border_assignment_counts_file")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else Path(context.config("data_path")) / path
    return reference_directory(context.config("data_path"), context.config("cross_border_reference_path")) / "traffic_count_references.json"


def check_reference(source, points):
    # BCP IDs are derived from source row order. Refuse a stale FID-to-BCP map.
    same = len(source) == len(points)
    if same:
        same = (all(a.equals(b) for a, b in zip(source.geometry, points.geometry))
                and np.array_equal(source.importance.astype(float), points.importance.astype(float))
                and source.crs == points.crs
                and all(np.array_equal(source[c].to_numpy(), points[c].to_numpy())
                        for c in ("label", "interview_place")))
    if not same:
        raise ValueError("Crossing reference differs from cached interview_places. Close the GIS editor, "
                         "refresh the GeoPackage timestamp and rerun data.cross_border.interview_places first.")


def build_sites(source, points, roads, exits):
    fids = dict(zip(points.border_crossing_point_id.astype(str), source.index.astype(int)))
    road = roads.loc[roads.label.eq("road")].copy()
    canonical = dict(zip(road.border_crossing_point_id.astype(str), road.canonical_point_id.astype(str)))
    rows = []
    for r in road.loc[road.is_candidate.eq(True)].itertuples():
        aliases = [p for p, target in canonical.items() if target == r.border_crossing_point_id]
        rows.append(dict(point_id=r.border_crossing_point_id, fid=fids[r.border_crossing_point_id],
                         aliases=";".join(str(fids[p]) for p in aliases), name=r.interview_place,
                         interview_place=r.interview_place, kind="entry_candidate", x=r.geometry.x, y=r.geometry.y,
                         importance=r.importance, capacity=r.entry_selection_capacity,
                         weight=r.selection_weight, prior_share=r.group_share_before_detour * len(aliases),
                         entry_link=str(r.entry_link_id), exit_link=str(r.exit_link_id)))
    known = {r["point_id"] for r in rows}
    for r in exits.itertuples():
        pid = canonical.get(str(r.border_crossing_point_id), str(r.border_crossing_point_id))
        if pid not in known:
            known.add(pid)
            rows.append(dict(point_id=pid, fid=fids.get(pid), name=r.survey_exit_name,
                             interview_place=r.interview_place, kind="exit_only", x=r.geometry.x, y=r.geometry.y))
        canonical[str(r.border_crossing_point_id)] = pid
    return pd.DataFrame(rows), canonical
