import numpy as np


def _distance_m(lat1, lon1, lat2, lon2):
    return 111_320 * np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(np.radians(lat1)))


def select_line_ids(counts, reference, stop_coordinates, max_distance = 250.0, label = "reference"):
    """The MATSim counts are matched to the observed counts by line name, but several transit lines (line_id) can have
    the same name: N1 is a night bus and a lake boat, 13 or 17 are buses in Lausanne and trams in Geneva, 3 or 5 exist
    in Geneva and in the French network. For each name with several transit lines in counts, only the line that serves
    most of the stops where the observed data has that line is kept.

    counts: MATSim counts with the columns line_id, line_name and stop_id_gtfs_base.
    reference: data frame with the columns line_name and gtfs_code, the stops of each line in the observed data.
    A stop counts as served when the transit line has a stop within max_distance metres."""
    ids_per_name = counts.groupby("line_name")["line_id"].nunique()
    reference    = reference.assign(line_name = reference["line_name"].astype(str))
    kept         = counts

    for name in ids_per_name[ids_per_name > 1].index:
        codes = [c for c in reference.loc[reference["line_name"] == name, "gtfs_code"].unique() if c in stop_coordinates]
        if not codes:
            continue  # the observed data does not have this line: nothing is compared

        overlap = {}
        for line_id, group in counts[counts["line_name"] == name].groupby("line_id"):
            stops = [stop_coordinates[c] for c in group["stop_id_gtfs_base"].unique() if c in stop_coordinates]
            overlap[line_id] = sum(
                any(_distance_m(*stop_coordinates[code], *stop) <= max_distance for stop in stops) for code in codes
            )

        best   = max(overlap.values())
        detail = ", ".join(f"{line_id}: {n}/{len(codes)}" for line_id, n in sorted(overlap.items(), key = lambda kv: -kv[1]))
        print(f"Line {name} has {len(overlap)} transit lines; {label} stops served by each: {detail}")

        if best > 0:
            kept = kept[(kept["line_name"] != name) | kept["line_id"].map(lambda i: overlap.get(i) == best)]

    return kept
