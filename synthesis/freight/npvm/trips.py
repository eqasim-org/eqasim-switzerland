"""Create NPVM heavy-goods trips with the legacy freight DataFrame contract."""

import numpy as np
import pandas as pd

COLUMNS = ["agent_id", "origin_x", "origin_y", "destination_x", "destination_y",
           "departure_time", "vehicle_type"]


def configure(context):
    context.stage("synthesis.freight.npvm.od")
    context.stage("synthesis.freight.npvm.crossing_points")
    context.stage("data.freight.departure_times")
    context.config("random_seed")


def execute(context):

    # Load sampled OD counts, zone coordinates and the freight departure-time profile.

    od = context.stage("synthesis.freight.npvm.od")
    locations = context.stage("synthesis.freight.npvm.crossing_points")
    departures = context.stage("data.freight.departure_times")
    rng = np.random.default_rng(context.config("random_seed"))
    border_rng = np.random.default_rng(np.random.SeedSequence([context.config("random_seed"), 1]))

    # Expand each integer OD count into trips with independently sampled endpoints.

    routed = locations.get("routed_crossings")
    trips = expand_trips(od, locations["points"], rng, locations.get("border_points"),
                         locations.get("border_probabilities"), routed, border_rng)

    # Assign freight-profile departure hours and uniformly distributed seconds within each hour.

    hours = departures.hour.to_numpy(dtype=float)
    probabilities = departures.probability.to_numpy(dtype=float)
    if (not len(hours) or not np.isfinite(hours).all() or (hours < 0).any()
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1.0)):
        raise ValueError("Invalid freight departure-hour probability profile")
    trips["departure_time"] = (3600 * rng.choice(hours, len(trips), p=probabilities)
                                + rng.integers(0, 3600, len(trips))).astype(np.int64)

    # Preserve the seven-column freight contract and the existing MATSim truck mode.

    trips["agent_id"] = np.arange(len(trips), dtype=np.int64)
    trips["vehicle_type"] = "truck"
    return trips[COLUMNS]


def expand_trips(od, points, rng, border_points=None, border_probabilities=None, routed=None, border_rng=None):
    """Keep row order stable and never discard a sampled OD pair without coordinates."""
    border_rng = rng if border_rng is None else border_rng
    border_points = border_points or {"in": {}, "out": {}}
    border_probabilities = border_probabilities or {"in": {}, "out": {}}
    routed_pairs = {} if routed is None or routed.empty else {
        key: group for key, group in routed.groupby(["origin_zone", "destination_zone"])}
    records = []
    for row in od.itertuples(index=False):
        count = int(row.n_trips)
        if count <= 0:
            continue
        # A foreign origin enters Switzerland; a foreign destination exits it, including transit.
        origin = border_points["in"].get(row.origin_zone, points.get(row.origin_zone))
        destination = border_points["out"].get(row.destination_zone, points.get(row.destination_zone))
        choices = routed_pairs.get((row.origin_zone, row.destination_zone))
        if choices is not None:
            if origin is not None:
                a = sample_points(origin, count, rng)
            selected = choices.iloc[border_rng.choice(len(choices), count, p=choices.probability.to_numpy())]
            if origin is None:
                a = selected[["origin_x", "origin_y"]].to_numpy()
            b = selected[["destination_x", "destination_y"]].to_numpy() if destination is None else sample_points(destination, count, rng)
        elif origin is None or destination is None:
            raise ValueError(f"Missing sampling points for NPVM OD {row.origin_zone} → {row.destination_zone}")
        else:
            a = sample_points(origin, count, border_rng if row.origin_zone in border_points["in"] else rng,
                              border_probabilities["in"].get(row.origin_zone))
            b = sample_points(destination, count, border_rng if row.destination_zone in border_points["out"] else rng,
                              border_probabilities["out"].get(row.destination_zone))
        records.append(pd.DataFrame({"origin_zone": row.origin_zone, "destination_zone": row.destination_zone,
            "origin_x": a[:, 0], "origin_y": a[:, 1], "destination_x": b[:, 0], "destination_y": b[:, 1],
            "source": row.source}))
    if not records:
        return pd.DataFrame({name: pd.Series(dtype="int64" if name.endswith("zone") else
                            "object" if name == "source" else "float64") for name in
                            ["origin_zone", "destination_zone", "origin_x", "origin_y",
                             "destination_x", "destination_y", "source"]})
    return pd.concat(records, ignore_index=True)


def sample_points(points, count, rng, probabilities=None):
    """Use hierarchy probabilities at border points, equal sampling for domestic points."""
    if probabilities is None:
        return points[rng.integers(len(points), size=count)]
    return points[rng.choice(len(points), size=count, p=probabilities)]
