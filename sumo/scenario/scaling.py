"""Seeded columnar demand expansion and bounded sampling of real locations."""

import copy
import math
from pathlib import Path

from lxml import etree
import numpy as np
import polars as pl
import pyarrow as pa
from pyproj import Transformer
from scipy.spatial import cKDTree

from sumo.tables import as_polars

PREFIX = "sumo_scaled."


def settings(sample, spatial, temporal):
    values = [sample, spatial, temporal]
    if any(isinstance(value, bool) for value in values):
        raise ValueError("SUMO sample/window settings must be numeric, not boolean")
    sample, spatial, temporal = map(float, values)
    if not all(map(math.isfinite, (sample, spatial, temporal))) or not 0 < sample <= 1:
        raise ValueError("input_downsampling must be in (0, 1] and SUMO windows must be finite")
    if spatial < 0 or temporal < 0:
        raise ValueError("SUMO upsampling windows must be nonnegative")
    return 1 / sample, spatial, temporal


def replication_counts(number, factor, rng):
    """Randomly allocate the fractional remainder, with an exact rounded unit total."""
    lower = math.floor(factor)
    counts = np.full(number, lower, dtype=np.int64)
    extra = math.floor(number * (factor - lower) + .5)
    if extra:
        counts[rng.choice(number, size=extra, replace=False)] += 1
    return counts


def candidate_pools(homes, destinations):
    """Convert only needed external columns; index unique addresses, not people."""
    home = as_polars(homes[["home_x", "home_y"]])
    pools = {"home": _points(home, "home_x", "home_y")}
    columns = [name for name in destinations.columns if name in
               {"destination_x", "destination_y", "geometry", "offers_work", "offers_leisure",
                "offers_other", "offers_shop", "offers_education"}]
    frame = as_polars(destinations[columns])
    if not {"destination_x", "destination_y"}.issubset(frame.columns):
        import shapely
        geometry = shapely.from_wkb(frame["geometry"].to_numpy())
        frame = frame.with_columns(pl.Series("destination_x", shapely.get_x(geometry)),
                                   pl.Series("destination_y", shapely.get_y(geometry)))
    for purpose in ("work", "leisure", "other", "shop", "education"):
        flag = "offers_" + purpose
        selected = frame.filter(pl.col(flag).fill_null(False)) if flag in frame.columns else frame.clear()
        pools[purpose] = _points(selected, "destination_x", "destination_y")
    return pools


def _points(frame, x, y):
    frame = frame.select(pl.col(x).cast(pl.Float64, strict=False).alias("x"),
                         pl.col(y).cast(pl.Float64, strict=False).alias("y"))
    return (frame.filter(pl.all_horizontal(pl.all().is_finite()).fill_null(False)
                         & (pl.col("x") > 0) & (pl.col("y") > 0))
            .unique().sort("x", "y").to_numpy())


def prepare(frame, activities=None, resolver=None):
    """Attach activity purposes, unit IDs and consistent endpoint anchor indices."""
    if frame["id"].str.starts_with(PREFIX).any():
        raise ValueError(f"Source SUMO IDs must not use the reserved scaling prefix {PREFIX}")
    # Explicit external edge trips may need conversion to geographic endpoints once.
    absent = frame["origin"].is_null() | frame["destination"].is_null()
    if absent.any():
        rows = frame.to_dicts()
        for index in np.flatnonzero(absent.to_numpy()):
            row = rows[index]
            if row["xml"] is None:
                raise ValueError(f"Missing coordinates for SUMO scaling trip {row['id']}")
            attributes = etree.fromstring(row["xml"]).attrib
            parameters = {item.get("key"): item.get("value")
                          for item in etree.fromstring(row["xml"]).findall("param")}
            if "fromLonLat" in attributes and "toLonLat" in attributes:
                row["origin"], row["destination"] = attributes["fromLonLat"], attributes["toLonLat"]
            elif resolver is not None:
                row["origin"], row["destination"] = resolver.xml_coordinates(attributes)
            else:
                raise ValueError(f"Cannot resolve endpoints for SUMO scaling trip {row['id']}")
            row["person_id"] = row["person_id"] or parameters.get("person_id") or row["id"]
            row["trip_index"] = row["trip_index"] or parameters.get("trip_index") or "0"
            row["mode"] = row["mode"] or parameters.get("mode") or "car"
        frame = pl.DataFrame(rows, schema=frame.schema)
    # Population and external/commercial person namespaces cannot collide.
    frame = frame.with_columns(pl.concat_str(
        pl.when(pl.col("xml").is_null()).then(pl.lit("person:")).otherwise(pl.lit("vehicle:")),
        pl.col("person_id").fill_null(pl.col("id"))).alias("_unit"))
    if activities is not None:
        # Border clipping adds XML to population rows; preserve their person namespace.
        population = (pl.col("xml").is_null() | pl.col("xml").cast(pl.String)
                      .str.contains('key="_sumo_population" value="true"', literal=True).fill_null(False))
        frame = frame.with_columns(pl.when(population)
            .then(pl.concat_str(pl.lit("person:"), pl.col("person_id"))).otherwise(pl.col("_unit")).alias("_unit"))
        activity = as_polars(activities).select("person_id", "activity_index", "purpose").sort("person_id", "activity_index")
        activity = activity.with_columns(pl.col("purpose").shift(-1).over("person_id").alias("_destination_purpose"))
        activity = activity.select(pl.col("person_id").cast(pl.String),
            pl.col("activity_index").cast(pl.String).alias("trip_index"),
            pl.col("purpose").alias("_origin_purpose"), "_destination_purpose")
        frame = frame.join(activity, on=["person_id", "trip_index"], how="left", validate="m:1", maintain_order="left")
    else:
        frame = frame.with_columns(pl.lit(None, pl.String).alias("_origin_purpose"),
                                   pl.lit(None, pl.String).alias("_destination_purpose"))
    transform = Transformer.from_crs(4326, 2056, always_xy=True)
    endpoint_parts = []
    for name in ("origin", "destination"):
        coordinates = frame[name].str.split_exact(",", 1)
        x, y = transform.transform(coordinates.struct.field("field_0").cast(pl.Float64).to_numpy(),
                                   coordinates.struct.field("field_1").cast(pl.Float64).to_numpy(), errcheck=True)
        if not np.isfinite([x, y]).all():
            raise ValueError("SUMO scaling requires finite geographic endpoints")
        purpose = pl.col("_" + name + "_purpose").fill_null("unknown")
        purpose = pl.when(purpose.is_in(["home", "work", "leisure", "shop", "education", "other", "unknown"]))
        purpose = purpose.then(pl.col("_" + name + "_purpose").fill_null("unknown")).otherwise(pl.lit("other"))
        endpoint_parts.append(frame.select("_unit", purpose.alias("purpose"))
                              .with_columns(pl.Series("x", x), pl.Series("y", y)))
    anchors = pl.concat(endpoint_parts).unique(maintain_order=True).with_row_index("_anchor")
    units = frame.group_by("_unit").agg(pl.col("depart").min().alias("first_departure"),
        pl.col("person_id").first().alias("source_person_id"),
        pl.col("_unit").str.starts_with("person:").all().alias("population_chain")).sort("_unit").with_row_index("_unit_index")
    frame = frame.join(units.select("_unit", "_unit_index"), on="_unit", how="left", validate="m:1", maintain_order="left")
    anchors = anchors.join(units.select("_unit", "_unit_index"), on="_unit", how="left", validate="m:1", maintain_order="left")
    for name, endpoints in zip(("origin", "destination"), endpoint_parts):
        keys = endpoints.join(anchors.select("_unit", "purpose", "x", "y", "_anchor"),
                              on=["_unit", "purpose", "x", "y"], how="left", validate="m:1", maintain_order="left")
        frame = frame.with_columns(keys["_anchor"].alias("_" + name + "_anchor"))
    return frame, units, anchors


def sample_anchors(anchors, counts, pools, radius, rng, threads):
    """Query each distinct source location once, then sample entire clone vectors."""
    coordinates = anchors.select("x", "y").to_numpy()
    copies = int(counts.max()) if len(counts) else 1
    result = np.repeat(coordinates[:, None, :], copies, axis=1)
    report = {"anchors": anchors.height, "anchors_without_alternatives": 0, "relocated_clone_endpoints": 0}
    if radius == 0 or copies <= 1:
        return result, report
    unit_indices = anchors["_unit_index"].to_numpy()
    categories = anchors["purpose"].to_numpy().copy()
    if "_fixed" in anchors.columns:
        categories[anchors["_fixed"].to_numpy()] = "border"
    unknown = np.flatnonzero(categories == "unknown")
    if len(unknown):
        # Commercial files have no activity purposes. Recognize existing home
        # addresses first, then use eligible workplaces for remaining endpoints.
        categories = categories.copy()
        categories[unknown] = "work"
        homes = pools.get("home", np.empty((0, 2)))
        if len(homes):
            distance, _ = cKDTree(homes).query(coordinates[unknown], distance_upper_bound=1., workers=threads)
            categories[unknown[np.isfinite(distance)]] = "home"
    for category in sorted(set(categories)):
        members = np.flatnonzero(categories == category)
        candidates = pools.get(category, np.empty((0, 2)))
        if not len(candidates):
            report["anchors_without_alternatives"] += len(members)
            continue
        tree = cKDTree(candidates)
        unique, inverse = np.unique(coordinates[members], axis=0, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(inverse))]
        for begin in range(0, len(unique), 2000):
            neighborhoods = tree.query_ball_point(unique[begin:begin+2000], radius, workers=threads, return_sorted=True)
            for relative, options in enumerate(neighborhoods):
                index = begin + relative
                options = np.asarray(options, dtype=np.int64)
                if len(options):
                    delta = candidates[options] - unique[index]
                    # Geographic round trips can differ by millimetres; do not
                    # mistake the original address for a new location.
                    options = options[np.einsum("ij,ij->i", delta, delta) > 1.]
                group = members[order[starts[index]:starts[index+1]]]
                if not len(options):
                    report["anchors_without_alternatives"] += len(group)
                    continue
                for anchor in group:
                    number = counts[unit_indices[anchor]] - 1
                    if number:
                        result[anchor, 1:number+1] = candidates[options[rng.integers(len(options), size=number)]]
                        report["relocated_clone_endpoints"] += int(number)
    return result, report


def time_windows(units, anchors, window):
    """Use a wider symmetric window for every chain touching a fixed border anchor."""
    border = np.zeros(units.height, dtype=bool)
    if "_fixed" in anchors.columns:
        indices = anchors.filter(pl.col("_fixed"))["_unit_index"].to_numpy()
        border[indices] = True
    return np.where(border, max(1800., 2 * window), window), border


def offsets(units, counts, window, rng):
    """Sample one shift per clone in [-window, +window], avoiding negative departures."""
    copies = int(counts.max()) if len(counts) else 1
    result = np.zeros((units.height, copies))
    if copies > 1:
        windows = np.broadcast_to(np.asarray(window, dtype=float), (units.height,))
        lower = np.maximum(-windows, -units["first_departure"].to_numpy())
        result[:, 1:] = lower[:, None] + rng.random((units.height, copies-1)) * (windows - lower)[:, None]
    return result


def expanded_batches(frame, counts, anchors, shifts, schema, batch_size=250000):
    """Expand copy layers in bounded Arrow batches; no per-vehicle spatial queries."""
    reverse = Transformer.from_crs(2056, 4326, always_xy=True)
    templates = {i: etree.fromstring(value) for i, value in enumerate(frame["xml"]) if value is not None}
    columns = schema.names
    source_schema = frame.select(columns).schema
    frame = frame.with_row_index("_source_row")
    for clone in range(int(counts.max()) if len(counts) else 1):
        eligible = frame.filter(pl.Series(counts[frame["_unit_index"].to_numpy()] > clone))
        for batch in eligible.iter_slices(batch_size):
            if clone == 0:
                yield batch.select(columns).cast(source_schema)
                continue
            source_rows = batch["_source_row"].to_numpy()
            unit = batch["_unit_index"].to_numpy()
            xy = [anchors[batch["_" + name + "_anchor"].to_numpy(), clone] for name in ("origin", "destination")]
            endpoint_columns = []
            for name, points in zip(("origin", "destination"), xy):
                previous = anchors[batch["_" + name + "_anchor"].to_numpy(), 0]
                moved = np.any(points != previous, axis=1)
                coordinates = batch[name].to_numpy().copy()
                if moved.any():
                    lon, lat = reverse.transform(points[moved, 0], points[moved, 1], errcheck=True)
                    coordinates[moved] = [f"{a:.10f},{b:.10f}" for a, b in zip(lon, lat)]
                endpoint_columns.append(pl.Series(name, coordinates))
            output = batch.with_columns(*endpoint_columns,
                pl.concat_str(pl.lit(f"{PREFIX}{clone}."), pl.col("id")).alias("id"),
                pl.concat_str(pl.lit(f"{PREFIX}{clone}."), pl.col("person_id")).alias("person_id"),
                pl.col("source_trip_id").fill_null(pl.col("id")),
                (pl.col("depart") + pl.Series(shifts[unit, clone])).alias("depart"))
            payloads = output["xml"].to_list()
            for row in map(int, np.flatnonzero(output["xml"].is_not_null().to_numpy())):
                trip = copy.deepcopy(templates[int(source_rows[row])])
                for key in ("from", "to", "fromXY", "toXY", "departPos", "arrivalPos"):
                    trip.attrib.pop(key, None)
                trip.set("id", output["id"][row]); trip.set("depart", format(output["depart"][row], ".12g"))
                trip.set("fromLonLat", output["origin"][row]); trip.set("toLonLat", output["destination"][row])
                metadata = {"person_id": output["person_id"][row], "departure_time": output["depart"][row],
                            "origin_x": xy[0][row, 0], "origin_y": xy[0][row, 1],
                            "destination_x": xy[1][row, 0], "destination_y": xy[1][row, 1]}
                for parameter in trip.findall("param"):
                    if parameter.get("key") in metadata:
                        parameter.set("value", str(metadata[parameter.get("key")]))
                payloads[row] = etree.tostring(trip)
            output = output.with_columns(pl.Series("xml", payloads, dtype=pl.Binary))
            yield output.select(columns).cast(source_schema)


def write_expanded(batches, path, schema):
    count, loading, first, last = 0, 0, math.inf, -math.inf
    with pa.OSFile(str(path), "wb") as stream, pa.ipc.new_file(stream, schema) as writer:
        for batch in batches:
            writer.write_table(batch.to_arrow().cast(schema))
            count += batch.height
            loading += int(batch["loads_network"].sum() or 0)
            if batch.height:
                first, last = min(first, batch["depart"].min()), max(last, batch["depart"].max())
    return {"table": str(Path(path).resolve()), "count": count, "loading_count": loading,
            "first_departure": first if count else None, "last_departure": last if count else None}
