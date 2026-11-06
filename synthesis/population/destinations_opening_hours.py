"""Enrich shop/leisure destination options with observed or locally inferred hours."""

from collections import Counter
from datetime import time
import hashlib
import json
import logging
from numbers import Integral, Real
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.spatial import cKDTree

import data.activity_purposes as activity_purposes
from data.activity_purposes import SHOP, LEISURE, noga_activity_types, normalize_noga, osm_activity_types
from data.utils import coerce_boolean_series


logger = logging.getLogger("synpp: destinations_opening_hours")
OPTIONS = {
    "match_radius_m": 50.0, "local_radius_m": 2000.0, "max_radius_m": 10000.0,
    "min_donors": 3, "max_neighbors": 128,
}
HOUR_FIELDS = {
    "opens_at": "opens_at", "closes_at": "closes_at", "closes_day_offset": "closes_day_offset",
    "open_intervals": "open_intervals", "hours_status": "status", "hours_source": "source",
    "hours_business_type": "business_type", "hours_donor_osm_id": "donor_osm_id",
    "hours_distance_m": "distance_m", "hours_radius_m": "radius_m",
    "hours_donor_count": "donor_count", "hours_support_count": "support_count",
    "hours_support_fraction": "support_fraction",
}
ASSIGNMENT_COLUMNS = ["destination_id", "purpose", "noga", "is_statent_destination",
                      "business_type", "status", "source", "open_intervals", "opens_at", "closes_at",
                      "closes_day_offset", "donor_osm_id", "distance_m", "radius_m",
                      "donor_count", "support_count", "support_fraction"]


def configure(context):
    context.stage("synthesis.population.destinations")
    context.stage("data.statent.statent")
    context.stage("data.osm.businesses")
    for name, default in OPTIONS.items():
        context.config("destinations_opening_hours." + name, default=default)


def execute(context):

    # Join STATENT provenance/NOGA without changing the upstream catalogue.

    destinations = context.stage("synthesis.population.destinations")
    statent = context.stage("data.statent.statent")
    osm_result = context.stage("data.osm.businesses")
    businesses = gpd.read_parquet(osm_result["businesses"])
    options = {name: context.config("destinations_opening_hours." + name) for name in OPTIONS}

    # Match compatible nearby businesses, then infer a local type-specific mode.

    enriched, assignments, report = assign_opening_hours(destinations, statent, businesses, **options)

    # Save a reusable destination catalogue and one audit row per purpose option.

    output = Path(context.path())
    enriched.to_parquet(output / "destinations.parquet", index=False)
    assignments.to_parquet(output / "opening_hours.parquet", index=False)
    report["options"] = options
    report["interval_unit"] = "minutes from local midnight; overnight ends can exceed 1440"
    report["timezone"] = "Europe/Zurich"
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Destination hours: %s", report["sources"])

    # Return the enriched frame for future RDA/NN consumers; no assignment changes yet.

    return enriched


def assign_opening_hours(destinations, statent, businesses, **options):
    settings = {**OPTIONS, **options}
    validate_options(settings)
    if destinations.crs is None or businesses.crs is None:
        raise ValueError("Destination and OSM geometries must have a CRS")
    if destinations.destination_id.isna().any() or destinations.destination_id.duplicated().any():
        raise ValueError("Destinations must have unique, non-null destination_id values")
    result = destinations.copy()
    noga, is_statent = join_statent(destinations, statent)
    result["noga"], result["is_statent_destination"] = noga, is_statent
    metric = destinations.to_crs(2056)
    points = metric.geometry.to_numpy()
    destination_ids = destinations.destination_id.to_numpy()
    noga_types = {code: noga_activity_types(code) for code in np.unique(noga)}
    donors, pool_members = prepare_donors(businesses)
    rows, groups = [], {}
    for purpose in (SHOP, LEISURE):
        eligible = coerce_boolean_series(result["offers_" + purpose], name="offers_" + purpose).to_numpy()
        for position in np.flatnonzero(eligible):
            code = noga[position]
            kind = noga_types[code].get(purpose, purpose)
            record = dict(destination_id=destination_ids[position], purpose=purpose,
                          noga=code or None, is_statent_destination=bool(is_statent[position]),
                          business_type=kind, status="unknown", source="unmatched", open_intervals=None,
                          opens_at=None, closes_at=None, closes_day_offset=None, donor_osm_id=None,
                          distance_m=None, radius_m=None, donor_count=0, support_count=0, support_fraction=None)
            rows.append(record)
            if code and purpose not in noga_types[code]:
                record["source"] = "noga_purpose_mismatch"
                continue
            point = points[position]
            if point is None or point.is_empty or point.geom_type != "Point" or not np.isfinite([point.x, point.y]).all():
                record["source"] = "invalid_location"
                continue
            groups.setdefault((purpose, kind), []).append((record, (point.x, point.y)))
    for key, requests in groups.items():
        members = pool_members.get(key, [])
        if not members:
            continue
        pool = donors.iloc[members].reset_index(drop=True)
        assign_pool(requests, pool, key[1] == key[0], settings)
    assignments = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    for name in ("opens_at", "closes_at"):
        assignments[name] = pd.array(assignments[name], dtype=pd.ArrowDtype(pa.time32("s")))
    assignments["closes_day_offset"] = pd.array(assignments.closes_day_offset, dtype="Int64")
    for purpose in (SHOP, LEISURE):
        assigned = assignments.loc[assignments.purpose.eq(purpose)].set_index("destination_id")
        for suffix, field in HOUR_FIELDS.items():
            values = result.destination_id.map(assigned[field])
            name = purpose + "_" + suffix
            if field in ("status", "source"):
                values = values.fillna("not_applicable")
            if field in ("opens_at", "closes_at"):
                result[name] = pd.array(
                    [None if pd.isna(value) else value for value in values],
                    dtype=pd.ArrowDtype(pa.time32("s")))
            elif field in ("closes_day_offset", "donor_count", "support_count"):
                result[name] = pd.array(values, dtype="Int64")
            else:
                result[name] = values.to_numpy()
    report = {
        "destinations": len(destinations), "statent_destinations": int(is_statent.sum()),
        "purpose_options": len(assignments), "usable_osm_businesses": len(donors),
        "sources": assignments.source.value_counts().to_dict(),
        "statuses": assignments.status.value_counts().to_dict(),
    }
    return result, assignments, report


def validate_options(options):
    match, local, maximum = (options[key] for key in ("match_radius_m", "local_radius_m", "max_radius_m"))
    if not all(isinstance(value, Real) and np.isfinite(value) for value in (match, local, maximum)):
        raise ValueError("Opening-hours radii must be finite numbers")
    if not 0 <= match <= local <= maximum or local <= 0:
        raise ValueError("Require 0 <= match_radius_m <= local_radius_m <= max_radius_m, with local_radius_m > 0")
    minimum, neighbors = options["min_donors"], options["max_neighbors"]
    if not all(isinstance(value, Integral) and not isinstance(value, bool) and value >= 1 for value in (minimum, neighbors)) or neighbors < minimum:
        raise ValueError("Require integer max_neighbors >= min_donors >= 1")


def identifier_key(value):
    if pd.isna(value):
        return None
    if isinstance(value, Real) and not isinstance(value, bool) and float(value).is_integer():
        return str(int(value))
    return str(value)


def join_statent(destinations, statent):
    registry = statent[["enterprise_id", "noga"]].copy()
    registry["key"] = registry.enterprise_id.map(identifier_key)
    registry["noga"] = registry.noga.map(normalize_noga)
    registry = registry[registry.key.notna()]
    if registry.groupby("key").noga.nunique().gt(1).any():
        raise ValueError("Conflicting NOGA codes for one STATENT enterprise_id")
    registry = registry.drop_duplicates("key").set_index("key")
    keys = destinations.destination_id.map(identifier_key)
    if keys.duplicated().any():
        raise ValueError("Destination IDs collide after normalizing numeric identifiers")
    return keys.map(registry.noga).fillna("").to_numpy(), keys.isin(registry.index).to_numpy()


def schedule_values(schedule):
    intervals = json.loads(schedule)
    if not isinstance(intervals, list):
        raise ValueError("Opening intervals must be a JSON list")
    previous = 0
    for pair in intervals:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Opening intervals must contain start/end pairs")
        start, end = pair
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in pair) or not previous <= start < 1440 or not start < end:
            raise ValueError("Invalid opening interval")
        previous = end
    opening, closing = (intervals[0][0], intervals[-1][1]) if intervals else (None, None)
    return dict(open_intervals=json.dumps(intervals, separators=(",", ":")),
                status="open" if intervals else "closed",
                opens_at=time(opening // 60 % 24, opening % 60) if opening is not None else None,
                closes_at=time(closing // 60 % 24, closing % 60) if closing is not None else None,
                closes_day_offset=closing // 1440 if closing is not None else None)


def prepare_donors(businesses):
    metric = businesses.to_crs(2056)
    rows, members = [], {}
    for record in metric.to_dict("records"):
        if record.get("opening_hours_status", "parsed") != "parsed":
            continue
        point = record["geometry"]
        if point is None or point.is_empty or point.geom_type != "Point" or not np.isfinite([point.x, point.y]).all():
            continue
        try:
            schedule = schedule_values(record["typical_intervals"])
            raw_tags = record.get("tags_json")
            tags = json.loads(raw_tags) if isinstance(raw_tags, str) and raw_tags else {}
            if not isinstance(tags, dict):
                continue
        except (ValueError, TypeError):
            continue
        tags.update({key: record[key] for key in ("shop", "amenity", "leisure", "tourism", "historic")
                     if isinstance(record.get(key), str) and record[key]})
        activity_types = osm_activity_types(tags)
        if not any(activity_types.values()):
            continue
        rows.append(dict(osm_id=str(record["osm_id"]), x=point.x, y=point.y,
                         types=activity_types, **schedule))
    donors = pd.DataFrame(rows, columns=["osm_id", "x", "y", "types", "open_intervals", "status", "opens_at", "closes_at", "closes_day_offset"])
    if donors.osm_id.duplicated().any():
        raise ValueError("OSM business IDs must be unique")
    donors = donors.sort_values("osm_id", kind="stable").reset_index(drop=True)
    for index, record in donors.iterrows():
        for purpose, kinds in record.types.items():
            if kinds:
                for kind in kinds | {purpose}:
                    members.setdefault((purpose, kind), []).append(index)
    return donors, members


def assign_pool(requests, pool, broad, options):
    coordinates = pool[["x", "y"]].to_numpy()
    donor_rows = list(pool.itertuples(index=False))
    requested = np.asarray([point for _, point in requests])
    tree = cKDTree(coordinates)
    # A bounded query plus explicit sorting makes equal-distance choices stable.
    nearby = tree.query_ball_point(requested, options["match_radius_m"])
    unresolved = []
    for (record, point), indices in zip(requests, nearby):
        if indices:
            chosen = min(indices, key=lambda i: (float(np.linalg.norm(coordinates[i]-point)), donor_rows[i].osm_id))
            donor = donor_rows[chosen]
            distance = float(np.linalg.norm(coordinates[chosen]-point))
            copy_schedule(record, donor, "osm_nearby_purpose" if broad else "osm_nearby_type",
                          distance, options["match_radius_m"], 1, 1)
        else:
            unresolved.append((record, point))
    pool = pool[pool.status.eq("open")].reset_index(drop=True)
    if not unresolved or pool.empty:
        return
    donor_rows = list(pool.itertuples(index=False))
    signatures = pool.open_intervals.to_numpy()
    tree = cKDTree(pool[["x", "y"]].to_numpy())
    k = min(options["max_neighbors"], len(pool))
    distances, indices = tree.query(np.asarray([point for _, point in unresolved]), k=k,
                                    distance_upper_bound=options["max_radius_m"] + 1e-8)
    distances, indices = np.asarray(distances).reshape(len(unresolved), k), np.asarray(indices).reshape(len(unresolved), k)
    scope = "purpose" if broad else "type"
    for (record, _), ds, ids in zip(unresolved, distances, indices):
        valid = np.isfinite(ds)
        ds, ids = ds[valid], ids[valid]
        if not len(ids):
            continue
        mask = ds <= options["local_radius_m"]
        if mask.sum() >= options["min_donors"]:
            source, radius = "local_" + scope + "_mode", options["local_radius_m"]
            ds, ids = ds[mask], ids[mask]
        else:
            source = ("regional_" if len(ids) >= options["min_donors"] else "sparse_") + scope + "_mode"
            radius = options["max_radius_m"]
        counts = Counter(signatures[ids])
        closest = {}
        for distance, index in zip(ds, ids):
            donor = donor_rows[index]
            signature = donor.open_intervals
            candidate = (float(distance), donor.osm_id, int(index))
            closest[signature] = min(closest.get(signature, candidate), candidate)
        signature = min(counts, key=lambda key: (-counts[key], closest[key]))
        distance, _, chosen = closest[signature]
        copy_schedule(record, donor_rows[chosen], source, distance, radius, len(ids), counts[signature])


def copy_schedule(record, donor, source, distance, radius, count, support):
    record.update({field: getattr(donor, field) for field in ("open_intervals", "status", "opens_at", "closes_at", "closes_day_offset")})
    record.update(source=source, donor_osm_id=donor.osm_id, distance_m=distance, radius_m=radius,
                  donor_count=int(count), support_count=int(support), support_fraction=float(support/count))


def validate(context):
    # Imported taxonomy changes must invalidate this enrichment cache too.
    return hashlib.sha256(Path(activity_purposes.__file__).read_bytes()).hexdigest()
