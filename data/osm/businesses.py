"""Extract simulation-region businesses with parsed typical weekday hours."""

from collections import Counter
from datetime import date, datetime, time, timedelta
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import tempfile
from time import perf_counter
from zoneinfo import ZoneInfo

import geopandas as gpd
import osmium
import pandas as pd
import pyarrow as pa
from opening_hours import OpeningHours, ParserError, State
from shapely import from_wkb, make_valid, box
from data.activity_purposes import SHOP, LEISURE, osm_activity_types


logger = logging.getLogger("synpp: osm.businesses")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
CALENDAR_RULE = re.compile(r"(?<![A-Za-z])(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|PH|SH|easter|week|sunrise|sunset|dawn|dusk)(?![A-Za-z])|\b\d{4}\b")
BUSINESS_KEYS = ("shop", "amenity", "leisure", "tourism", "historic", "office", "craft")
INACTIVE = {"no", "vacant", "closed", "disused", "abandoned", "construction", "proposed"}
LEISURE_AMENITIES = {
    "bar", "pub", "biergarten", "cafe", "restaurant", "fast_food", "food_court",
    "ice_cream", "cinema", "theatre", "nightclub", "casino", "arts_centre",
}
LEISURE_VENUES = {
    "fitness_centre", "sports_centre", "bowling_alley", "amusement_arcade", "dance",
    "escape_game", "sauna", "swimming_pool", "water_park", "miniature_golf",
    "ice_rink", "trampoline_park", "golf_course",
}
LEISURE_TOURISM = {
    "hotel", "motel", "hostel", "guest_house", "apartment", "camp_site", "caravan_site",
    "museum", "gallery", "zoo", "aquarium", "theme_park",
}
OTHER_AMENITIES = {
    "bank", "post_office", "car_rental", "car_wash", "doctors", "dentist",
    "veterinary", "clinic", "driving_school", "money_transfer",
}
BUSINESS_COLUMNS = [
    "osm_id", "osm_type", "source_file", "name", "brand", "operator", "purpose", "is_shopping",
    "is_leisure", "offers_shop", "offers_leisure", "classification_tags", *BUSINESS_KEYS, "opening_hours",
    "opening_hours_status", "opening_hours_error", "access", "tags_json",
    "lon", "lat", "x", "y", "opens_at", "closes_at", "closes_day_offset",
    "typical_status", "typical_intervals", "typical_open_minutes",
    "typical_weekday_count", "typical_weekdays", "typical_occurrences", "geometry",
]
TYPICAL_COLUMNS = ["opens_at", "closes_at", "closes_day_offset", "typical_status",
                   "typical_intervals", "typical_open_minutes", "typical_weekday_count", "typical_weekdays", "typical_occurrences"]
HOURS_COLUMNS = [
    "osm_id", "weekday", "timezone", "status", "open_intervals",
    "unknown_intervals", "known_open_minutes", "unknown_minutes", "open_minutes", "session_intervals",
    "open_days", "evaluated_days", "schedule_occurrences",
]


def configure(context):
    context.config("data_path")
    context.config("latest_osm_for_correction", default=None)
    context.config("osm_file", default="switzerland-latest.osm.gz")
    context.config("data.osm.businesses.calendar_year", default=2026)
    context.stage("data.spatial.simulation_region")


def execute(context):

    # Select configured extracts, replacing the Swiss source when requested.

    sources = select_osm_files(context.config("data_path"),
                              context.config("latest_osm_for_correction"), context.config("osm_file"))
    year = calendar_year(context.config("data.osm.businesses.calendar_year"))
    region = context.stage("data.spatial.simulation_region")
    if region is None or region.is_empty or not region.is_valid:
        raise ValueError("The simulation region must be a non-empty valid geometry in EPSG:2056")
    region_wgs84 = gpd.GeoSeries([region], crs=2056).to_crs(4326).iloc[0]
    output = Path(context.path())

    # Skip places without hours before constructing geometry or serializing tags.

    started = perf_counter()
    frames, processed, skipped, failures = [], [], [], 0
    for source in sources:
        if not extract_overlaps_region(source, region_wgs84):
            logger.info("Skipping OSM extract outside the simulation region: %s", source.name)
            skipped.append(str(source))
            continue
        logger.info("Extracting businesses from %s", source.name)
        candidates, invalid = read_businesses(source, output)
        candidates = candidates[candidates.geometry.intersects(region)].copy()
        candidates["source_file"] = str(source)
        frames.append(candidates)
        processed.append(str(source))
        failures += invalid
    businesses = (gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=2056) if frames
                  else gpd.GeoDataFrame(columns=BUSINESS_COLUMNS, geometry="geometry", crs=2056))
    duplicate_count = int(businesses.osm_id.duplicated().sum())
    businesses = businesses.drop_duplicates("osm_id", keep="first").reset_index(drop=True)
    businesses["x"], businesses["y"] = businesses.geometry.x, businesses.geometry.y
    locations = businesses.to_crs(4326)
    businesses["lon"], businesses["lat"] = locations.geometry.x, locations.geometry.y
    extraction_seconds = perf_counter() - started
    logger.info("Extracted %d simulation-region candidates from %d files in %.1f seconds; removed %d duplicates",
                len(businesses), len(processed), extraction_seconds, duplicate_count)

    # Reuse repeated schedules and retain only businesses with fully parsed hours.

    started = perf_counter()
    _weekday_hours.cache_clear()
    _calendar_hours.cache_clear()
    records, summaries, retained = [], [], []
    statuses = Counter()
    for row in businesses.itertuples():
        hours, typical, status, error = business_hours(row.opening_hours, year, row.lat, row.lon)
        statuses[status] += 1
        if status != "parsed":
            continue
        retained.append(row.Index)
        summaries.append(typical)
        records.extend({"osm_id": row.osm_id, **day} for day in hours)
    businesses = businesses.loc[retained].reset_index(drop=True)
    summary = pd.DataFrame(summaries, columns=TYPICAL_COLUMNS)
    for column in TYPICAL_COLUMNS:
        businesses[column] = summary[column]
    for column in ("opens_at", "closes_at"):
        businesses[column] = pd.array(businesses[column], dtype=pd.ArrowDtype(pa.time32("s")))
    for column in ("closes_day_offset", "typical_open_minutes", "typical_weekday_count", "typical_occurrences"):
        businesses[column] = pd.array(businesses[column], dtype="Int64")
    businesses["opening_hours_status"], businesses["opening_hours_error"] = "parsed", ""
    businesses = businesses.reindex(columns=BUSINESS_COLUMNS)
    hours = pd.DataFrame(records, columns=HOURS_COLUMNS)
    for column in ("known_open_minutes", "unknown_minutes", "open_minutes"):
        hours[column] = pd.to_numeric(hours[column]).astype("Float64")
    for column in ("open_days", "evaluated_days", "schedule_occurrences"):
        hours[column] = pd.array(hours[column], dtype="Int64")

    # Export clean Parquet tables and coverage metadata.

    hours_seconds = perf_counter() - started
    business_path, hours_path = output / "businesses.parquet", output / "weekday_hours.parquet"
    businesses.to_parquet(business_path, index=False)
    hours.to_parquet(hours_path, index=False)
    metadata = {
        "sources": [str(source) for source in sources], "processed_sources": processed,
        "skipped_sources": skipped, "duplicate_osm_objects": duplicate_count,
        "duplicate_policy": "Swiss source first, then configured order; first OSM ID wins",
        "calendar_year": year, "timezone": "Europe/Zurich",
        "crs": "EPSG:2056", "holiday_calendar": "Country inferred from business coordinates for PH rules; no canton/school calendar",
        "businesses": len(businesses), "invalid_geometries": failures,
        "purposes": businesses.purpose.value_counts().to_dict(),
        "opening_hours_status": {"parsed": len(businesses)},
        "candidates_with_opening_hours": sum(statuses.values()),
        "excluded_hours_status": {key: value for key, value in statuses.items() if key != "parsed"},
        "typical_hours": "Most frequent open weekday session schedule across all seasons; closed days excluded; ties prefer Monday-to-Friday order",
        "extraction_seconds": extraction_seconds, "hours_seconds": hours_seconds,
        "schedule_cache": _weekday_hours.cache_info()._asdict(),
        "calendar_cache": _calendar_hours.cache_info()._asdict(),
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Simulation-region businesses with parsed hours: %d; excluded %s; schedule evaluation %.1f seconds; cache %s",
                len(businesses), metadata["excluded_hours_status"], hours_seconds, metadata["schedule_cache"])

    # Return file paths so downstream consumers can load only the tables they need.

    return {"businesses": str(business_path), "weekday_hours": str(hours_path), "metadata": str(metadata_path)}


def select_osm_files(data_path, latest, configured):
    names = list(configured) if isinstance(configured, list) else [configured]
    if not names or any(not isinstance(name, (str, Path)) or not str(name).strip() for name in names):
        raise ValueError("osm_file must specify an OSM filename or a non-empty list of filenames")
    swiss = [name for name in names if "switzerland" in Path(name).name.lower()]
    if len(swiss) > 1:
        raise ValueError("osm_file must contain at most one Swiss extract")
    # Prefer the Swiss source when extracts contain overlapping OSM objects.
    others = [name for name in names if name not in swiss]
    names = ([latest] if latest else swiss) + others
    paths = []
    for name in names:
        if not isinstance(name, (str, Path)) or not str(name).strip():
            raise ValueError("latest_osm_for_correction must be an OSM filename")
        path = Path(name).expanduser()
        path = path if path.is_absolute() else Path(data_path) / "osm" / path
        if not path.is_file():
            raise FileNotFoundError(path)
        path = path.resolve()
        if path not in paths:
            paths.append(path)
    return paths


def extract_overlaps_region(path, region_wgs84):
    # Header-only reads avoid scanning distant extracts. Missing bounds are not
    # evidence of non-overlap: scan those files and apply the exact point filter.
    with osmium.io.Reader(str(path), osmium.osm.NOTHING) as reader:
        bounds = reader.header().box()
        if not bounds.valid():
            return True
        footprint = box(bounds.bottom_left.lon, bounds.bottom_left.lat,
                        bounds.top_right.lon, bounds.top_right.lat)
    return footprint.intersects(region_wgs84)


def calendar_year(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1901 <= value <= 9998:
        raise ValueError("data.osm.businesses.calendar_year must be an integer from 1901 to 9998")
    return value


def classify_business(tags):
    if any(tags.get(key) == "yes" for key in ("disused", "abandoned", "construction")):
        return None
    values = {key: {v.strip() for v in tags.get(key, "").split(";")} - INACTIVE - {""}
              for key in BUSINESS_KEYS}
    activity_types = osm_activity_types(tags)
    shopping, leisure = bool(activity_types[SHOP]), bool(activity_types[LEISURE])
    other = bool(values["office"] or values["craft"] or values["amenity"] & OTHER_AMENITIES
                 or values["shop"] or values["tourism"] & LEISURE_TOURISM
                 or values["leisure"] & LEISURE_VENUES)
    if not (shopping or leisure or other):
        return None
    purpose = "mixed" if shopping and leisure else "shopping" if shopping else "leisure" if leisure else "other"
    return purpose, shopping, leisure


def read_businesses(path, cache_directory):
    factory, rows, failures = osmium.geom.WKBFactory(), {}, 0
    with tempfile.TemporaryDirectory(dir=cache_directory, prefix="business_nodes_") as temporary:
        processor = (osmium.FileProcessor(str(path))
                     .with_locations(f"sparse_file_array,{temporary}/nodes")
                     .with_areas(osmium.filter.KeyFilter("opening_hours"))
                     .with_filter(osmium.filter.KeyFilter(*BUSINESS_KEYS))
                     .with_filter(osmium.filter.KeyFilter("opening_hours")))
        for obj in processor:
            tags = dict(obj.tags)
            if not tags.get("opening_hours", "").strip():
                continue
            classification = classify_business(tags)
            if classification is None or obj.is_relation():
                continue
            try:
                if obj.is_node():
                    shape, kind, identifier = from_wkb(factory.create_point(obj)), "node", f"n{obj.id}"
                elif obj.is_area():
                    shape = make_valid(from_wkb(factory.create_multipolygon(obj)))
                    kind = "way" if obj.from_way() else "relation"
                    identifier = ("w" if obj.from_way() else "r") + str(obj.orig_id())
                elif obj.is_way():
                    if obj.is_closed():
                        continue  # Process polygon ways through their assembled areas only.
                    shape, kind, identifier = from_wkb(factory.create_linestring(obj)), "way", f"w{obj.id}"
                else:
                    continue
                if shape.is_empty:
                    failures += 1
                    continue
                purpose, shopping, leisure = classification
                rows[identifier] = {
                    "osm_id": identifier, "osm_type": kind,
                    **{key: tags.get(key, "") for key in ("name", "brand", "operator", *BUSINESS_KEYS, "opening_hours", "access")},
                    "purpose": purpose, "is_shopping": shopping, "is_leisure": leisure,
                    "offers_shop": shopping, "offers_leisure": leisure,
                    "classification_tags": json.dumps({k: tags[k] for k in BUSINESS_KEYS if k in tags}),
                    "tags_json": json.dumps(tags, ensure_ascii=False), "geometry": shape,
                }
            except (RuntimeError, osmium.InvalidLocationError, ValueError):
                failures += 1
    features = gpd.GeoDataFrame(list(rows.values()), columns=[c for c in BUSINESS_COLUMNS
                                if c not in (*TYPICAL_COLUMNS, "lon", "lat", "x", "y", "opening_hours_status", "opening_hours_error")],
                                geometry="geometry", crs=4326).to_crs(2056)
    # A point on the business footprint is safer than a polygon centroid outside a concave building.
    features.geometry = features.geometry.representative_point()
    return features, failures


def business_hours(expression, year, lat, lon):
    """Return typical opening hours considering all weekdays and all seasons."""
    expression = expression.strip()
    if not re.search(r"\b(?:sunrise|sunset|dawn|dusk|PH)\b", expression):
        lat = lon = None
    profiles, typical, status, error = _calendar_hours(expression, year, lat, lon)
    return [row.copy() for row in profiles], typical.copy(), status, error


@lru_cache(maxsize=16384)
def _calendar_hours(expression, year, lat, lon):
    first, stop = date(year, 1, 1), date(year+1, 1, 1)
    unquoted = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', "", expression)
    if not CALENDAR_RULE.search(unquoted):
        # Ordinary weekday/time rules repeat all year: evaluate once, then
        # weight each weekday by its actual number of occurrences in the year.
        monday = first + timedelta(days=(-first.weekday()) % 7)
        rows, status, error = _weekday_hours(expression, monday, lat, lon)
        if status != "parsed":
            return [], {}, status, error
        days = Counter((first + timedelta(days=i)).weekday() for i in range((stop-first).days))
        weights = [days[i] for i in range(5)]
    else:
        rules, status, error = parse_hours_rules(expression, lat, lon)
        if rules is None:
            return [], {}, status, error
        timezone = ZoneInfo("Europe/Zurich")
        start = datetime.combine(first, time(), timezone)
        end = datetime.combine(stop, time(), timezone)
        try:
            spans = opening_spans(rules, start-timedelta(days=2), end+timedelta(days=2))
        except (ValueError, RuntimeError) as exception:
            return [], {}, "invalid", str(exception)
        rows, cursor = [], 0
        for offset in range((stop-first).days):
            day = first + timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            left, right = start+timedelta(days=offset), start+timedelta(days=offset+1)
            while cursor < len(spans) and spans[cursor][1] <= left:
                cursor += 1
            limit = cursor
            while limit < len(spans) and spans[limit][0] < right:
                limit += 1
            row = day_hours(day, left, right, spans[cursor:limit])
            if row["unknown_minutes"]:
                return [], {}, "unknown", "Uncertain weekday hours in the evaluation calendar"
            rows.append(row)
        weights = [1] * len(rows)
    typical = typical_weekday_hours(rows, weights)
    profiles = []
    for weekday in WEEKDAYS:
        group = [(row, weight) for row, weight in zip(rows, weights) if row["weekday"] == weekday]
        selected = typical_weekday_hours([row for row, _ in group], [weight for _, weight in group])
        use_sessions = any(json.loads(row["session_intervals"]) for row, _ in group)
        field = "session_intervals" if use_sessions else "open_intervals"
        schedule = json.loads(selected["typical_intervals"])
        representative = next(row for row, _ in group if json.loads(row[field]) == schedule).copy()
        representative.pop("date")
        representative.update(
            open_days=sum(weight for row, weight in group if row["open_minutes"] > 0),
            evaluated_days=sum(weight for _, weight in group),
            schedule_occurrences=selected["typical_occurrences"],
        )
        profiles.append(representative)
    return profiles, typical, "parsed", ""


def weekday_hours(expression, monday, lat, lon):
    """Cache weekly rules; solar and public-holiday rules retain coordinates."""
    expression = expression.strip()
    if not re.search(r"\b(?:sunrise|sunset|dawn|dusk|PH)\b", expression):
        lat = lon = None
    rows, status, error = _weekday_hours(expression, monday, lat, lon)
    # Cached rows contain only immutable values, so shallow copies isolate callers.
    return [row.copy() for row in rows], status, error


@lru_cache(maxsize=16384)
def _weekday_hours(expression, monday, lat, lon):
    timezone = ZoneInfo("Europe/Zurich")
    rules, status, error = parse_hours_rules(expression, lat, lon)
    anchor = datetime.combine(monday, time(), timezone)
    spans = []
    try:
        if rules is not None:
            spans = opening_spans(rules, anchor-timedelta(days=1), anchor+timedelta(days=6))
    except (ValueError, RuntimeError) as exception:
        status, error = "invalid", str(exception)
    rows = [day_hours(monday+timedelta(days=offset), anchor+timedelta(days=offset),
                      anchor+timedelta(days=offset+1), spans, status) for offset in range(5)]
    if status == "parsed" and any(row["unknown_minutes"] for row in rows):
        status = "unknown"
    return rows, status, error


def parse_hours_rules(expression, lat, lon):
    timezone = ZoneInfo("Europe/Zurich")
    status, error, rules = "parsed", "", None
    # Alternative/fallback expressions can describe different services (e.g. an
    # ATM versus a staffed bank), so do not collapse them into one firm schedule.
    unquoted = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', "", expression)
    uncertain = (bool(re.fullmatch(r'\s*".*"\s*', expression))
                 or "||" in unquoted
                 or bool(re.search(r"\bSH\b|\d{1,2}:\d{2}\s*\+", unquoted)))
    if not expression:
        status = "missing"
    elif uncertain:
        status, error = "unsupported", "Comment-only, fallback, open-ended, or school-holiday rule"
    else:
        try:
            rules = OpeningHours(expression, timezone=timezone, country="CH" if lat is None else None,
                                 coords=None if lat is None else (lat, lon),
                                 auto_country=lat is not None, auto_timezone=False)
        except (ParserError, SyntaxError, ValueError) as exception:
            status, error = "invalid", str(exception)
    return rules, status, error


def opening_spans(rules, start, end):
    """Merge continuous states once, including overnight and all-day openings."""
    spans = []
    for left, right, state, comment in rules.intervals(start, end):
        if re.search(r"appointment|rendez[- ]vous|vereinbarung", comment, re.I):
            state = State.UNKNOWN
        if spans and spans[-1][1] == left and spans[-1][2] == state:
            spans[-1] = (spans[-1][0], right, state)
        else:
            spans.append((left, right, state))
    return spans


def day_hours(day, start, end, spans, status="parsed"):
    opened, unknown, sessions = [], [], []
    if status != "parsed":
        unknown = [[0, 1440]]
    for left, right, state in spans:
        if left >= end or right <= start:
            continue
        interval = [int((max(left, start)-start).total_seconds()/60),
                    int((min(right, end)-start).total_seconds()/60)]
        if interval[1] <= interval[0]:
            continue
        if state == State.UNKNOWN:
            unknown.append(interval)
        elif state == State.OPEN:
            opened.append(interval)
            if left < start:
                if right >= end:
                    sessions.append([0, 1440])
            else:
                closing = end if right-left > timedelta(days=1) else right
                sessions.append([int((left-start).total_seconds()/60), int((closing-start).total_seconds()/60)])
    open_minutes = sum(b-a for a, b in opened)
    unknown_minutes = sum(b-a for a, b in unknown)
    daily_status = status if status != "parsed" else "unknown" if unknown else "open" if opened else "closed"
    return {"weekday": day.strftime("%A"), "date": day.isoformat(), "timezone": "Europe/Zurich",
            "status": daily_status, "open_intervals": json.dumps(opened),
            "unknown_intervals": json.dumps(unknown), "known_open_minutes": open_minutes,
            "unknown_minutes": unknown_minutes, "open_minutes": None if unknown else open_minutes,
            "session_intervals": json.dumps(sessions)}


def typical_weekday_hours(rows, weights=None):
    """Choose the modal open-weekday schedule, ignoring closed weekdays."""
    schedules = [tuple(tuple(interval) for interval in json.loads(row["session_intervals"])) for row in rows]
    if not any(schedules):
        # A Sunday-only overnight session can still open during early Monday.
        # If no weekday has a new session, retain that weekday availability.
        schedules = [tuple(tuple(interval) for interval in json.loads(row["open_intervals"])) for row in rows]
    weights = [1] * len(rows) if weights is None else weights
    counts, first_weekday = Counter(), {}
    for row, schedule, weight in zip(rows, schedules, weights):
        if schedule:
            counts[schedule] += weight
            first_weekday[schedule] = min(first_weekday.get(schedule, 5), WEEKDAYS.index(row["weekday"]))
    typical = max(counts, key=lambda schedule: (counts[schedule], -first_weekday[schedule])) if counts else ()
    matching = {row["weekday"] for row, schedule in zip(rows, schedules) if schedule == typical}
    opening, closing = (typical[0][0], typical[-1][1]) if typical else (None, None)
    return {
        "opens_at": time(opening // 60 % 24, opening % 60) if opening is not None else None,
        "closes_at": time(closing // 60 % 24, closing % 60) if closing is not None else None,
        "closes_day_offset": closing // 1440 if closing is not None else None,
        "typical_status": "open" if typical else "closed",
        "typical_intervals": json.dumps(typical),
        "typical_open_minutes": sum(end-start for start, end in typical),
        "typical_weekday_count": len(matching),
        "typical_weekdays": json.dumps([weekday for weekday in WEEKDAYS if weekday in matching]),
        "typical_occurrences": counts[typical] if typical else sum(weights),
    }


def validate(context):
    paths = select_osm_files(context.config("data_path"), context.config("latest_osm_for_correction"), context.config("osm_file"))
    import hashlib
    import data.activity_purposes as classification
    return [[[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in paths],
            hashlib.sha256(Path(classification.__file__).read_bytes()).hexdigest()]
