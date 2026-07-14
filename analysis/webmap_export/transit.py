"""PT static assets from output_transitSchedule.xml[.gz] and events_extras counts.
Schedule coords are EPSG:2056 (LV95); stop-to-gemeinde/canton mapping uses hot_polygons.
"""

from __future__ import annotations

import gzip
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import pyarrow as pa

from .static_assets import GEOJSON_CT, JSON_CT, put_asset

log = logging.getLogger(__name__)


def parse_transit_schedule(path: Path) -> tuple[dict, dict]:
    """Parse the transit schedule into (stops, lines) dicts."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            root = ET.parse(f).getroot()
    else:
        root = ET.parse(str(path)).getroot()

    stops: dict[str, dict] = {}
    ts = root.find("transitStops")
    if ts is not None:
        for sf in ts.findall("stopFacility"):
            try:
                stops[sf.attrib["id"]] = {
                    "x": float(sf.attrib["x"]),
                    "y": float(sf.attrib["y"]),
                    "name": sf.attrib.get("name", ""),
                }
            except (KeyError, ValueError):
                continue

    lines: dict[str, dict] = {}
    for line in root.findall("transitLine"):
        lid = line.attrib["id"]
        info = lines.setdefault(
            lid, {"name": line.attrib.get("name") or lid, "modes": set(), "stops": []}
        )
        seen = set(info["stops"])
        for route in line.findall("transitRoute"):
            mode = route.findtext("transportMode")
            if mode:
                info["modes"].add(mode)
            rp = route.find("routeProfile")
            if rp is not None:
                for st in rp.findall("stop"):
                    ref = st.attrib.get("refId")
                    if ref and ref not in seen:
                        seen.add(ref)
                        info["stops"].append(ref)
    return stops, lines


def _facility_geo(db: duckdb.DuckDBPyConnection, stops: dict) -> dict:
    """Map facility_id to {bfs, gemeinde, canton_id} via swisstopo polygons."""
    if not stops:
        return {}
    fids = list(stops.keys())
    tbl = pa.table({
        "fid": pa.array(fids, type=pa.string()),
        "x": pa.array([stops[f]["x"] for f in fids], type=pa.float64()),
        "y": pa.array([stops[f]["y"] for f in fids], type=pa.float64()),
    })
    db.register("_tmp_stops", tbl)
    rows = db.execute("""
        SELECT s.fid,
            (SELECT CAST(SPLIT_PART(g.polygon_id, ':', 2) AS INTEGER)
               FROM hot_polygons g
              WHERE g.polygon_type = 'gemeinde'
                AND ST_Contains(g.polygon_geom, ST_Point(s.x, s.y)) LIMIT 1) AS bfs,
            (SELECT g.polygon_name
               FROM hot_polygons g
              WHERE g.polygon_type = 'gemeinde'
                AND ST_Contains(g.polygon_geom, ST_Point(s.x, s.y)) LIMIT 1) AS gemeinde,
            (SELECT CAST(SPLIT_PART(c.polygon_id, ':', 2) AS INTEGER)
               FROM hot_polygons c
              WHERE c.polygon_type = 'canton'
                AND ST_Contains(c.polygon_geom, ST_Point(s.x, s.y)) LIMIT 1) AS canton
        FROM _tmp_stops s
    """).fetchall()
    db.unregister("_tmp_stops")
    return {fid: {"bfs": bfs, "gemeinde": gem, "canton_id": canton}
            for (fid, bfs, gem, canton) in rows}


def build_stop_municipality(db: duckdb.DuckDBPyConnection, stops: dict, geo: dict) -> bool:
    if not stops:
        log.info("transit: no stops - stop_municipality.json skipped")
        return False
    payload = json.dumps(
        {fid: {"name": stops[fid]["name"], **geo.get(fid, {"bfs": None, "gemeinde": None, "canton_id": None})}
         for fid in stops},
        ensure_ascii=False,
    ).encode("utf-8")
    put_asset(db, "stop_municipality", JSON_CT, payload)
    log.info("transit: stop_municipality - %d stops", len(stops))
    return True


def _scale_factor(sample_rate: float | None) -> float:
    """Return 1/sample_rate, or 1.0 when the rate is unknown/invalid."""
    if sample_rate and sample_rate > 0:
        return 1.0 / sample_rate
    return 1.0


def _sc(n: float, factor: float) -> int:
    return int(round(n * factor))


def build_boarding_data_by_line(
    db: duckdb.DuckDBPyConnection, stops: dict, lines: dict, geo: dict, board_acc: dict,
    sample_rate: float | None = None, scale_pt: bool = True,
) -> bool:
    if not lines:
        log.info("transit: no transit lines - boarding_data_by_line.json skipped")
        return False
    f = _scale_factor(sample_rate) if scale_pt else 1.0
    by_line_fac: dict[tuple[str, str], dict[int, list]] = {}
    for (line_id, fac, hour), ba in board_acc.items():
        by_line_fac.setdefault((line_id, fac), {})[hour] = ba

    out = []
    for line_id, info in lines.items():
        cantons = sorted({geo.get(f2, {}).get("canton_id") for f2 in info["stops"]} - {None})
        stops_out = []
        for fac in info["stops"]:
            hours = by_line_fac.get((line_id, fac), {})
            data = [
                {"hour": h, "boardings": _sc(hours[h][0], f), "alightings": _sc(hours[h][1], f)}
                for h in sorted(hours)
            ]
            g = geo.get(fac, {})
            stops_out.append({
                "stop_id": fac,
                "name": stops.get(fac, {}).get("name", ""),
                "bfs": g.get("bfs"),
                "canton_id": g.get("canton_id"),
                "data": data,
            })
        out.append({
            "line_id": line_id,
            "line_name": info["name"],
            "modes": sorted(info["modes"]),
            "cantons": cantons,
            "stops": stops_out,
        })
    payload = json.dumps(out, ensure_ascii=False).encode("utf-8")
    put_asset(db, "boarding_data_by_line", JSON_CT, payload)
    log.info("transit: boarding_data_by_line - %d lines", len(out))
    return True


def build_stop_transfer_data_by_canton(
    db: duckdb.DuckDBPyConnection, stops: dict, lines: dict, geo: dict,
    transfer_data: dict, sample_rate: float | None = None, scale_pt: bool = True,
) -> bool:
    """Emit PT transfer volumes grouped by canton with per-stop transfer matrices.

    Counts are scaled to full population (1/sample_rate); stop_id is the full
    MATSim facility id, matching boarding_data_by_line / stop_municipality.
    """
    if not transfer_data:
        log.info("transit: no transfers detected - stop_transfer_data_by_canton skipped")
        return False
    f = _scale_factor(sample_rate) if scale_pt else 1.0
    by_canton: dict[int, dict] = {}
    for fac, d in transfer_data.items():
        g = geo.get(fac, {})
        canton = g.get("canton_id")
        if canton is None:
            continue
        line_transfers = {
            a: {b: _sc(n, f) for b, n in tos.items()}
            for a, tos in d["lines"].items()
        }
        stop_transfers = {z: _sc(n, f) for z, n in d["dests"].items()}
        t_in = _sc(d["in"], f)
        t_out = _sc(d["out"], f)
        entry = by_canton.setdefault(
            int(canton), {"canton_id": int(canton), "total_transfers": 0, "stops": []}
        )
        entry["total_transfers"] += t_in
        entry["stops"].append({
            "stop_id": fac,
            "name": stops.get(fac, {}).get("name", ""),
            "bfs": g.get("bfs"),
            "transfers": t_in,
            "total_transfers_in": t_in,
            "total_transfers_out": t_out,
            "line_transfers": line_transfers,
            "stop_transfers": stop_transfers,
        })
    out = []
    for canton in sorted(by_canton):
        e = by_canton[canton]
        e["stops"].sort(key=lambda s: s["transfers"], reverse=True)
        out.append(e)
    payload = json.dumps(out, ensure_ascii=False).encode("utf-8")
    put_asset(db, "stop_transfer_data_by_canton", JSON_CT, payload)
    log.info("transit: stop_transfer_data_by_canton - %d cantons, %d transfer stops (scaled ×%.2f)",
             len(out), len(transfer_data), f)
    return True


def parse_transit_routes(path: Path) -> list[dict]:
    """Parse the schedule into one record per transitRoute."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            root = ET.parse(fh).getroot()
    else:
        root = ET.parse(str(path)).getroot()
    routes: list[dict] = []
    for line in root.findall("transitLine"):
        lid = line.attrib["id"]
        lname = line.attrib.get("name") or lid
        for route in line.findall("transitRoute"):
            rid = route.attrib.get("id", "")
            mode = route.findtext("transportMode") or ""
            link_refs = []
            r = route.find("route")
            if r is not None:
                link_refs = [lk.attrib["refId"] for lk in r.findall("link") if "refId" in lk.attrib]
            stop_refs = []
            rp = route.find("routeProfile")
            if rp is not None:
                stop_refs = [s.attrib["refId"] for s in rp.findall("stop") if "refId" in s.attrib]
            routes.append({"line_id": lid, "line_name": lname, "route_id": rid,
                           "mode": mode, "link_refs": link_refs, "stop_refs": stop_refs})
    return routes


def build_transit_routes(db: duckdb.DuckDBPyConnection, schedule_xml: Path,
                         stops: dict) -> bool:
    """Emit transit_routes GeoJSON: one WGS84 LineString per transitRoute.

    Geometry comes from ordered network links (stop coords as fallback);
    identical (line_id, link-sequence) routes are de-duped.
    """
    routes = parse_transit_routes(schedule_xml)
    if not routes:
        log.info("transit: no routes - transit_routes skipped")
        return False

    needed = {lk for r in routes for lk in r["link_refs"]}
    link_coords: dict[str, list] = {}
    if needed:
        db.execute("CREATE TEMP TABLE _rt_links (link_id VARCHAR)")
        db.executemany("INSERT INTO _rt_links VALUES (?)", [(x,) for x in needed])
        rows = db.execute("""
            SELECT l.link_id,
                   ST_AsGeoJSON(ST_Transform(n.geom, 'EPSG:2056', 'EPSG:4326', true))
            FROM _rt_links l JOIN network_links n USING (link_id)
            WHERE n.geom IS NOT NULL
        """).fetchall()
        db.execute("DROP TABLE _rt_links")
        for lid, gj in rows:
            try:
                link_coords[lid] = json.loads(gj)["coordinates"]
            except (TypeError, KeyError, ValueError):
                continue

    stop_lonlat: dict[str, list] = {}
    if stops:
        db.execute("CREATE TEMP TABLE _rt_stops (fid VARCHAR, x DOUBLE, y DOUBLE)")
        db.executemany("INSERT INTO _rt_stops VALUES (?,?,?)",
                       [(fid, s["x"], s["y"]) for fid, s in stops.items()])
        for fid, gj in db.execute("""
            SELECT fid, ST_AsGeoJSON(ST_Transform(ST_Point(x, y), 'EPSG:2056', 'EPSG:4326', true))
            FROM _rt_stops
        """).fetchall():
            try:
                stop_lonlat[fid] = json.loads(gj)["coordinates"]
            except (TypeError, KeyError, ValueError):
                continue
        db.execute("DROP TABLE _rt_stops")

    def _assemble(link_refs: list, stop_refs: list) -> list:
        coords: list = []
        for lk in link_refs:
            seg = link_coords.get(lk)
            if not seg:
                continue
            for pt in seg:
                if not coords or coords[-1] != pt:
                    coords.append(pt)
        if len(coords) < 2:
            coords = [stop_lonlat[s] for s in stop_refs if s in stop_lonlat]
        return coords

    features = []
    seen = set()
    for r in routes:
        dedup_key = (r["line_id"], tuple(r["link_refs"]))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        coords = _assemble(r["link_refs"], r["stop_refs"])
        if len(coords) < 2:
            continue
        features.append({
            "type": "Feature",
            "properties": {"line_id": r["line_id"], "route_id": r["route_id"],
                           "line_name": r["line_name"], "mode": r["mode"]},
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    payload = json.dumps({"type": "FeatureCollection", "features": features},
                         ensure_ascii=False).encode("utf-8")
    put_asset(db, "transit_routes", GEOJSON_CT, payload)
    log.info("transit: transit_routes - %d route features (%.1f MB)",
             len(features), len(payload) / 1e6)
    return True


def build_all(
    db: duckdb.DuckDBPyConnection, schedule_xml: Path,
    board_acc: dict, transfer_data: dict | None = None,
    sample_rate: float | None = None, scale_pt: bool = True,
) -> None:
    """Parse the schedule and emit all PT static assets.

    When scale_pt is True, passenger counts are scaled to full population via sample_rate.
    """
    log.info("=== transit assets START (sample_rate=%s, scale_pt=%s)", sample_rate, scale_pt)
    stops, lines = parse_transit_schedule(schedule_xml)
    geo = _facility_geo(db, stops)
    build_stop_municipality(db, stops, geo)
    build_boarding_data_by_line(db, stops, lines, geo, board_acc or {}, sample_rate, scale_pt)
    build_stop_transfer_data_by_canton(db, stops, lines, geo, transfer_data or {}, sample_rate, scale_pt)
    build_transit_routes(db, schedule_xml, stops)
    log.info("=== transit assets DONE")
