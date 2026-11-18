"""Streaming XML/geometry import and provenance-based cleaning classification."""
import json
from contextlib import closing
import logging
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd
from lxml import etree
from pyproj import Transformer
import shapely
from shapely.geometry import LineString
import xopen

logger = logging.getLogger(__name__)
UNPROJECT = Transformer.from_crs(2056, 4326, always_xy=True)

SCHEMA = """
CREATE TABLE nodes(network TEXT,id TEXT,x REAL,y REAL,PRIMARY KEY(network,id)) WITHOUT ROWID;
CREATE TABLE links(k INTEGER PRIMARY KEY,network TEXT,id TEXT,road TEXT,status TEXT DEFAULT 'retained',
 from_id TEXT,to_id TEXT,record TEXT,geom BLOB,fine TEXT,coarse TEXT,minx REAL,maxx REAL,miny REAL,maxy REAL,
 UNIQUE(network,id));
CREATE TABLE members(before_id TEXT,after_key INTEGER,PRIMARY KEY(before_id,after_key)) WITHOUT ROWID;
CREATE TABLE ways(way TEXT,k INTEGER,PRIMARY KEY(way,k)) WITHOUT ROWID;
CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT);
"""


def attr_value(record, name, default=""):
    return record.get("attributes", {}).get(name, {}).get("value", default)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def polyline(points):
    """Google encoded polyline at 1e-6 degrees, in latitude/longitude order."""
    previous = [0, 0]
    result = []
    for lon, lat in points:
        for axis, value in enumerate((lat, lon)):
            current = round(value * 1e6)
            delta = current - previous[axis]
            previous[axis] = current
            encoded = ~(delta << 1) if delta < 0 else delta << 1
            while encoded >= 32:
                result.append(chr((32 | (encoded & 31)) + 63))
                encoded >>= 5
            result.append(chr(encoded + 63))
    return "".join(result)


def geometry_values(shape):
    if not isinstance(shape, LineString) or shape.is_empty or len(shape.coords) < 2:
        return None
    if not np.isfinite(shapely.get_coordinates(shape)).all():
        return None
    def encode(tolerance):
        xy = np.asarray(shape.simplify(tolerance, preserve_topology=False).coords)[:, :2]
        lon, lat = UNPROJECT.transform(xy[:, 0], xy[:, 1])
        return polyline(zip(lon, lat))
    xmin, ymin, xmax, ymax = shape.bounds
    return (sqlite3.Binary(shapely.to_wkb(shape, output_dimension=2)), encode(1.5), encode(40), xmin, xmax, ymin, ymax)


def import_xml(connection, path, network):
    pending_nodes, pending_links, pending_members, pending_ways = [], [], [], []
    count = 0
    key = connection.execute("SELECT coalesce(max(k),0) FROM links").fetchone()[0]
    def flush():
        connection.executemany("INSERT INTO nodes VALUES(?,?,?,?)", pending_nodes)
        connection.executemany("INSERT INTO links(k,network,id,road,from_id,to_id,record) VALUES(?,?,?,?,?,?,?)", pending_links)
        connection.executemany("INSERT OR IGNORE INTO members VALUES(?,?)", pending_members)
        connection.executemany("INSERT OR IGNORE INTO ways VALUES(?,?)", pending_ways)
        connection.commit()
        pending_nodes.clear(); pending_links.clear(); pending_members.clear(); pending_ways.clear()
    with xopen.xopen(str(path), "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",), tag=("node", "link"),
                                         load_dtd=False, no_network=True, resolve_entities=False):
            if element.tag == "node":
                pending_nodes.append((network, element.get("id"), float(element.get("x")), float(element.get("y"))))
            else:
                key += 1
                record = dict(element.attrib)
                record["attributes"] = {a.get("name"): {"value": "".join(a.itertext()), "class": a.get("class")}
                                        for a in element.findall("attributes/attribute")}
                children = [etree.tostring(child, encoding="unicode") for child in element
                            if isinstance(child.tag, str) and child.tag != "attributes"]
                if children:
                    record["nested_xml"] = children
                road = attr_value(record, "osm:way:highway") or ("railway" if attr_value(record, "osm:way:railway") else "other")
                pending_links.append((key, network, record["id"], road, record["from"], record["to"], json_text(record)))
                if network == "after":
                    provenance = attr_value(record, "old_link_id")
                    # The raw pt2matsim IDs are numeric; split only the recorded link provenance.
                    members = re.split(r"[_,;\s]+", str(provenance)) if provenance else [record["id"]]
                    for member in set(filter(None, members)):
                        pending_members.append((member, key))
                osm_ids = [str(attr_value(record, "osm:way:id")), str(attr_value(record, "old_osm_id"))]
                for way in {v for text in osm_ids for v in re.split(r"[_,;\s]+", text) if v.isdigit()}:
                    pending_ways.append((way, key))
                count += 1
            if len(pending_nodes) + len(pending_links) >= 5000:
                flush()
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
    flush()
    logger.info("Imported %s XML: %s links", network, f"{count:,}")


def import_geometry(connection, path, network):
    count = 0
    sql = "UPDATE links SET geom=?,fine=?,coarse=?,minx=?,maxx=?,miny=?,maxy=? WHERE network=? AND id=?"
    for chunk in pd.read_csv(path, dtype=str, chunksize=10000, keep_default_na=False):
        if not {"LinkId", "Geometry"}.issubset(chunk.columns):
            raise ValueError(f"Geometry CSV needs LinkId and Geometry columns: {path}")
        shapes = shapely.from_wkt(chunk.Geometry.to_numpy(), on_invalid="ignore")
        updates = []
        for identifier, shape in zip(chunk.LinkId, shapes):
            values = geometry_values(shape)
            if values:
                updates.append((*values, network, identifier))
        connection.executemany(sql, updates)
        connection.commit()
        count += len(updates)
        if count % 250000 < 10000:
            logger.info("Imported %s detailed geometries: %s", network, f"{count:,}")
    # Only links missing a valid CSV row fall back to their actual directed endpoints.
    fallback = 0
    query = """SELECT l.k,a.x,a.y,b.x,b.y FROM links l
      JOIN nodes a ON a.network=l.network AND a.id=l.from_id
      JOIN nodes b ON b.network=l.network AND b.id=l.to_id
      WHERE l.network=? AND l.geom IS NULL AND l.k>? ORDER BY l.k LIMIT 5000"""
    last = 0
    while rows := connection.execute(query, (network, last)).fetchall():
        updates = []
        for key, x0, y0, x1, y1 in rows:
            updates.append((*geometry_values(LineString([(x0, y0), (x1, y1)])), key))
        connection.executemany("UPDATE links SET geom=?,fine=?,coarse=?,minx=?,maxx=?,miny=?,maxy=? WHERE k=?", updates)
        connection.commit()
        last = rows[-1][0]
        fallback += len(rows)
    if connection.execute("SELECT count(*) FROM links WHERE network=? AND geom IS NULL", (network,)).fetchone()[0]:
        raise ValueError(f"{network} has links without geometry or valid endpoint nodes")
    logger.info("%s geometry complete: %s endpoint fallbacks", network, fallback)
    return fallback


def classify(connection):
    connection.executescript("""
    DELETE FROM members WHERE NOT EXISTS(SELECT 1 FROM links b WHERE b.network='before' AND b.id=members.before_id);
    CREATE INDEX members_after ON members(after_key);
    -- Retained is the import default. Update only changed outcomes, avoiding
    -- rewrites of millions of large geometry/attribute records that stay retained.
    UPDATE links SET status='added' WHERE network='after'
      AND NOT EXISTS(SELECT 1 FROM members m WHERE m.after_key=links.k);
    UPDATE links SET status='merged' WHERE network='after' AND k IN
      (SELECT after_key FROM members GROUP BY after_key HAVING count(*)>1);
    UPDATE links SET status='removed' WHERE network='before'
      AND NOT EXISTS(SELECT 1 FROM members m WHERE m.before_id=links.id);
    UPDATE links SET status='merged' WHERE network='before' AND EXISTS
      (SELECT 1 FROM members m JOIN links a ON a.k=m.after_key
       WHERE m.before_id=links.id AND a.status='merged');
    CREATE INDEX links_node ON links(network,from_id);
    CREATE INDEX links_id ON links(id);
    CREATE INDEX links_road_status ON links(network,road,status);
    CREATE VIRTUAL TABLE spatial USING rtree(k,minx,maxx,miny,maxy);
    INSERT INTO spatial SELECT k,minx,maxx,miny,maxy FROM links;
    """)
    connection.commit()


def build_database(sources, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(".building.sqlite")
    if temporary.exists():
        # Preserve an interrupted diagnostic build for inspection.
        temporary.rename(destination.with_name(f"interrupted-{time.time_ns()}.sqlite"))
    try:
        with closing(sqlite3.connect(temporary, timeout=60)) as connection:
            connection.executescript("PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-131072;" + SCHEMA)
            fallbacks = {}
            for network in ("before", "after"):
                import_xml(connection, sources[network], network)
                fallbacks[network] = import_geometry(connection, sources[network + "_geometry"], network)
            classify(connection)
            counts = [{"network": n, "road": r, "status": s, "count": c} for n, r, s, c in
                      connection.execute("SELECT network,road,status,count(*) FROM links GROUP BY network,road,status")]
            summary = {"counts": counts, "fallbacks": fallbacks, "classification": "Exact link-ID and old_link_id provenance; no OSM-ID or geometric guessing."}
            connection.execute("INSERT INTO metadata VALUES('summary',?)", (json_text(summary),))
            connection.commit()
            connection.execute("PRAGMA optimize")
        temporary.replace(destination)
        return summary
    except Exception:
        logger.exception("Build interrupted; diagnostic database retained at %s", temporary)
        raise
