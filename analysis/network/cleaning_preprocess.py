"""Precomputed cleaning reasons, directional display shapes and review circles."""
from contextlib import closing
import gzip
import json
import math
import logging
from pathlib import Path
import sqlite3

import numpy as np
import shapely

from .cleaning_data import geometry_values, UNPROJECT

OFFSET_METRES = 2.0


def prepare_removals(directory, audit):
    directory, audit = Path(directory), Path(audit).resolve()
    target = directory / "cleaning_audit.sqlite"
    fingerprint = [str(audit), audit.stat().st_size, audit.stat().st_mtime_ns]
    if target.exists():
        with closing(sqlite3.connect(target.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)) as connection:
            summary = json.loads(connection.execute("SELECT value FROM metadata WHERE name='summary'").fetchone()[0])
            if summary.get("fingerprint") == fingerprint:
                return summary
    temporary = directory / "cleaning_audit.building.sqlite"
    temporary.unlink(missing_ok=True)
    counts = {}
    with closing(sqlite3.connect(temporary)) as connection:
        connection.executescript("""PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
          CREATE TABLE removals(id TEXT PRIMARY KEY,reason TEXT,operation_link TEXT) WITHOUT ROWID;
          CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT);""")
        batch = []
        with gzip.open(audit, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                if event["action"] != "removed":
                    continue
                for identifier in event["original_link_ids"]:
                    batch.append((identifier, event["reason"], event["link_id"]))
                if len(batch) >= 10000:
                    connection.executemany("INSERT INTO removals VALUES(?,?,?)", batch)
                    connection.commit()
                    batch.clear()
        connection.executemany("INSERT INTO removals VALUES(?,?,?)", batch)
        counts = dict(connection.execute("SELECT reason,count(*) FROM removals GROUP BY reason"))
        summary = {"source": str(audit), "counts": counts, "schema": 1, "fingerprint": fingerprint}
        connection.execute("INSERT INTO metadata VALUES('summary',?)", (json.dumps(summary),))
        connection.commit()
    temporary.replace(directory / "cleaning_audit.sqlite")
    return summary


def separate_directions(connection):
    # Test actual curves as well as reciprocal nodes: separate carriageways with
    # different shapes must not be mistaken for an overlapping two-way road.
    connection.execute("CREATE INDEX ends ON links(network,from_id,to_id)")
    cursor = connection.execute("""SELECT a.k,a.geom,b.k,b.geom FROM links a JOIN links b
      ON b.network=a.network AND b.from_id=a.to_id AND b.to_id=a.from_id
      WHERE a.k<b.k AND a.from_id<>a.to_id""")
    count = 0
    shifted_keys = set()
    while rows := cursor.fetchmany(4000):
        left = shapely.from_wkb([row[1] for row in rows])
        right = shapely.from_wkb([row[3] for row in rows])
        matches = shapely.equals_exact(left, shapely.reverse(right), tolerance=0.25)
        # Exact reversed vertex lists are the common case and cost O(vertices).
        # Only similarly bounded curves with different segmentation need the
        # more expensive distance comparison.
        possible = ~matches & np.all(np.abs(shapely.bounds(left)-shapely.bounds(right)) <= 0.25, axis=1)
        matches[possible] = shapely.hausdorff_distance(left[possible], right[possible]) <= 0.25
        updates = []
        for index in np.flatnonzero(matches):
            for key, original in ((rows[index][0], left[index]), (rows[index][2], right[index])):
                if key in shifted_keys:
                    continue
                shifted = shapely.offset_curve(original, -OFFSET_METRES, join_style="round")
                values = geometry_values(shifted)
                if values:
                    updates.append((*values, OFFSET_METRES, key))
                    shifted_keys.add(key)
        connection.executemany("UPDATE links SET geom=?,fine=?,coarse=?,minx=?,maxx=?,miny=?,maxy=?,display_offset_m=? WHERE k=?", updates)
        previous = count
        count += len(updates)
        if count//200000 != previous//200000:
            logging.info("Precomputed %s directional offsets", f"{count:,}")
    # Opposite directions can have different chain boundaries after merging.
    # Carry raw overlap evidence through exact provenance rather than requiring
    # identical final endpoints or guessing from nearby parallel carriageways.
    cursor = connection.execute("""SELECT a.k,a.geom FROM links a
      WHERE a.network='after' AND a.display_offset_m=0 AND a.k IN
        (SELECT m.after_key FROM links b JOIN source.members m ON m.before_id=b.id
         WHERE b.network='before' AND b.display_offset_m>0)""")
    while rows := cursor.fetchmany(4000):
        updates = []
        for key, blob in rows:
            shifted = shapely.offset_curve(shapely.from_wkb(blob), -OFFSET_METRES, join_style="round")
            values = geometry_values(shifted)
            if values:
                updates.append((*values, OFFSET_METRES, key))
        connection.executemany("UPDATE links SET geom=?,fine=?,coarse=?,minx=?,maxx=?,miny=?,maxy=?,display_offset_m=? WHERE k=?", updates)
        count += len(updates)
    connection.execute("DROP INDEX ends")
    connection.commit()
    logging.info("Directional offsets complete: %s links", f"{count:,}")
    return count


def prepare_circles(connection):
    # A circle encloses each disconnected raw link, including its detailed bends.
    # Store normalized Mercator coordinates once for inexpensive tile drawing.
    cursor = connection.execute("""SELECT l.k,l.minx,l.maxx,l.miny,l.maxy FROM links l
      JOIN audit.removals r ON r.id=l.id WHERE l.network='before'
      AND l.status='removed' AND r.reason='disconnected_car'""")
    while rows := cursor.fetchmany(5000):
        updates = []
        for key, xmin, xmax, ymin, ymax in rows:
            x, y = (xmin+xmax)/2, (ymin+ymax)/2
            radius = math.hypot(xmax-xmin, ymax-ymin)/2+4
            lon, lat = UNPROJECT.transform(x, y)
            cx = (lon+180)/360
            cy = (1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2
            cr = radius/(40075016.686*math.cos(math.radians(lat)))
            updates.append((cx, cy, cr, x-radius, x+radius, y-radius, y+radius, key))
        connection.executemany("UPDATE links SET cx=?,cy=?,cr=?,minx=?,maxx=?,miny=?,maxy=? WHERE k=?", updates)
    connection.commit()
