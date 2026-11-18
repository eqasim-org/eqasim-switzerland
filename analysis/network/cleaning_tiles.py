"""Complete, spatially indexed network tiles and cached OSM background tiles."""
from collections import OrderedDict
from contextlib import closing
from io import BytesIO
import math
import os
from pathlib import Path
import json
import hashlib
import sqlite3
import threading
import time

from PIL import Image, ImageDraw
import requests

COLORS = {"retained": "#3565a5", "removed": "#dc3545", "merged": "#914ac4", "added": "#099c8b"}


def build_visual_index(directory):
    """Extract small drawing records once; never change the source database."""
    directory = Path(directory)
    source = directory / "network.sqlite"
    target = directory / "drawing.sqlite"
    audit = directory / "cleaning_audit.sqlite"
    stamp = json.dumps([source.stat().st_size, source.stat().st_mtime_ns,
                        audit.stat().st_mtime_ns if audit.exists() else None, 2,
                        hashlib.sha256(Path(__file__).with_name("cleaning_preprocess.py").read_bytes()).hexdigest()])
    if target.exists():
        with closing(sqlite3.connect(target.as_uri()+"?mode=ro&immutable=1", uri=True)) as connection:
            if connection.execute("SELECT value FROM stamp").fetchone()[0] == stamp:
                return target
    temporary = directory / f"drawing.{os.getpid()}.{threading.get_ident()}.tmp"
    temporary.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(temporary.as_uri(), uri=True)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-262144")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("ATTACH DATABASE ? AS source", (source.as_uri()+"?mode=ro&immutable=1",))
            connection.execute("""CREATE TABLE links(k INTEGER PRIMARY KEY, network TEXT,id TEXT,
              road TEXT,status TEXT,from_id TEXT,to_id TEXT,geom BLOB,fine TEXT,coarse TEXT,
              minx REAL,maxx REAL,miny REAL,maxy REAL,display_offset_m REAL DEFAULT 0,
              cx REAL,cy REAL,cr REAL)""")
            connection.execute("""INSERT INTO links(k,network,id,road,status,from_id,to_id,geom,fine,coarse,minx,maxx,miny,maxy)
              SELECT k,network,id,road,status,from_id,to_id,geom,fine,coarse,minx,maxx,miny,maxy FROM source.links""")
            from .cleaning_preprocess import separate_directions, prepare_circles
            offset_count = separate_directions(connection)
            if audit.exists():
                connection.execute("ATTACH DATABASE ? AS audit", (audit.as_uri()+"?mode=ro&immutable=1",))
                prepare_circles(connection)
            connection.execute("CREATE VIRTUAL TABLE spatial USING rtree(k,minx,maxx,miny,maxy)")
            connection.execute("INSERT INTO spatial SELECT k,minx,maxx,miny,maxy FROM links")
            connection.execute("CREATE TABLE display_metadata(name TEXT PRIMARY KEY,value TEXT)")
            connection.execute("INSERT INTO display_metadata VALUES('offset_links',?)", (str(offset_count),))
            connection.execute("CREATE TABLE stamp(value TEXT)")
            connection.execute("INSERT INTO stamp VALUES (?)", (stamp,))
            connection.commit()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def decode(text):
    position, latitude, longitude = 0, 0, 0
    while position < len(text):
        values = []
        for _ in range(2):
            value, shift = 0, 0
            while True:
                byte = ord(text[position]) - 63
                position += 1
                value |= (byte & 31) << shift
                shift += 5
                if byte < 32:
                    break
            values.append(~(value >> 1) if value & 1 else value >> 1)
        latitude += values[0]
        longitude += values[1]
        yield longitude / 1e6, latitude / 1e6


def tile_bounds(z, x, y, padding=0):
    n = 2**z
    def lat(value):
        return max(-85, min(85, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * value / n))))))
    return [max(-180, (x-padding)/n*360-180), lat(y+1+padding),
            min(180, (x+1+padding)/n*360-180), lat(y-padding)]


def dashed_line(draw, pixels, color, width):
    distance = 0
    for start, end in zip(pixels, pixels[1:]):
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = math.hypot(dx, dy)
        position = 0
        while position < length:
            step = min(length-position, 10-distance % 10)
            if int(distance // 10) % 2 == 0:
                a, b = position/length, (position+step)/length
                draw.line([(start[0]+a*dx, start[1]+a*dy), (start[0]+b*dx, start[1]+b*dy)], fill=color, width=width)
            position += step
            distance += step


def render_network(database, z, x, y, params, drawing=None):
    from .cleaning_server import connect, geographic, spatial_rows
    import shapely

    # Twice the display resolution keeps lines sharp on retina screens. Padding
    # includes strokes crossing tile boundaries; no matching-feature limit applies.
    size, pad = 512, 20
    canvas = Image.new("RGBA", (size+2*pad, size+2*pad))
    draw = ImageDraw.Draw(canvas)
    world = size * 2**z
    origin_x, origin_y = x*size-pad, y*size-pad
    count = 0
    geometry = "l.geom" if z >= 16 else "l.fine" if z >= 12 else "l.coarse"
    # Empty tiles outside the Swiss projection's supported region need no query.
    bounds = tile_bounds(z, x, y, pad/size)
    if bounds[2] < 3 or bounds[0] > 17 or bounds[3] < 43 or bounds[1] > 51:
        rows = ()
    else:
        connection = connect(drawing or database)
        circles = ",l.cx,l.cy,l.cr" if drawing is not None else ",NULL AS cx,NULL AS cy,NULL AS cr"
        rows = spatial_rows(connection, bounds, params, None, f"l.network,l.status,{geometry} AS path"+circles)
    try:
        for row in rows:
            points = ((lon, lat) for lat, lon in geographic(shapely.from_wkb(row["path"]))) if z >= 16 else decode(row["path"])
            pixels = [(round((lon+180)/360*world-origin_x),
                       round((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*world-origin_y))
                      for lon, lat in points]
            if len(pixels) >= 2:
                width = 5 if row["status"] == "removed" else 4
                if row["network"] == "before" and row["status"] == "merged":
                    dashed_line(draw, pixels, COLORS[row["status"]], width)
                else:
                    draw.line(pixels, fill=COLORS[row["status"]], width=width)
            if params.get("highlight", ["0"])[0] == "1" and row["cx"] is not None:
                cx, cy = row["cx"]*world-origin_x, row["cy"]*world-origin_y
                radius = max(14, row["cr"]*world)
                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline="#a90014", width=3)
            count += 1
    finally:
        if rows != ():
            connection.close()
    buffer = BytesIO()
    canvas.crop((pad, pad, pad+size, pad+size)).save(buffer, "PNG", compress_level=2)
    return buffer.getvalue(), count


class TileStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.drawing = build_visual_index(self.directory)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.workers = threading.BoundedSemaphore(4)

    def get(self, kind, z, x, y, params):
        if kind not in ("osm", "network"):
            raise ValueError("Unknown tile layer")
        key = kind, z, x, y, tuple((name, tuple(params.get(name, []))) for name in ("view", "roads", "statuses", "originals", "highlight"))
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        with self.workers:
            with self.lock:
                if key in self.cache:
                    return self.cache[key]
            result = (self.osm(z, x, y), None) if kind == "osm" else render_network(self.directory / "network.sqlite", z, x, y, params, self.drawing)
        with self.lock:
            self.cache[key] = result
            while len(self.cache) > 256:
                self.cache.popitem(last=False)
        return result

    def osm(self, z, x, y):
        target = self.directory / ".tile_cache" / "osm" / str(z) / str(x) / f"{y}.png"
        if target.exists() and time.time() - target.stat().st_mtime < 7*86400:
            return target.read_bytes()
        try:
            response = requests.get(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", timeout=15,
                                    headers={"User-Agent": "eqasim-switzerland-network-inspector/1.0 (interactive academic research map)"})
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            if image.size != (256, 256) or image.format != "PNG":
                raise ValueError("Unexpected OSM tile")
            image.verify()
            target.parent.mkdir(parents=True, exist_ok=True)
            # Concurrent users may request the same tile. Publish only complete files.
            temporary = target.with_suffix(f".{threading.get_ident()}.tmp")
            temporary.write_bytes(response.content)
            temporary.replace(target)
            return response.content
        except (requests.RequestException, OSError, ValueError):
            if target.exists():
                return target.read_bytes()
            raise
