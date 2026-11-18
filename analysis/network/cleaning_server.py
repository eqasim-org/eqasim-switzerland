"""Local HTTP viewer; source networks stay read-only, derived tiles are cached."""
import argparse
from contextlib import closing
import gzip
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sqlite3
import time
import os
import getpass
from urllib.parse import parse_qs, urlparse

from pyproj import Transformer
import shapely
from shapely.geometry import Point
from shapely.ops import substring

from .cleaning_data import attr_value

PROJECT = Transformer.from_crs(4326, 2056, always_xy=True)
UNPROJECT = Transformer.from_crs(2056, 4326, always_xy=True)
STATUSES = {"retained", "removed", "merged", "added"}


def connect(database):
    # Published bundles are replaced atomically, never modified in place. Immutable
    # read-only handles avoid network-filesystem lock traffic on every map request.
    connection = sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    audit = Path(database).resolve().parent / "cleaning_audit.sqlite"
    if audit.exists():
        connection.execute("ATTACH DATABASE ? AS audit", (audit.as_uri()+"?mode=ro&immutable=1",))
        connection.execute("CREATE TEMP VIEW removal_reasons AS SELECT id,reason,operation_link FROM audit.removals")
    else:
        connection.execute("CREATE TEMP VIEW removal_reasons AS SELECT NULL AS id,NULL AS reason,NULL AS operation_link WHERE 0")
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-16384")
    return connection


def filter_sql(params):
    roads = [v for v in params.get("roads", [""])[0].split(",") if v]
    statuses = [v for v in params.get("statuses", [""])[0].split(",") if v in STATUSES]
    if not roads or not statuses:
        return "0", []
    if len(roads) > 100 or any(len(road) > 80 for road in roads):
        raise ValueError("Too many road categories")
    view = params.get("view", ["compare"])[0]
    if view == "before":
        source = "l.network='before'"
    elif view == "after":
        source = "l.network='after'"
    elif view == "compare":
        source = "(l.network='after' OR (l.network='before' AND l.status='removed'))"
        if params.get("originals", ["0"])[0] == "1":
            source = "(l.network='after' OR (l.network='before' AND l.status IN ('removed','merged')))"
    else:
        raise ValueError("Invalid network view")
    # A missing audit fails closed: never label an inferred removal as disconnected.
    sql = f"""{source} AND l.road IN ({','.join('?' for _ in roads)}) AND l.status IN ({','.join('?' for _ in statuses)})
      AND (l.status<>'removed' OR EXISTS(SELECT 1 FROM removal_reasons r
        WHERE r.id=l.id AND r.reason='disconnected_car'))"""
    return sql, roads + statuses


def spatial_rows(connection, bounds, params, limit, columns):
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError("Expected finite west,south,east,north bounds")
    west, south, east, north = bounds
    if not -180 <= west < east <= 180 or not -85 <= south < north <= 85:
        raise ValueError("Invalid map bounds")
    xmin, ymin, xmax, ymax = PROJECT.transform_bounds(west, south, east, north, densify_pts=21)
    if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
        raise ValueError("Map bounds are outside the supported network projection")
    sql, arguments = filter_sql(params)
    # CROSS JOIN keeps the RTree as the outer loop, avoiding a national road-type scan.
    query = f"""SELECT {columns} FROM spatial r CROSS JOIN links l ON l.k=r.k
      WHERE r.maxx>=? AND r.minx<=? AND r.maxy>=? AND r.miny<=? AND ({sql})"""
    values = [xmin, xmax, ymin, ymax, *arguments]
    if limit is not None:
        query += " LIMIT ?"
        values.append(limit)
    cursor = connection.execute(query, values)
    return cursor if limit is None else cursor.fetchall()


def viewport(connection, params, max_features=10000):
    start = time.monotonic()
    bounds = [float(v) for v in params.get("bounds", [""])[0].split(",")]
    zoom = float(params.get("zoom", ["13"])[0])
    geometry = "l.coarse" if zoom < 12 else "l.fine"
    rows = spatial_rows(connection, bounds, params, max_features + 1,
                        f"l.k,l.network,l.id,l.road,l.status,{geometry} AS path")
    return {"rows": [[r["k"], r["network"], r["id"], r["road"], r["status"], r["path"]] for r in rows[:max_features]],
            "limited": len(rows) > max_features, "limit": max_features,
            "simplification_m": 40 if zoom < 12 else 1.5,
            "elapsed_ms": round((time.monotonic() - start) * 1000)}


def geographic(shape):
    xy = shapely.get_coordinates(shape)
    lon, lat = UNPROJECT.transform(xy[:, 0], xy[:, 1])
    return [[round(float(y), 7), round(float(x), 7)] for x, y in zip(lon, lat)]


def link_detail(connection, key):
    row = connection.execute("SELECT * FROM links WHERE k=?", (key,)).fetchone()
    if row is None:
        raise KeyError("Link not found")
    record = json.loads(row["record"])
    if row["network"] == "before":
        matches = connection.execute("SELECT l.k,l.id,l.status FROM members m JOIN links l ON l.k=m.after_key WHERE m.before_id=? LIMIT 201", (row["id"],)).fetchall()
    else:
        matches = connection.execute("SELECT l.k,l.id,l.status FROM members m JOIN links l ON l.network='before' AND l.id=m.before_id WHERE m.after_key=? LIMIT 201", (key,)).fetchall()
    reason = connection.execute("SELECT reason,operation_link FROM removal_reasons WHERE id=?", (row["id"],)).fetchone() if row["network"] == "before" else None
    return {"key": row["k"], "network": row["network"], "status": row["status"], "road": row["road"],
            "removal": dict(reason) if reason else None,
            "record": record, "coordinates": geographic(shapely.from_wkb(row["geom"])),
            "counterparts": [dict(r) for r in matches[:200]], "counterparts_limited": len(matches) > 200}


def turn_detail(connection, key, mode="car"):
    row = connection.execute("SELECT * FROM links WHERE k=?", (key,)).fetchone()
    if row is None:
        raise KeyError("Link not found")
    record = json.loads(row["record"])
    if mode not in record.get("modes", "").split(","):
        return {"turns": [], "sequences": [], "warning": f"Selected link does not allow {mode}"}
    raw = attr_value(record, "disallowedNextLinks", "{}")
    restrictions = json.loads(raw) if isinstance(raw, str) else raw
    sequences = restrictions.get(mode, [])
    source = shapely.from_wkb(row["geom"])
    tail = geographic(substring(source, max(0, source.length - 90), source.length))
    outgoing = connection.execute("SELECT k,id,record,geom FROM links WHERE network=? AND from_id=? LIMIT 256",
                                  (row["network"], row["to_id"])).fetchall()
    turns, warnings = [], []
    # Validate sequences against this exact network and mode; dangling or mode-incompatible
    # references are reported rather than interpreted as real prohibitions.
    valid = []
    for sequence in sequences[:256]:
        chain, last = [], row["to_id"]
        if not sequence or len(sequence) > 80:
            warnings.append("Empty or overly long restriction sequence was not drawn")
            continue
        for identifier in sequence:
            target = connection.execute("SELECT id,from_id,to_id,record,geom FROM links WHERE network=? AND id=?",
                                        (row["network"], str(identifier))).fetchone()
            if target is None or target["from_id"] != last or mode not in json.loads(target["record"]).get("modes", "").split(","):
                warnings.append(f"Unresolved or incompatible restriction: {sequence}")
                chain = []
                break
            chain.append(target)
            last = target["to_id"]
        if chain:
            valid.append(([str(i) for i in sequence], chain))
    if len(sequences) > 256:
        warnings.append("Only the first 256 restriction sequences are shown")
    for target in outgoing:
        if mode not in json.loads(target["record"]).get("modes", "").split(","):
            continue
        state = "allowed"
        if any(ids == [target["id"]] for ids, _ in valid):
            state = "forbidden"
        elif any(ids[0] == target["id"] and len(ids) > 1 for ids, _ in valid):
            state = "conditional"
        shape = shapely.from_wkb(target["geom"])
        head = geographic(substring(shape, 0, min(shape.length, 140)))
        turns.append({"key": target["k"], "id": target["id"], "state": state, "coordinates": tail + head})
    return {"turns": turns, "warning": "; ".join(dict.fromkeys(warnings)),
            "sequences": [{"ids": ids, "coordinates": tail + [p for link in chain for p in geographic(shapely.from_wkb(link["geom"]))]}
                          for ids, chain in valid if len(ids) > 1]}


def search(connection, query):
    query = query.strip()
    if not query or len(query) > 100:
        return []
    rows = connection.execute("""SELECT k,network,id,road,status FROM links WHERE id=?
      UNION SELECT l.k,l.network,l.id,l.road,l.status FROM ways w JOIN links l ON l.k=w.k WHERE w.way=? LIMIT 100""",
                              (query, query)).fetchall()
    return [dict(r) for r in rows]


def pick(connection, params):
    lat, lon = float(params["lat"][0]), float(params["lon"][0])
    radius = min(250, max(2, float(params.get("radius", ["25"])[0])))
    point = Point(*PROJECT.transform(lon, lat))
    dx, dy = radius / (111000 * max(0.05, math.cos(math.radians(lat)))), radius / 110000
    rows = spatial_rows(connection, [lon - dx, lat - dy, lon + dx, lat + dy], params, 500,
                        "l.k,l.network,l.id,l.road,l.status,l.geom")
    found = []
    for row in rows:
        distance = shapely.from_wkb(row["geom"]).distance(point)
        if distance <= radius:
            found.append({k: row[k] for k in ("k", "network", "id", "road", "status")} | {"distance": round(distance, 2)})
    return sorted(found, key=lambda row: (row["distance"], row["network"]))[:20]


def handler_for(directory):
    from .cleaning_tiles import TileStore
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    tiles = TileStore(directory)
    manifest["display_revision"] = str(tiles.drawing.stat().st_mtime_ns)
    manifest["audit_available"] = (directory / "cleaning_audit.sqlite").exists()
    if manifest["audit_available"]:
        with closing(connect(directory / "network.sqlite")) as connection:
            manifest["audit"] = json.loads(connection.execute("SELECT value FROM audit.metadata WHERE name='summary'").fetchone()[0])
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/tiles/"):
                try:
                    _, _, kind, z, x, filename = parsed.path.split("/")
                    if not filename.endswith(".png"):
                        raise ValueError("Invalid tile filename")
                    z, x, y = int(z), int(x), int(filename[:-4])
                    if not 0 <= z <= 20 or not 0 <= x < 2**z or not 0 <= y < 2**z:
                        raise ValueError("Invalid tile coordinates")
                    params = parse_qs(parsed.query, keep_blank_values=True)
                    content, count = tiles.get(kind, z, x, y, params)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Cache-Control", "private, max-age=604800")
                    if count is not None:
                        self.send_header("X-Links-Rendered", str(count))
                    self.end_headers()
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except (ValueError, KeyError):
                    self.send_error(400, "Invalid tile request")
                except Exception as error:
                    self.log_error("Tile failed: %s", error)
                    self.send_error(502, "Tile unavailable; please retry")
                return
            if parsed.path.rstrip("/") == "/api/manifest":
                self.send_json(manifest)
                return
            if parsed.path.startswith("/api/"):
                try:
                    params = parse_qs(parsed.query, keep_blank_values=True)
                    with closing(connect(directory / "network.sqlite")) as connection:
                        if parsed.path == "/api/viewport":
                            data = viewport(connection, params, manifest["max_features"])
                        elif parsed.path == "/api/link":
                            data = link_detail(connection, int(params["key"][0]))
                            with closing(connect(tiles.drawing)) as display:
                                row = display.execute("SELECT geom,display_offset_m FROM links WHERE k=?", (int(params["key"][0]),)).fetchone()
                                data["display_coordinates"] = geographic(shapely.from_wkb(row["geom"]))
                                data["display_offset_m"] = row["display_offset_m"]
                        elif parsed.path == "/api/turns":
                            data = turn_detail(connection, int(params["key"][0]), params.get("mode", ["car"])[0])
                        elif parsed.path == "/api/search":
                            data = search(connection, params.get("q", [""])[0])
                        elif parsed.path == "/api/pick":
                            with closing(connect(tiles.drawing)) as display:
                                data = pick(display, params)
                        else:
                            self.send_error(404)
                            return
                    self.send_json(data)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    self.send_json({"error": str(error)}, status=400)
                return
            # Do not expose the database, build logs or arbitrary files through HTTP.
            if parsed.path not in ("/", "/index.html", "/manifest.json") and not parsed.path.startswith("/assets/"):
                self.send_error(404)
                return
            target = Path(self.translate_path(parsed.path)).resolve()
            asset_root = directory / "assets"
            if (directory not in target.parents and target != directory) or (
                    parsed.path.startswith("/assets/") and asset_root not in target.parents):
                self.send_error(403)
                return
            super().do_GET()

        def send_json(self, data, status=200):
            content = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if "gzip" in self.headers.get("Accept-Encoding", "") and len(content) > 1500:
                content = gzip.compress(content, compresslevel=3)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            # Panning can generate many requests; keep the terminal quiet.
            if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
                super().log_message(format, *args)
    return Handler


def viewer_urls(port):
    urls = {"Local browser / SSH tunnel": f"http://127.0.0.1:{port}/"}
    if Path("/cluster").exists() or os.environ.get("JUPYTERHUB_SERVICE_PREFIX"):
        prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", f"/user/{getpass.getuser()}/")
        host = urlparse(os.environ.get("JUPYTERHUB_API_URL", "https://jupyter.euler.hpc.ethz.ch/hub/api")).netloc
        urls["Euler Jupyter (start this server on your Jupyter node)"] = f"https://{host}{prefix.rstrip('/')}/proxy/{port}/"
    return urls


def serve(directory, host="127.0.0.1", port=8765, prepare=None):
    # Reserve the port before expensive preprocessing, then accept requests only
    # once the complete bundle and its request handler are ready.
    with ThreadingHTTPServer((host, port), SimpleHTTPRequestHandler, bind_and_activate=False) as server:
        server.server_bind()
        if prepare is not None:
            prepare()
        server.RequestHandlerClass = handler_for(directory)
        server.server_activate()
        for label, url in viewer_urls(server.server_port).items():
            print(f"{label}: {url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.directory, args.host, args.port)
