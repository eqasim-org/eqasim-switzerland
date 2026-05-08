"""Phase 5b — network_links + network_nodes from MATSim XML.

SAX-streamed for memory efficiency. Coords are EPSG:2056 (LV95) as the
synpop pipeline runs everything in LV95.
"""

from __future__ import annotations

import gzip
import logging
import xml.sax
import xml.sax.handler
from pathlib import Path

import duckdb
import pyarrow as pa

log = logging.getLogger(__name__)

BATCH_SIZE = 200_000


class _NetworkParser(xml.sax.handler.ContentHandler):
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        super().__init__()
        self._con = con
        self._nodes: dict[str, tuple[float, float]] = {}
        self._link_buf: list[tuple] = []

    def startElement(self, name, attrs):
        if name == "node":
            nid = attrs.get("id")
            x = float(attrs.get("x", "nan"))
            y = float(attrs.get("y", "nan"))
            self._nodes[nid] = (x, y)
        elif name == "link":
            lid = attrs.get("id")
            f = attrs.get("from")
            t = attrs.get("to")
            length = _to_float(attrs.get("length"))
            capacity = _to_float(attrs.get("capacity"))
            freespeed = _to_float(attrs.get("freespeed"))
            permlanes = _to_float(attrs.get("permlanes"))
            modes = attrs.get("modes")
            self._link_buf.append((lid, f, t, length, capacity, freespeed, permlanes, modes))
            if len(self._link_buf) >= BATCH_SIZE:
                self._flush_links()

    def endDocument(self):
        self._flush_nodes()
        self._flush_links()

    def _flush_nodes(self):
        if not self._nodes:
            return
        ids = list(self._nodes.keys())
        xs = [self._nodes[i][0] for i in ids]
        ys = [self._nodes[i][1] for i in ids]
        tbl = pa.table({
            "node_id": pa.array(ids, type=pa.string()),
            "x": pa.array(xs, type=pa.float64()),
            "y": pa.array(ys, type=pa.float64()),
        })
        self._con.register("_tmp_n", tbl)
        self._con.execute("""
            INSERT INTO network_nodes SELECT node_id, ST_Point(x, y) FROM _tmp_n
        """)
        self._con.unregister("_tmp_n")
        log.info("network_nodes: %d", len(ids))

    def _flush_links(self):
        if not self._link_buf:
            return
        link_id, fr, to, length, cap, fs, pl, modes = zip(*self._link_buf)
        from_xy = [self._nodes.get(f) for f in fr]
        to_xy = [self._nodes.get(t) for t in to]
        x1 = [p[0] if p else None for p in from_xy]
        y1 = [p[1] if p else None for p in from_xy]
        x2 = [p[0] if p else None for p in to_xy]
        y2 = [p[1] if p else None for p in to_xy]
        tbl = pa.table({
            "link_id":  pa.array(link_id, type=pa.string()),
            "from_node": pa.array(fr, type=pa.string()),
            "to_node":   pa.array(to, type=pa.string()),
            "length":    pa.array(length, type=pa.float64()),
            "capacity":  pa.array(cap, type=pa.float64()),
            "freespeed": pa.array(fs, type=pa.float64()),
            "permlanes": pa.array(pl, type=pa.float64()),
            "modes":     pa.array(modes, type=pa.string()),
            "x1": pa.array(x1, type=pa.float64()),
            "y1": pa.array(y1, type=pa.float64()),
            "x2": pa.array(x2, type=pa.float64()),
            "y2": pa.array(y2, type=pa.float64()),
        })
        self._con.register("_tmp_l", tbl)
        self._con.execute("""
            INSERT INTO network_links
                (link_id, from_node, to_node, length, capacity, freespeed, permlanes, modes, geom)
            SELECT link_id, from_node, to_node, length, capacity, freespeed, permlanes, modes,
                   CASE WHEN x1 IS NULL OR x2 IS NULL THEN NULL
                        ELSE ST_MakeLine(ST_Point(x1,y1), ST_Point(x2,y2)) END
            FROM _tmp_l
        """)
        self._con.unregister("_tmp_l")
        log.info("network_links: %d", len(link_id))
        self._link_buf.clear()


def build_network(db: duckdb.DuckDBPyConnection, network_xml: Path) -> tuple[int, int]:
    parser = xml.sax.make_parser()
    handler = _NetworkParser(db)
    parser.setContentHandler(handler)
    if str(network_xml).endswith(".gz"):
        with gzip.open(network_xml, "rb") as f:
            parser.parse(f)
    else:
        parser.parse(str(network_xml))
    n_nodes = db.execute("SELECT COUNT(*) FROM network_nodes").fetchone()[0]
    n_links = db.execute("SELECT COUNT(*) FROM network_links").fetchone()[0]
    return n_nodes, n_links


def load_link_speeds(db: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    """Optional eqasim upstream input. Skipped silently if missing."""
    if not parquet_path.exists():
        return 0
    db.execute(f"""
        INSERT INTO link_speeds (link_id, time_bucket, speed)
        SELECT CAST(link_id AS VARCHAR), time_bucket, speed
        FROM read_parquet('{parquet_path}')
    """)
    n = db.execute("SELECT COUNT(*) FROM link_speeds").fetchone()[0]
    log.info("link_speeds: %d rows", n)
    return n


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None
