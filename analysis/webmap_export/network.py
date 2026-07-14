"""Phase 5b - network_links + network_nodes from MATSim XML.
SAX-streamed; coords are EPSG:2056 (LV95) like the rest of the pipeline.
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
    # Road class is carried as a per-link OSM attribute in the MATSim XML.
    _ROAD_TYPE_ATTR = "osm:way:highway"

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        super().__init__()
        self._con = con
        self._nodes: dict[str, tuple[float, float]] = {}
        self._link_buf: list[tuple] = []
        self._cur_link: tuple | None = None
        self._cur_road_type: str | None = None
        self._capture_road_type = False
        self._char_buf: list[str] = []

    def startElement(self, name, attrs):
        if name == "node":
            nid = attrs.get("id")
            x = float(attrs.get("x", "nan"))
            y = float(attrs.get("y", "nan"))
            self._nodes[nid] = (x, y)
        elif name == "link":
            # Append deferred to endElement('link') so the child road_type attribute can be folded in.
            self._cur_link = (
                attrs.get("id"), attrs.get("from"), attrs.get("to"),
                _to_float(attrs.get("length")), _to_float(attrs.get("capacity")),
                _to_float(attrs.get("freespeed")), _to_float(attrs.get("permlanes")),
                attrs.get("modes"),
            )
            self._cur_road_type = None
        elif name == "attribute" and self._cur_link is not None:
            if attrs.get("name") == self._ROAD_TYPE_ATTR:
                self._capture_road_type = True
                self._char_buf = []

    def characters(self, content):
        if self._capture_road_type:
            self._char_buf.append(content)

    def endElement(self, name):
        if name == "attribute" and self._capture_road_type:
            self._cur_road_type = ("".join(self._char_buf).strip() or None)
            self._capture_road_type = False
        elif name == "link" and self._cur_link is not None:
            self._link_buf.append((*self._cur_link, self._cur_road_type))
            self._cur_link = None
            self._cur_road_type = None
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
            INSERT INTO network_nodes (node_id, geom)
            SELECT node_id, ST_Point(x, y) FROM _tmp_n
        """)
        self._con.unregister("_tmp_n")
        log.info("network_nodes: %d", len(ids))

    def _flush_links(self):
        if not self._link_buf:
            return
        link_id, fr, to, length, cap, fs, pl, modes, road_type = zip(*self._link_buf)
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
            "road_type": pa.array(road_type, type=pa.string()),
            "x1": pa.array(x1, type=pa.float64()),
            "y1": pa.array(y1, type=pa.float64()),
            "x2": pa.array(x2, type=pa.float64()),
            "y2": pa.array(y2, type=pa.float64()),
        })
        self._con.register("_tmp_l", tbl)
        self._con.execute("""
            INSERT INTO network_links
                (link_id, from_node, to_node, length, capacity, freespeed, permlanes, modes, road_type, geom)
            SELECT link_id, from_node, to_node, length, capacity, freespeed, permlanes, modes, road_type,
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


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None
