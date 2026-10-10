"""Phase 5a - spider_routes + spider_link_index from MATSim events XML.
SAX-streamed; flushes every BATCH_SIZE trips, so memory scales with open trips only.
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


class _VehicleState:
    __slots__ = ("trip_counter", "dep_time", "route", "in_trip")

    def __init__(self) -> None:
        self.trip_counter = 0
        self.dep_time = 0.0
        self.route: list[str] = []
        self.in_trip = False


class _TripExtractor(xml.sax.handler.ContentHandler):
    _RELEVANT = frozenset(["vehicle enters traffic", "entered link", "vehicle leaves traffic"])

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        super().__init__()
        self._con = con
        self._buf_person: list[int] = []
        self._buf_trip_idx: list[int] = []
        self._buf_dep: list[float] = []
        self._buf_links: list[list[str]] = []
        self._vehicles: dict[str, _VehicleState] = {}
        self._written = 0

    def startElement(self, name, attrs) -> None:
        if name != "event":
            return
        et = attrs.get("type")
        if et not in self._RELEVANT:
            return
        veh = attrs.get("vehicle")
        if not veh or not veh.endswith(":car"):
            return
        link = attrs.get("link")
        vs = self._vehicles.get(veh)
        if vs is None:
            vs = _VehicleState()
            self._vehicles[veh] = vs

        if et == "vehicle enters traffic":
            if vs.in_trip and vs.route:
                self._emit(veh, vs)
                vs.trip_counter += 1
            vs.dep_time = float(attrs.get("time", 0))
            vs.route = []
            if link:
                vs.route.append(link)
            vs.in_trip = True
        elif et == "entered link" and vs.in_trip:
            if link:
                vs.route.append(link)
        elif et == "vehicle leaves traffic":
            if vs.in_trip and vs.route:
                self._emit(veh, vs)
                vs.trip_counter += 1
            vs.route = []
            vs.in_trip = False

    def _emit(self, veh: str, vs: _VehicleState) -> None:
        person_id_str = veh.rsplit(":car", 1)[0]
        try:
            pid = int(person_id_str)
        except ValueError:
            log.warning("Non-numeric person_id %r in spider - skipped", person_id_str)
            return
        self._buf_person.append(pid)
        self._buf_trip_idx.append(vs.trip_counter)
        self._buf_dep.append(vs.dep_time)
        self._buf_links.append(list(vs.route))
        if len(self._buf_person) >= BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_person:
            return
        tbl = pa.table({
            "person_id":      pa.array(self._buf_person, type=pa.int64()),
            "trip_index":     pa.array(self._buf_trip_idx, type=pa.int32()),
            "departure_time": pa.array(self._buf_dep, type=pa.float64()),
            "route_links":    pa.array(self._buf_links, type=pa.list_(pa.string())),
        })
        self._con.register("_tmp_spider", tbl)
        self._con.execute("INSERT INTO spider_routes SELECT * FROM _tmp_spider")
        self._con.unregister("_tmp_spider")
        self._written += len(self._buf_person)
        log.info("  ... %d spider routes written", self._written)
        self._buf_person.clear()
        self._buf_trip_idx.clear()
        self._buf_dep.clear()
        self._buf_links.clear()

    def endDocument(self) -> None:
        for veh, vs in self._vehicles.items():
            if vs.in_trip and vs.route:
                self._emit(veh, vs)
        self._flush()
        log.info("spider_routes: %d total", self._written)

    @property
    def total(self) -> int:
        return self._written


def build_spider(db: duckdb.DuckDBPyConnection, events_xml: Path) -> int:
    handler = _TripExtractor(db)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    if str(events_xml).endswith(".gz"):
        with gzip.open(events_xml, "rb") as f:
            parser.parse(f)
    else:
        parser.parse(str(events_xml))

    # ORDER BY link_id clusters rows physically so backend WHERE link_id = ?
    # filters hit a tight row-group range (zonemap pruning) instead of a full scan.
    db.execute("""
        INSERT INTO spider_link_index
        SELECT link_id, person_id, trip_index, departure_time,
               pos AS position, len(route_links) AS route_length
        FROM (
            SELECT person_id, trip_index, departure_time, route_links,
                   UNNEST(route_links) AS link_id,
                   generate_subscripts(route_links, 1) AS pos
            FROM spider_routes
        )
        ORDER BY link_id
    """)
    n_idx = db.execute("SELECT COUNT(*) FROM spider_link_index").fetchone()[0]
    log.info("spider_link_index: %d rows", n_idx)
    return handler.total
