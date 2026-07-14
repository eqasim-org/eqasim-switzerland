"""Single SAX pass over output_events.xml[.gz]: per-link speeds per 15-min bin plus
PT boarding/alighting and transfer counts, all from standard MATSim event types."""

from __future__ import annotations

import gzip
import logging
import xml.sax
import xml.sax.handler
from pathlib import Path

import duckdb
import pyarrow as pa

log = logging.getLogger(__name__)

LINK_SPEED_BATCH = 500_000


def _bin15(t: float) -> int:
    """15-minute time bin index 0..95; times after midnight wrap into the day."""
    b = int((t % 86400) // 900)
    return b if 0 <= b <= 95 else (0 if b < 0 else 95)


def _hour(t: float) -> int:
    """Hour-of-day bucket 0..23."""
    h = int((t % 86400) // 3600)
    return h if 0 <= h <= 23 else (0 if h < 0 else 23)


def _is_real_activity(act_type: str | None) -> bool:
    """True for a genuine destination activity, False for MATSim interaction connectors."""
    if not act_type:
        return False
    return not act_type.endswith("interaction")


class _EventsHandler(xml.sax.handler.ContentHandler):
    def __init__(self, link_lengths: dict[str, float], track_speeds: bool = True) -> None:
        super().__init__()
        self._len = link_lengths
        self._track_speeds = track_speeds
        # (link_id, time_bin) -> [sum_speed, count]
        self.speed_acc: dict[tuple[str, int], list] = {}
        # vehicle -> (current link, enter time)
        self._veh_link: dict[str, tuple[str, float]] = {}
        self._veh_line: dict[str, str] = {}        # vehicle -> line_id
        self._veh_driver: dict[str, str] = {}      # vehicle -> driver person id
        self._veh_facility: dict[str, str] = {}    # vehicle -> current facility
        # (line_id, facility, hour) -> [boardings, alightings]
        self.board_acc: dict[tuple[str, str, int], list] = {}
        # person -> (alight_line, alight_facility, alight_time); cleared on a real activity
        self._last_alight: dict[str, tuple[str, str, float]] = {}
        # person -> transfer_stop awaiting the onward leg's egress
        self._pending_egress: dict[str, str] = {}
        # transfer stop -> {'in':int,'out':int,'lines':{from:{to:n}},'dests':{egress_stop:n}}
        self.transfer_data: dict[str, dict] = {}

    def startElement(self, name, attrs) -> None:
        if name != "event":
            return
        et = attrs.get("type")

        if et == "entered link":
            if not self._track_speeds:
                return
            veh = attrs.get("vehicle")
            link = attrs.get("link")
            if veh and link:
                self._veh_link[veh] = (link, float(attrs.get("time", 0)))
        elif et == "left link":
            veh = attrs.get("vehicle")
            link = attrs.get("link")
            prev = self._veh_link.pop(veh, None)
            if prev and link and prev[0] == link and link in self._len:
                tt = float(attrs.get("time", 0)) - prev[1]
                length = self._len[link]
                if tt > 0 and length and length > 0:
                    spd = length / tt  # m/s
                    key = (link, _bin15(prev[1]))
                    acc = self.speed_acc.get(key)
                    if acc is None:
                        self.speed_acc[key] = [spd, 1]
                    else:
                        acc[0] += spd
                        acc[1] += 1

        elif et == "TransitDriverStarts":
            veh = attrs.get("vehicleId")
            if veh:
                self._veh_line[veh] = attrs.get("transitLineId")
                self._veh_driver[veh] = attrs.get("driverId")
        elif et == "VehicleArrivesAtFacility":
            veh = attrs.get("vehicle")
            fac = attrs.get("facility")
            if veh and fac:
                self._veh_facility[veh] = fac
        elif et in ("actstart", "actend"):
            # a real activity ends the journey; the previous alighting cannot be a transfer
            if _is_real_activity(attrs.get("actType")):
                self._last_alight.pop(attrs.get("person"), None)
        elif et in ("PersonEntersVehicle", "PersonLeavesVehicle"):
            veh = attrs.get("vehicle")
            line = self._veh_line.get(veh)
            if line is None:  # not a transit vehicle
                return
            person = attrs.get("person")
            if person is not None and person == self._veh_driver.get(veh):
                return  # the driver, not a passenger
            fac = self._veh_facility.get(veh)
            if fac is None:
                return
            t = float(attrs.get("time", 0))
            key = (line, fac, _hour(t))
            acc = self.board_acc.get(key)
            if acc is None:
                self.board_acc[key] = [0, 0]
                acc = self.board_acc[key]
            if et == "PersonEntersVehicle":
                acc[0] += 1
                # transfer: prior PT alighting with no real activity since;
                # attributed to the inbound (alight) facility
                prev = self._last_alight.pop(person, None)
                if prev is not None:
                    from_line, x_stop, _ = prev
                    d = self.transfer_data.get(x_stop)
                    if d is None:
                        d = {"in": 0, "out": 0, "lines": {}, "dests": {}}
                        self.transfer_data[x_stop] = d
                    d["in"] += 1
                    d["out"] += 1
                    fl = d["lines"].setdefault(from_line, {})
                    fl[line] = fl.get(line, 0) + 1
                    self._pending_egress[person] = x_stop
            else:  # PersonLeavesVehicle (alighting)
                acc[1] += 1
                x_stop = self._pending_egress.pop(person, None)
                if x_stop is not None:
                    dests = self.transfer_data[x_stop]["dests"]
                    dests[fac] = dests.get(fac, 0) + 1
                self._last_alight[person] = (line, fac, t)


def parse_events(
    db: duckdb.DuckDBPyConnection, events_xml: Path, *, track_speeds: bool = True,
) -> "_EventsHandler":
    """Single SAX pass over the events file; returns the handler with accumulators filled.

    Does NOT write to the DB - callers decide what to insert."""
    link_lengths = (
        dict(db.execute(
            "SELECT link_id, length FROM network_links WHERE length IS NOT NULL"
        ).fetchall()) if track_speeds else {}
    )
    handler = _EventsHandler(link_lengths, track_speeds=track_speeds)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    if str(events_xml).endswith(".gz"):
        with gzip.open(events_xml, "rb") as f:
            parser.parse(f)
    else:
        parser.parse(str(events_xml))
    return handler


def extract(
    db: duckdb.DuckDBPyConnection,
    events_xml: Path,
) -> tuple[dict[tuple[str, str, int], list], dict[str, dict]]:
    """Run the events pass, fill the link_speeds table, and return (board_acc, transfer_data)."""
    handler = parse_events(db, events_xml)
    _insert_link_speeds(db, handler.speed_acc)
    log.info("events_extras: link_speeds rows=%d, boarding keys=%d, transfer stops=%d",
             len(handler.speed_acc), len(handler.board_acc), len(handler.transfer_data))
    return handler.board_acc, handler.transfer_data


def _insert_link_speeds(db: duckdb.DuckDBPyConnection, acc: dict) -> int:
    """Build link_speeds: one row per (link_id, 15-min bin), enriched from network_links."""
    if not acc:
        return 0
    links, bins, speeds, vols = [], [], [], []
    for (link_id, tbin), (sum_spd, cnt) in acc.items():
        links.append(link_id)
        bins.append(tbin)
        speeds.append(sum_spd / cnt)
        vols.append(cnt)

    db.execute("""
        CREATE TEMP TABLE _ls_raw (
            link_id VARCHAR, time_bin INTEGER, avg_speed DOUBLE, volume INTEGER
        )
    """)
    for start in range(0, len(links), LINK_SPEED_BATCH):
        sl = slice(start, start + LINK_SPEED_BATCH)
        tbl = pa.table({
            "link_id": pa.array(links[sl], type=pa.string()),
            "time_bin": pa.array(bins[sl], type=pa.int32()),
            "avg_speed": pa.array(speeds[sl], type=pa.float64()),
            "volume": pa.array(vols[sl], type=pa.int32()),
        })
        db.register("_tmp_ls", tbl)
        db.execute("INSERT INTO _ls_raw SELECT * FROM _tmp_ls")
        db.unregister("_tmp_ls")

    # LEFT JOIN so links absent from the network still get a speed row
    db.execute("""
        INSERT INTO link_speeds (link_id, time_bin, avg_speed, volume, freespeed, road_type, canton_id)
        SELECT r.link_id, r.time_bin, r.avg_speed, r.volume,
               nl.freespeed, nl.road_type, nl.canton_id
        FROM _ls_raw r
        LEFT JOIN network_links nl ON nl.link_id = r.link_id
    """)
    db.execute("DROP TABLE _ls_raw")
    n = db.execute("SELECT COUNT(*) FROM link_speeds").fetchone()[0]
    log.info("link_speeds: %d (link,15min) rows inserted", n)
    return n
