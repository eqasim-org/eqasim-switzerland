"""The PT pricing inputs: SBB_all_distances.csv, gtfs_zones.csv, pricingDescription.xml and the Léman Pass tables.

gtfs_zones.csv gives the fare zones of each stop by its id. It was made for feeds whose stop ids are DiDok numbers
(8503000, 8503000:0:3, Parent8503000). Since the 2026 timetable, the Swiss feed uses SLOIDs (ch:1:sloid:3000:2:3), so
almost no stop of the feed is found in it any more. The stage therefore checks the stop ids of data.gtfs.cleaned: when
too few of the stops of a priced station are found by their id (or their station, as eqasim-java does), the feed stops
get rows of their own, keyed by their id, with the zones of their station (its DiDok number: the didok column of the
feed, else the number of the SLOID). The original rows are kept unchanged, and the new ones are appended.

The stops abroad then get rows of their own as well, so that they work exactly as the Swiss stops, also in eqasim-java:
each feed stop in a commune of the list of zones abroad (unireso_france_zones.csv, with the outlines of
communes_france.gpkg) gets the zone of its commune, unless gtfs_zones.csv already gives it a zone of that network. A
commune listed in several zones (Pougny: 240 and 250) gives each of its stops the zone of the nearest stop that has one
of them, else the first listed.

The French local networks (data.pt_pricing.french_networks: STAR'T, EVA'D and TAC) are then added as zone networks: the
stops served by their operators get their zones in gtfs_zones.csv, with a higher priority than the other networks (the
network priority column), and their fares are written to pricingDescription.xml.

The single tickets of the Léman Pass (the tickets across the border around Geneva) are read from the fare table of tpg
(tpg_<date>_Fare-system_Geneva.xlsx in pt_pricing_lemanpass_path) and written as two CSV files for eqasim-java,
described in utils.gtfs_router.lemanpass.write_java_tables: lemanpass_products.csv (one row per product, with its
places and its prices) and lemanpass_stations.csv (the stops of each station). The zone of each station is that of its stops in gtfs_zones.csv,
the stops abroad included. Without the fare table, the tables have only their header (no Léman Pass).

The stage returns (SBB_all_distances.csv, gtfs_zones.csv, pricingDescription.xml, {name: path of a Léman Pass table}),
the names being those of utils.gtfs_router.lemanpass.JAVA_FILES.
"""
import csv
import logging
import os
import re
import shutil
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd

from data.pt_pricing import french_networks
from utils.gtfs_router.lemanpass import NETWORK as LEMANPASS_NETWORK, LemanPass, fare_table_path, write_java_tables
from utils.gtfs_router.pricing import LOCATION_COLUMN, Pricing, java_station_id, station_id
from utils.gtfs_router.zones import FOREIGN_NETWORK, foreign_communes

logger = logging.getLogger("synpp")

# An SNCF stop id with the UIC code of its station and a check digit: StopPoint:OCETrain:::TER-85171421 -> 8517142
SNCF_STOP = re.compile(r"^StopPoint:OCE\w*:::\w*-(\d{7})\d$")


def configure(context):
    context.config("data_path")
    pt_pricing_option = context.config("generate_pt_pricing_inputs", default=False)

    # Re-keying of gtfs_zones.csv to the stop ids of the feed: always (a row for each feed stop it does not find, with the
    # zones of its station), auto (only when the check below fails; the SNCF stops always) or never
    context.config("pt_pricing_rekey_stops", default = "always")
    # The share of the feed stops of priced stations that must be found in gtfs_zones.csv for auto to keep it as it is
    context.config("pt_pricing_min_stop_coverage", default = 0.9)

    # The communes abroad and their zones, relative to data_path (the same as data.pt_pricing.zones)
    context.config("pt_zones_foreign_communes_file", default = "pt_pricing/communes_france.gpkg")
    context.config("pt_zones_foreign_zones_file", default = "pt_pricing/unireso_france_zones.csv")
    # The folder of the fare table of tpg (tpg_<date>_Fare-system_Geneva.xlsx), relative to data_path; without it, no Léman Pass
    context.config("pt_pricing_lemanpass_path", default = "pt_pricing/output")

    context.stage("data.gtfs.cleaned")
    context.stage("data.spatial.swiss_border")

    if pt_pricing_option:
        context.stage("data.pt_pricing.t603.create_sbb_distances")
        context.stage("data.pt_pricing.t651.process_all_authorities")


def execute(context):

    # Copy or generate the pricing inputs.

    data_path         = context.config("data_path")
    pt_pricing_option = context.config("generate_pt_pricing_inputs")

    if not pt_pricing_option:
        sbb_input_file     = f"{data_path}/pt_pricing/output/SBB_all_distances.csv"
        zones_input_file   = f"{data_path}/pt_pricing/output/gtfs_zones.csv"
        pricing_input_file = f"{data_path}/pt_pricing/output/pricingDescription.xml"

        shutil.copy(sbb_input_file, context.path())
        shutil.copy(zones_input_file, context.path())
        shutil.copy(pricing_input_file, context.path())

        logger.info("Files copied")

    else:
        sbb_distances = context.stage("data.pt_pricing.t603.create_sbb_distances")
        authorities   = context.stage("data.pt_pricing.t651.process_all_authorities")

        # TODO create the stages that create the pricing description xml file
        pricing_input_file = f"{data_path}/pt_pricing/output/pricingDescription.xml"
        shutil.copy(pricing_input_file, context.path())

        sbb_distances.to_csv(f"{context.path()}/SBB_all_distances.csv", index=False)
        authorities.to_csv(f"{context.path()}/gtfs_zones.csv", index=False)

    sbb_path     = f"{context.path()}/SBB_all_distances.csv"
    zones_path   = f"{context.path()}/gtfs_zones.csv"
    pricing_path = f"{context.path()}/pricingDescription.xml"

    # Check that the stop ids of the feed are found in gtfs_zones.csv, and add the feed stops it does not find (by their
    # id or their station, as Python and Java look them up) with the zones of their station. Existing rows are kept.

    mode = str(context.config("pt_pricing_rekey_stops")).lower()
    if mode not in ("auto", "always", "never"):
        raise ValueError(f"pt_pricing_rekey_stops must be auto, always or never, not {mode}")

    zones = pd.read_csv(zones_path, dtype = str, keep_default_na = False)
    stops = _read_feed_stops("%s/output/stops.txt" % context.path("data.gtfs.cleaned"))
    coverage = check_stop_ids(zones, stops)

    rekey = mode == "always" or (mode == "auto" and coverage["share"] < context.config("pt_pricing_min_stop_coverage"))
    if rekey:
        rows = rekeyed_rows(zones, stops)
        _append_rows(zones_path, list(zones.columns), rows)
        logger.info(f"  gtfs_zones.csv re-keyed to the feed: {len(rows)} stops added with the zones of their station")
    elif mode == "auto":
        logger.info("  gtfs_zones.csv kept as it is: the feed uses its stop ids")

        # The SNCF stops of the stations of gtfs_zones.csv (the Léman Express in Geneva) all the same: their ids hold no
        # DiDok number, so neither Python nor Java can find their station

        rows = rekeyed_rows(zones, stops[stops["stop_id"].str.match(SNCF_STOP)])
        _append_rows(zones_path, list(zones.columns), rows)
        logger.info(f"  SNCF stops: {len(rows)} added to gtfs_zones.csv with the zones of their station")

    # Give the feed stops in the communes abroad the zone of their commune, as rows of their own.

    communes_path = os.path.join(data_path, context.config("pt_zones_foreign_communes_file"))
    listed_path = os.path.join(data_path, context.config("pt_zones_foreign_zones_file"))
    if os.path.exists(communes_path) and os.path.exists(listed_path):
        zones = pd.read_csv(zones_path, dtype = str, keep_default_na = False)
        rows = foreign_rows(zones, stops, foreign_communes(communes_path, listed_path))
        _append_rows(zones_path, list(zones.columns), rows)
        by_network = pd.Series([r["zones"].split(":")[0] for r in rows], dtype = str).value_counts().to_dict()
        logger.info(f"Stops abroad: {len(rows)} feed stops added to gtfs_zones.csv with the zone of their commune {by_network}")
    else:
        logger.warning(f"Stops abroad: no zones without {communes_path} and {listed_path}")

    # Add the French local networks (data.pt_pricing.french_networks): the zones of the stops of their operators, with
    # their priority, and their fares.

    if os.path.exists(communes_path):
        french_networks.add_column(zones_path, french_networks.PRIORITY_COLUMN)
        columns = list(pd.read_csv(zones_path, nrows = 0).columns)
        rows = french_networks.zone_rows(french_networks.FRENCH_NETWORKS, stops, "%s/output" % context.path("data.gtfs.cleaned"),
                                         communes_path, listed_path, columns)
        _append_rows(zones_path, columns, rows)
        french_networks.add_authorities(pricing_path, french_networks.FRENCH_NETWORKS)
    else:
        logger.warning(f"French networks: not priced without {communes_path}")

    # Locate the feed stops: whether each one is in the area of the Swiss fares, where the GA and the half-fare card hold
    # (the Swiss border, with Liechtenstein and the enclaves of Büsingen and Campione, and the stops abroad in a zone of a
    # Swiss network), rather than guessing it from its id.

    area = swiss_fare_area(context.stage("data.spatial.swiss_border"), data_path)
    zones = pd.read_csv(zones_path, dtype = str, keep_default_na = False)
    foreign = foreign_zones(listed_path) if os.path.exists(listed_path) else set()
    if not foreign:
        logger.warning(f"Stops abroad: no Unireso zones abroad without {listed_path}, all its zones count as Swiss")
    swiss_zoned = swiss_zone_stops(zones, stops, {n.name for n in french_networks.FRENCH_NETWORKS}, foreign)
    located = locate_stops(zones_path, stops, area, swiss_zoned)
    logger.info(f"Feed stops in the area of the Swiss fares: {located['swiss']} of {located['stops']} "
                f"({located['zoned']} abroad in a zone of a Swiss network), {located['added']} added to gtfs_zones.csv without zones")

    # Write the Léman Pass fares as flat tables for eqasim-java.

    lemanpass_paths = write_lemanpass(os.path.join(data_path, context.config("pt_pricing_lemanpass_path")), stops, context.path())

    # Return the paths of the pricing inputs.

    return sbb_path, zones_path, pricing_path, lemanpass_paths


def validate(context):
    """Reruns the stage when an input file of data_path changes (e.g. new fares in pricingDescription.xml)."""
    data_path = context.config("data_path")
    files = [f"{data_path}/pt_pricing/output/pricingDescription.xml"]
    if not context.config("generate_pt_pricing_inputs"):
        files += [f"{data_path}/pt_pricing/output/SBB_all_distances.csv", f"{data_path}/pt_pricing/output/gtfs_zones.csv"]
    files += [os.path.join(data_path, context.config(option)) for option in ("pt_zones_foreign_communes_file", "pt_zones_foreign_zones_file")]
    excel = fare_table_path(os.path.join(data_path, context.config("pt_pricing_lemanpass_path")))
    if excel:
        files.append(excel)
    return [(path, os.path.getsize(path), os.path.getmtime(path)) for path in files if os.path.exists(path)]


def write_lemanpass(folder, stops, output):
    """Writes the Léman Pass tables (utils.gtfs_router.lemanpass.write_java_tables) in output, from the fare table of tpg
    in folder and the pricing inputs of output, and returns their paths (tables with only a header without the fare
    table). stops: the feed
    stops (stop_id, stop_name, stop_lat, stop_lon). The zone of a station is the Unireso zone of its stops in
    gtfs_zones.csv (the lowest, with a warning, when they have several)."""
    excel = fare_table_path(folder)
    if excel is None:
        logger.warning(f"Léman Pass: no fare table of tpg in {folder}, the Léman Pass tables are empty")
        return write_java_tables(None, output)

    pricing = Pricing(output)
    net = SimpleNamespace(stop_ids = stops["stop_id"].to_numpy(), stop_names = stops["stop_name"].to_numpy(),
                          stop_lat = stops["stop_lat"].astype(float).to_numpy(), stop_lon = stops["stop_lon"].astype(float).to_numpy())

    def station_zone(stop_ids):
        zones = sorted({z for s in stop_ids for a, z in pricing._stop_zones(s) if a.id == LEMANPASS_NETWORK})
        if len(zones) > 1:
            logger.warning(f"  Léman Pass: stops {', '.join(stop_ids)} in the zones {', '.join(zones)}; zone {zones[0]} kept")
        return zones[0] if zones else None

    lemanpass = LemanPass(excel, net, station_zone)   # read each time: the stage folder is new on each run
    paths = write_java_tables(lemanpass, output)
    logger.info(f"Léman Pass tables written: {', '.join(os.path.basename(p) for p in paths.values())}")
    return paths


def swiss_fare_area(swiss_border, data_path):
    """The area of the Swiss fares, as one geometry (EPSG:2056): the Swiss border (data.spatial.swiss_border), with the
    other areas of its boundaries file (swissBOUNDARIES3D: Liechtenstein and the enclaves of Büsingen and Campione), where
    the Swiss fares and the half-fare card hold too."""
    area = gpd.GeoSeries(swiss_border, crs = 2056).union_all()
    path = "%s/spatial/country/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp" % data_path
    if os.path.exists(path):
        others = gpd.read_file(path, encoding = "latin1").to_crs(2056)
        area = area.union(others.loc[others["NAME"] != "Schweiz", "geometry"].union_all())
    return area


def foreign_zones(path):
    """The zones abroad of the list of the communes abroad (insee, zone, optionally network, else Unireso), as
    {(network, zone)}: the Unireso zones in France, where the GA and the half-fare card do not hold."""
    listed = pd.read_csv(path, dtype = str).apply(lambda s: s.str.strip())
    network = listed["network"] if "network" in listed else pd.Series(FOREIGN_NETWORK, index = listed.index)
    return set(zip(network, listed["zone"]))


def swiss_zone_stops(zones, stops, foreign_networks, foreign):
    """The feed stops in a zone of a Swiss network, as a set of stop ids: the stops with a zone in gtfs_zones.csv (by their
    id or their station, as eqasim-java looks them up) and none of a foreign network (foreign_networks: the French local
    networks) or of a zone abroad (foreign: {(network, zone)}, the Unireso zones in France). The GA and the half-fare card
    hold in these zones, also abroad (TNW 14: Basel EuroAirport); a stop that also has a zone abroad (Saint-Julien,
    douane: Unireso 10 and 230) is not one of them."""
    zones_of = {}
    for key, text in zip(zones["stop_id"].str.strip(), zones["zones"]):
        for entry in text.split(","):
            parts = [p.strip() for p in entry.split(":")]
            if len(parts) == 2 and parts[0] and parts[1]:
                zones_of.setdefault(key, set()).add((parts[0], parts[1]))

    result = set()
    for stop_id in stops["stop_id"]:
        entries = zones_of.get(stop_id, set()) | zones_of.get(java_station_id(stop_id), set())
        if entries and not any(n in foreign_networks or (n, z) in foreign for n, z in entries):
            result.add(stop_id)
    return result


def locate_stops(path, stops, area, swiss_zoned = frozenset()):
    """Writes whether each feed stop is in the area of the Swiss fares to the LOCATION_COLUMN of gtfs_zones.csv (true /
    false): in area, or in swiss_zoned (the stops in a zone of a Swiss network, swiss_zone_stops). It is written on the
    rows of the stop, or on a row of its own without zones for a stop that has none. The other rows (stations by DiDok
    number) are left empty. Returns the counts {"stops", "swiss", "zoned" (the stops outside area in swiss_zoned), "added"}.

    stops: the feed stops (stop_id, stop_name, stop_lat, stop_lon)."""
    points = gpd.GeoSeries(gpd.points_from_xy(stops["stop_lon"].astype(float), stops["stop_lat"].astype(float)), crs = 4326).to_crs(2056)
    within = dict(zip(stops["stop_id"], points.within(area).to_numpy()))
    inside = {s: bool(w) or s in swiss_zoned for s, w in within.items()}
    zoned = sum(1 for s, w in within.items() if not w and s in swiss_zoned)
    names = dict(zip(stops["stop_id"], stops["stop_name"]))

    with open(path, encoding = "utf-8", newline = "") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    if LOCATION_COLUMN not in header:
        header = header + [LOCATION_COLUMN]
        rows = [row + [""] for row in rows]
    stop_index, location_index = header.index("stop_id"), header.index(LOCATION_COLUMN)

    seen = set()
    for row in rows:
        stop_id = row[stop_index].strip()
        if stop_id in inside:
            row[location_index] = "true" if inside[stop_id] else "false"
            seen.add(stop_id)
    added = []
    for stop_id in stops["stop_id"]:
        if stop_id not in seen:
            row = dict.fromkeys(header, "")
            row.update({"stop_id": stop_id, "stop_name": names[stop_id], "tarif network": "[nan]", "local network": "[nan]",
                        LOCATION_COLUMN: "true" if inside[stop_id] else "false"})
            added.append([row[c] for c in header])
            seen.add(stop_id)

    with open(path, "w", encoding = "utf-8", newline = "") as f:
        writer = csv.writer(f, lineterminator = "\n")
        writer.writerow(header)
        writer.writerows(rows + added)
    return {"stops": len(inside), "swiss": int(sum(inside.values())), "zoned": zoned, "added": len(added)}


def check_stop_ids(zones, stops):
    """How many of the feed stops of priced stations gtfs_zones.csv finds by their id (or their station, the part before
    the first ':' as in eqasim-java), as {"priced": stops of a station with zones, "found": those found, "share"}.

    zones: gtfs_zones.csv (stop_id, zones...); stops: the feed stops (stop_id, station)."""
    keys = _zoned_keys(zones)
    stations = set(_by_station(zones))
    priced = stops["station"].isin(stations)
    found = priced & stops["stop_id"].map(lambda s: s in keys or java_station_id(s) in keys)
    share = found.sum() / priced.sum() if priced.sum() else 1.0

    logger.info(f"Stop ids of the feed in gtfs_zones.csv: {found.sum()} of the {priced.sum()} stops of priced stations "
                f"({share:.1%}), of {len(stops)} stops")
    if priced.sum() == 0:
        logger.warning("  no stop of the feed is in a station of gtfs_zones.csv: check the didok column of the feed")
    return {"priced": int(priced.sum()), "found": int(found.sum()), "share": float(share)}


def rekeyed_rows(zones, stops):
    """The rows to add to gtfs_zones.csv, one per feed stop not found by its id (or station) but in a priced station:
    the row of the station (its DiDok key, else Parent<key>, else one of its platforms), with that stop id and the zones
    of all the rows of the station. A list of dicts with the columns of zones."""
    keys = _zoned_keys(zones)
    by_station = _by_station(zones)

    rows = []
    for stop_id, station in zip(stops["stop_id"], stops["station"]):
        if stop_id in keys or java_station_id(stop_id) in keys or station not in by_station:
            continue
        group = by_station[station]
        first = next((r for r in group if r["stop_id"] == station), None) or \
                next((r for r in group if r["stop_id"] == "Parent" + station), None) or group[0]
        entries = dict.fromkeys(e.strip() for r in group for e in r["zones"].split(",") if e.strip())
        rows.append({**first, "stop_id": stop_id, "zones": ", ".join(entries)})
    return rows


def foreign_rows(zones, stops, abroad):
    """The rows to add to gtfs_zones.csv for the feed stops in the communes abroad (abroad: id, network, zone, geometry,
    one row per commune and zone, as utils.gtfs_router.zones.foreign_communes): one per stop and network, with the zone
    of its commune, for the stops without a zone of that network in gtfs_zones.csv (by their id, station or DiDok
    number). A commune in several zones gives the zone of the nearest stop of the feed that has one of them in
    gtfs_zones.csv, else the first listed. A list of dicts with the columns of zones.

    stops: the feed stops (stop_id, stop_name, station, stop_lat, stop_lon)."""
    if not len(abroad):
        return []
    zones_of = {}
    for key, text in zip(zones["stop_id"].str.strip(), zones["zones"]):
        for entry in text.split(","):
            parts = [p.strip() for p in entry.split(":")]
            if len(parts) == 2 and parts[0] and parts[1]:
                zones_of.setdefault(key, set()).add((parts[0], parts[1]))

    points = gpd.GeoDataFrame(stops[["stop_id", "stop_name", "station"]].copy(), crs = 4326,
                              geometry = gpd.points_from_xy(stops["stop_lon"].astype(float), stops["stop_lat"].astype(float))).to_crs(2056)
    points["known"] = [set().union(*(zones_of.get(k, set()) for k in {s, java_station_id(s), st} if k))
                       for s, st in zip(points["stop_id"], points["station"])]
    hits = gpd.sjoin(points, abroad[["id", "network", "zone", "geometry"]], predicate = "within").drop(columns = "index_right")

    # the local network of the rows of each network, for the new rows (Unireso: ['Genève'])
    local = {}
    if "local network" in zones:
        for text, value in zip(zones["zones"], zones["local network"]):
            for network in {e.split(":")[0].strip() for e in text.split(",") if ":" in e}:
                local.setdefault(network, []).append(value)
    local = {n: pd.Series(v).mode().iloc[0] for n, v in local.items()}

    rows, references = [], {}
    for (stop_id, network), group in hits.groupby(["stop_id", "network"], sort = False):
        first = group.iloc[0]
        if any(n == network for n, _ in first["known"]):
            continue
        listed = sorted(set(group["zone"]))
        zone = listed[0]
        if len(listed) > 1:
            key = (network, tuple(listed))
            if key not in references:
                references[key] = points[[any(n == network and z in listed for n, z in k) for k in points["known"]]]
            refs = references[key]
            if len(refs):
                nearest = refs.distance(first.geometry).idxmin()
                zone = next(z for n, z in sorted(refs.at[nearest, "known"]) if n == network and z in listed)
        row = {c: "" for c in zones.columns}
        row.update({"stop_id": stop_id, "stop_name": first["stop_name"], "tarif network": f"['{network}']",
                    "local network": local.get(network, "[nan]"), "zones": f"{network}:{zone}"})
        rows.append({c: row[c] for c in zones.columns})
    return rows


def _zoned_keys(zones):
    """The stop ids of gtfs_zones.csv that have zones."""
    return set(zones.loc[zones["zones"].str.strip() != "", "stop_id"].str.strip())


def _by_station(zones):
    """The rows of gtfs_zones.csv with zones by station: {DiDok number: [row dicts]} (Parent8578189 and 8578189:0:1 are in
    station 8578189). The ids that are not DiDok numbers (StopArea:OCE..., 0:CHRA of the buses of Annecy) are in no
    station."""
    result = {}
    for row in zones[zones["zones"].str.strip() != ""].to_dict("records"):
        key = row["stop_id"].strip()
        station = java_station_id(key[len("Parent"):] if key.startswith("Parent") else key)
        if _is_didok(station):
            result.setdefault(station, []).append(row)
    return result


def _is_didok(station):
    """Whether a station key is a DiDok number (7 digits: 8501008, 1400001), not the start of another id (0:CHRA -> 0)."""
    return len(station) == 7 and station.isdigit()


def _read_feed_stops(path):
    """The stops of the feed: stop_id, stop_name, station (its DiDok number: the didok column when there is one, else
    from the stop id as utils.gtfs_router.pricing.station_id, or the UIC code of an SNCF stop id; empty when it is not a
    DiDok number), stop_lat, stop_lon."""
    columns = pd.read_csv(path, nrows = 0).columns
    stops = pd.read_csv(path, dtype = str, keep_default_na = False,
                        usecols = [c for c in ("stop_id", "stop_name", "stop_lat", "stop_lon", "didok") if c in columns])
    stops["stop_id"] = stops["stop_id"].str.strip()
    from_id = stops["stop_id"].map(lambda s: m.group(1) if (m := SNCF_STOP.match(s)) else station_id(s))
    if "didok" in stops:
        didok = stops["didok"].str.strip().str.split(".").str[0]
        stops["station"] = didok.where(didok != "", from_id)
    else:
        stops["station"] = from_id
    stops["station"] = stops["station"].where(stops["station"].map(_is_didok), "")
    return stops[["stop_id", "stop_name", "station", "stop_lat", "stop_lon"]]


def _append_rows(path, columns, rows):
    """Appends rows (dicts) to the CSV file, leaving its lines as they are."""
    if not rows:
        return
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
    with open(path, "a", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = columns, lineterminator = "\n")
        writer.writerows(rows)
