"""The fare networks of the French local operators, added to the PT pricing inputs (data.pt_pricing.pt_pricing) so that
they are priced exactly as the Swiss zone networks, in Python and in eqasim-java.

Each network (FRENCH_NETWORKS) is defined by the agencies of the feed that operate it, the fare zone of the stops of
each agency (one zone, or a zone for each commune it serves, by INSEE code with the outlines of communes_france.gpkg)
and its prices in CHF by number of zones. Only the stops served by its agencies get its zones: a train stop in
Thonon-les-Bains is not a STAR'T stop, so a Léman Express ride is never priced as a STAR'T ticket. A stop served by
several agencies of a network gets the zones of each (EVA'D and STAR'T share no stop in the French feeds).

The rows written to gtfs_zones.csv carry the priority of the network (network priority column): LOCAL_PRIORITY, above
the other zone networks (2) and the Z-Pass (1). A ride whose stops are all in two networks (a STAR'T bus in Thonon,
also in Unireso zone 300 abroad) is therefore priced by the local network, while a journey with a Léman Express train
stays with Unireso (and the Léman Pass), the only network common to all its rides.

The prices go to pricingDescription.xml as a zone network: the full prices (priceListByZone halbtax="false"), and the
prices of some ages (priceListByZone with age_range, and zones when they are only sold within some zones: the seniors
of STAR'T). There is no half-fare list, so the half-fare card never applies. The children under 6 travel free, as
everywhere.
"""
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from lxml import etree

# PRIORITY_COLUMN: the column of gtfs_zones.csv with the priority of the network of a row (empty: the usual one), read by
# utils.gtfs_router.pricing and eqasim-java (ZonalReader); LOCAL_PRIORITY: the priority of the French local networks,
# above the other zone networks (2) and the Z-Pass (1)
from utils.gtfs_router.pricing import LOCAL_PRIORITY, PRIORITY_COLUMN

logger = logging.getLogger("synpp")

# Two stops of two operators of a network with the same name closer than this are the same stop (Thonon: STAR'T, EVA'D)
SHARED_STOP_M = 50
# A stop in no commune outline but closer than this to one is in that commune: the stops on the border, whose
# coordinates are a few metres off (TAC: Sous Bossey, 8 m from Bossey, on the Swiss side of the line)
COMMUNE_SNAP_M = 100


@dataclass(frozen = True)
class AgePrices:
    """The prices in CHF by number of zones for the travellers aged low to high (inclusive), on the tickets whose zones
    are all in zones (None: any ticket)."""
    low: int
    high: int
    prices: tuple
    zones: tuple = None


@dataclass(frozen = True)
class ZonesOf:
    """The zones of the stops of an agency: the zone of their commune in another network (the zones abroad of
    pt_zones_foreign_zones_file: Unireso)."""
    network: str


@dataclass(frozen = True)
class FrenchNetwork:
    """A French local fare network: its name in the pricing inputs, the zones of the stops of each agency of the feed
    that operates it ({agency_id: a zone, {INSEE code: zone} by commune, or ZonesOf another network}), and its prices in CHF by number of zones
    (beyond the end of a list, its last price): full, and those of some ages (AgePrices, the first that applies)."""
    name: str
    zones: dict
    prices: tuple
    age_prices: tuple = ()
    priority: int = LOCAL_PRIORITY

    @property
    def agencies(self):
        return tuple(self.zones)


# STAR'T (Thonon-les-Bains) and EVA'D (Évian-les-Bains), one network for their common offers: zones A and B of STAR'T,
# and EVA'D as a third zone. One zone 1.20 EUR (1.11 CHF), two zones (A + B, A + EVA'D) 2.20 EUR (2.03 CHF). Reduced
# fares 1.10 EUR (1.02 CHF) and 2.00 EUR (1.85 CHF): aged 6 to 15 in both networks, and 65 or more in STAR'T only (zones
# A and B; no evidence of it in EVA'D). B + EVA'D is not sold: such a journey crosses zone A.
START_COMMUNES = {
    # zone A: Allinges, Anthy-sur-Léman, Armoy, Cervens, Draillant, Lyaud, Margencel, Orcier, Perrignier, Sciez,
    # Thonon-les-Bains, and the one STAR'T stop in Évian-les-Bains (Lycée Anna de Noailles)
    **dict.fromkeys(("74005", "74013", "74020", "74053", "74106", "74157", "74163", "74206", "74210", "74263",
                     "74281", "74119"), "A"),
    # zone B: Ballaison, Brenthonne, Bons-en-Chablais, Chens-sur-Léman, Douvaine, Excenevex, Fessy, Loisin, Lully,
    # Massongy, Messery, Nernier, Veigy-Foncenex, Yvoire, and Annemasse, Machilly, Saint-Cergues and Ville-la-Grand
    # (Annemasse Agglo, served by STAR'T), and the STAR'T stop in Cranves-Sales (Parc-relais Altéa)
    **dict.fromkeys(("74025", "74048", "74043", "74070", "74105", "74121", "74126", "74150", "74156", "74171",
                     "74180", "74199", "74293", "74315", "74012", "74158", "74229", "74305", "74094"), "B"),
}
START_EVAD = FrenchNetwork(
    name = "StartEvad",
    # STAR'T and EVA'D as the French feeds have them (their copies in the Swiss feed, 87_BUT and 87_EVA, are removed by
    # data.gtfs.cleaned: gtfs_replaced_agencies). Every EVA'D stop is in zone EVAD, also in Thonon-les-Bains.
    zones = {"THONON:Network:1:LOC": START_COMMUNES, "EVAD": "EVAD"},
    prices = (1.11, 2.03),
    age_prices = (AgePrices(6, 15, (1.02, 1.85)), AgePrices(65, 150, (1.02, 1.85), zones = ("A", "B"))),
)

# TAC (Annemasse), with the zones of Unireso: 210 (Annemasse Agglo), 230 (Saint-Julien-en-Genevois and around). One
# zone 1.70 EUR (1.60 CHF, as the other zones of Unireso), 210 + 230 2.50 EUR (2.35 CHF), the same for all ages from 6.
# 210 + 10 (Geneva) is a ticket of the Léman Pass: the TAC stops are all in France.
TAC = FrenchNetwork(
    name = "TAC",
    # TAC Mobilités as the French feed has it (its copy in the Swiss feed, 87_TAC, is removed by data.gtfs.cleaned)
    zones = {"1": ZonesOf("Unireso"), "87_TAC": ZonesOf("Unireso")},
    prices = (1.60, 2.35),
)

# Arvi Mobilités (Cluses, Scionzier, Thyez, Marnaz; agency TMBB, Transdev Mont Blanc Bus): one zone, 1.20 EUR (1.11 CHF)
# for all ages from 6. Its Les Carroz - Flaine lines (LCFE, LCFE2) are assumed to have the same price: none is published
ARVI = FrenchNetwork(
    name = "Arvi",
    zones = {"TMBB": "Arvi"},
    prices = (1.11,),
)

# ProximiTi (La Roche-sur-Foron, Bonneville, Reignier and to Annemasse): one zone, 1.70 EUR (1.60 CHF) for all ages
# from 6
PROXIMITI = FrenchNetwork(
    name = "ProximiTi",
    zones = {"1_m10": "ProximiTi"},
    prices = (1.60,),
)

# SIBRA (Annecy and around): one zone, 2.00 EUR (1.88 CHF) per trip for all ages from 6 (no discount for the children
# or the seniors). Its copy in the Swiss feed (87_SIB) is removed by data.gtfs.cleaned
SIBRA = FrenchNetwork(
    name = "SIBRA",
    zones = {"SIBRA": "SIBRA"},
    prices = (1.88,),
)

FRENCH_NETWORKS = (START_EVAD, TAC, ARVI, PROXIMITI, SIBRA)


def zone_rows(networks, stops, feed_path, communes_path, foreign_zones_path, columns):
    """The rows to add to gtfs_zones.csv for the French networks: one per network and stop served by its agencies,
    with the zones its agencies give it (the zone of the agency, of the commune of the stop, or the zone of its commune
    in another network: ZonesOf). The stops of two operators of a network at the same place (the same name within
    SHARED_STOP_M) get the zones of both, for the transfers; a ride is then priced with the fewest of them
    (utils.gtfs_router.pricing.smallest_zone_cover). A list of dicts with the columns of gtfs_zones.csv (columns), which
    must include PRIORITY_COLUMN.

    stops: the feed stops (stop_id, stop_name, stop_lat, stop_lon); feed_path: the folder of the feed (routes.txt,
    trips.txt, stop_times.txt); communes_path: the outlines of the communes (insee, geometry); foreign_zones_path: the
    zones of the communes abroad in the other networks (insee, zone, optionally network, else Unireso)."""
    agencies = {a for n in networks for a in n.agencies}
    served = served_stops(feed_path, agencies)
    missing = sorted(agencies - set(served))
    if missing:
        logger.warning(f"French networks: agencies without any stop in the feed: {', '.join(missing)}")

    communes = gpd.read_file(communes_path).to_crs(2056)[["insee", "geometry"]]
    points = gpd.GeoDataFrame(stops[["stop_id", "stop_name"]].copy(), crs = 4326,
                              geometry = gpd.points_from_xy(stops["stop_lon"].astype(float), stops["stop_lat"].astype(float))).to_crs(2056)
    commune_of = gpd.sjoin(points, communes, predicate = "within").drop_duplicates("stop_id").set_index("stop_id")["insee"]
    outside = points[~points["stop_id"].isin(commune_of.index)]
    near = gpd.sjoin_nearest(outside, communes, max_distance = COMMUNE_SNAP_M, distance_col = "distance_m")
    near = near.sort_values("distance_m").drop_duplicates("stop_id").set_index("stop_id")["insee"]
    commune_of = pd.concat([commune_of, near])
    names = dict(zip(stops["stop_id"], stops["stop_name"]))
    foreign = _foreign_zones(foreign_zones_path)

    rows = []
    for network in networks:
        zones, unzoned, agency_of = {}, {}, {}   # stop id -> its zones (in the order of the agencies), its agency
        for agency, zone_of in network.zones.items():
            for stop_id in sorted(served.get(agency, ())):
                insee = commune_of.get(stop_id)
                if isinstance(zone_of, str):
                    zone = zone_of
                elif isinstance(zone_of, ZonesOf):
                    zone = foreign.get(zone_of.network, {}).get(insee)
                else:
                    zone = zone_of.get(insee)
                if zone is None:
                    unzoned.setdefault(insee or "outside the communes", set()).add(stop_id)
                elif zone not in zones.setdefault(stop_id, []):
                    zones[stop_id].append(zone)
                agency_of.setdefault(stop_id, agency)
        zones = {s: z for s, z in zones.items() if z}
        shared = _share_zones(zones, agency_of, points)
        for stop_id, listed in zones.items():
            row = {c: "" for c in columns}
            row.update({"stop_id": stop_id, "stop_name": names.get(stop_id, ""), "tarif network": f"['{network.name}']",
                        "local network": "[nan]", "zones": ", ".join(f"{network.name}:{z}" for z in listed),
                        PRIORITY_COLUMN: str(network.priority)})
            rows.append({c: row[c] for c in columns})

        by_zone = pd.Series([" + ".join(z) for z in zones.values()], dtype = str).value_counts().sort_index().to_dict()
        logger.info(f"French network {network.name}: {len(zones)} stops in zones {by_zone}"
                    + (f", {shared} at the same place as a stop of another of its operators" if shared else ""))
        if unzoned:
            logger.warning(f"  {network.name}: {sum(len(s) for s in unzoned.values())} stops of its agencies in no zone, by "
                           f"commune: {', '.join(f'{c} ({len(s)})' for c, s in sorted(unzoned.items()))}")
    return rows


def _share_zones(zones, agency_of, points):
    """Gives the stops of two agencies at the same place (the same name, ignoring case, accents and punctuation, within
    SHARED_STOP_M) the zones of both, in place (zones: stop id -> its zones). Returns the number of stops concerned."""
    located = points[points["stop_id"].isin(zones)].copy()
    located["agency"] = located["stop_id"].map(agency_of)
    located["key"] = located["stop_name"].map(_name_key)
    if located["agency"].nunique() < 2:
        return 0
    pairs = gpd.sjoin(located, located[["stop_id", "agency", "key", "geometry"]], predicate = "dwithin", distance = SHARED_STOP_M)
    pairs = pairs[(pairs["agency_left"] != pairs["agency_right"]) & (pairs["key_left"] == pairs["key_right"])]
    added = {}
    for a, b in zip(pairs["stop_id_left"], pairs["stop_id_right"]):
        added.setdefault(a, []).extend(zones[b])
    for stop_id, extra in added.items():
        zones[stop_id] = list(dict.fromkeys(zones[stop_id] + extra))
    return len(added)


def _name_key(name):
    """A stop name to compare the stops of two operators: lower case, without accents or punctuation."""
    text = unicodedata.normalize("NFKD", str(name).lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in text if not unicodedata.combining(c)))


def _foreign_zones(path):
    """The zone of each commune abroad in each network: {network: {INSEE code: zone}}, from the zones file of the stops
    abroad (insee, zone, optionally network, else Unireso). A commune in several zones keeps the first, with a warning."""
    if not path or not os.path.exists(path):
        return {}
    listed = pd.read_csv(path, dtype = str).apply(lambda c: c.str.strip())
    if "network" not in listed:
        listed["network"] = "Unireso"
    several = listed[listed.duplicated(["network", "insee"], keep = False)]
    if len(several):
        logger.warning(f"  zones abroad: communes in several zones, the first kept: {', '.join(sorted(set(several['commune'])))}")
    listed = listed.drop_duplicates(["network", "insee"])
    return {n: dict(zip(g["insee"], g["zone"])) for n, g in listed.groupby("network")}


def served_stops(feed_path, agencies):
    """The stops served by the trips of each agency of the feed: {agency_id: set of stop ids}."""
    routes = pd.read_csv(f"{feed_path}/routes.txt", dtype = str, usecols = ["route_id", "agency_id"])
    routes = routes[routes["agency_id"].isin(agencies)]
    trips = pd.read_csv(f"{feed_path}/trips.txt", dtype = str, usecols = ["route_id", "trip_id"])
    trips = trips[trips["route_id"].isin(routes["route_id"])].merge(routes, on = "route_id")
    if trips.empty:
        return {}

    # stop_times.txt has several GB: only its trip and stop ids, read with pyarrow, the rows of the trips kept
    table = pacsv.read_csv(f"{feed_path}/stop_times.txt", convert_options = pacsv.ConvertOptions(
        include_columns = ["trip_id", "stop_id"], column_types = {"trip_id": pa.string(), "stop_id": pa.string()}))
    table = table.filter(pc.is_in(table["trip_id"], value_set = pa.array(trips["trip_id"].unique())))
    stop_times = table.to_pandas().drop_duplicates().merge(trips[["trip_id", "agency_id"]], on = "trip_id")
    stop_times["stop_id"] = stop_times["stop_id"].str.strip()
    return {agency: set(group["stop_id"]) for agency, group in stop_times.groupby("agency_id")}


def add_authorities(pricing_path, networks):
    """Writes the French networks as zone networks in pricingDescription.xml, replacing a network of the same name."""
    tree = etree.parse(pricing_path, etree.XMLParser(remove_blank_text = False))
    root = tree.getroot()
    for network in networks:
        for old in root.findall("authority"):
            if old.find("name") is not None and old.find("name").get("value") == network.name:
                logger.warning(f"  pricingDescription.xml: {network.name} replaced by the French networks of the pipeline")
                root.remove(old)
        authority = etree.SubElement(root, "authority")
        etree.SubElement(authority, "name", value = network.name)
        etree.SubElement(authority, "zonalBased", value = "true")
        # no half-fare list: the half-fare card does not apply
        etree.SubElement(authority, "priceListByZone", halbtax = "false", prices = _prices(network.prices))
        for ages in network.age_prices:
            element = etree.SubElement(authority, "priceListByZone", age_range = f"{ages.low}_{ages.high}", prices = _prices(ages.prices))
            if ages.zones:
                element.set("zones", ", ".join(ages.zones))
    etree.indent(root, space = "    ")
    tree.write(pricing_path, encoding = "utf-8")


def add_column(path, name):
    """Adds an empty column to a CSV file (if it is not there yet), leaving the values of its lines as they are."""
    with open(path, encoding = "utf-8") as f:
        lines = f.read().splitlines()
    if not lines or name in pd.read_csv(path, nrows = 0).columns:
        return
    lines = [lines[0] + "," + name] + [line + "," if line else line for line in lines[1:]]
    with open(path, "w", encoding = "utf-8", newline = "") as f:
        f.write("\n".join(lines) + "\n")


def _prices(values):
    """A price list as pricingDescription.xml writes it ("1.11, 2.03")."""
    return ", ".join(f"{v:.2f}" for v in values)
