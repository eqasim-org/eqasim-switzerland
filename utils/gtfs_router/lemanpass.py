"""The Léman Pass: the tickets and subscriptions of the trips across the border around Geneva.

The fares are read from the fare table of tpg, an Excel file in the pricing folder (tpg_<date>_Fare-system_Geneva.xlsx,
the latest when there are several): sheet 1 (Full fares, in CHF) and sheet 3 (PT Fares FR-SW, for the prices in
euros). Reading it takes about 15 s, so what is read is kept in the cache folder (lemanpass_<hash>.pkl) and read again
only when the Excel file changes. Two tables are read (read_fare_table):

- the prices: one row per product, traveller and class: trip (the trip type, e.g. "Zone 300 + Thonon-les-Bains (LEX)
  + Zone 10"), ticket (single, annual, monthly, annual_monthly), profile (adult, child, youth, reduced, child_youth),
  class (1 or 2), chf and eur (eur empty when the table has none). A trip type with two prices in the table (60 of
  them, between Annemasse, Saint-Julien or Bellegarde and Geneva, probably the two lines between them) has two rows;
- the places: the stations and zones of the trip types, with their country (FR or SW).

A Léman Pass product is a list of pieces joined by "+": train stations and zones. It goes from its first end to its
last: with stations, the first end is the pieces up to the first station and the last end those from the last
station; a product with one station puts it at the end with its own zone ("Zone 300 + Thonon-les-Bains (LEX) + Zone
10": Thonon is in Zone 300, the first end). A product of zones only goes from its first zone to its last. An end holds
its stations, its zones and every station inside its zones of Unireso, sold on their own (Genève-Champel is in Zone
10; a town zone like Zone 300, sold only with its station, holds no other station). The price depends on the
trip type alone (not on the trip it is used for), and is the same both ways.

Without the Excel file, the tables written by older versions of this module (lemanpass.csv and lemanpass_places.csv)
are read instead when they are in the pricing folder.

LemanPass(excel_path, net, zones, cache_dir) reads the fare table and finds the cheapest product for a journey of the
router (cheapest_for): the train part of the journey (its rides between two stations of the Léman Pass), and the zones
of the rides before and after it (the first mile and the last mile), which the two ends of the product must hold.
"""
import glob
import hashlib
import logging
import os
import pickle
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")

EXCEL = "tpg_*_Fare-system_Geneva.xlsx"   # the fare table of tpg in the pricing folder (the date of its fares in the name)
CSV_FILES = ("lemanpass.csv", "lemanpass_places.csv")   # the prices and places written by older versions, read without the Excel file
# the tables of write_java_tables, for eqasim-java (see there): the single tickets only
JAVA_FILES = {"products": "lemanpass_products.csv", "stations": "lemanpass_stations.csv"}
JAVA_PROFILES = ("adult", "child", "youth", "reduced")   # the travellers of the single tickets in the table
JAVA_COLUMNS = {"products": ["product_id", "label", "pieces", "end_a", "end_b", "cover"]
                            + [f"{profile}_{c}" for profile in JAVA_PROFILES for c in (2, 1)],
                "stations": ["stop_id", "station", "zone", "country"]}
HOME_ZONE = "10"      # the Unireso zone of Geneva: a journey into any other zone, or to a French station, is abroad
VERSION = 1           # change it when what is read from the fare table changes, so that the old cache files are not used
NETWORK = "Unireso"   # the network of the zones of the Léman Pass in the zone polygons and the stop zones
# the ids of the rail stops: the stations of the Swiss feed (85xxxxx, and 87xxxxx for the French ones) and the SNCF
# train stops (not those of their coaches, OCECar)
RAIL_STOP = r"^(?:Parent)?8[57]\d{5}(?::|$)|OCETrain"
MIN_ZONE_M = 200      # a zone crossed by a ride for less than this does not count (as in the pricing)
HEADER_ROW = 6        # the row of the column names in the sheets of the fare table (from 0)
TICKETS = {"Billet Léman Pass": "single", "Abonnement Léman Pass annuel": "annual", "Abonnement Léman Pass mensuel": "monthly",
           "Abonnement Léman Pass annuel mensualisé": "annual_monthly"}
PROFILES = {"Adult": "adult", "Child (6-15 yrs)": "child", "Youth (16-25 yrs)": "youth", "Reduced Fare": "reduced",
            "Child/Youth (6-25 yrs)": "child_youth"}
CLASSES = {"2nd Class": 2, "1st Class": 1}


# ------------------------------------------------------------------ the fare table of tpg

def fare_table_path(folder):
    """The fare table of tpg in the pricing folder (the latest, by the date in its name, when there are several), or
    None."""
    found = sorted(glob.glob(os.path.join(folder, EXCEL))) if folder else []
    if len(found) > 1:
        logger.info(f"  Léman Pass: {len(found)} fare tables in {folder}, the latest used: {os.path.basename(found[-1])}")
    return found[-1] if found else None


def read_fare_table(excel_path, cache_dir = None):
    """(prices, places) of the fare table of tpg (see the top of the module), from the cache folder when they were read
    from the same file before. excel_path can also be a folder with CSV_FILES (the tables of older versions)."""
    if os.path.isdir(excel_path):
        logger.info(f"  Léman Pass fares from {', '.join(CSV_FILES)} in {excel_path}")
        prices = pd.read_csv(os.path.join(excel_path, CSV_FILES[0]), dtype = {"trip": str, "ticket": str, "profile": str})
        places = pd.read_csv(os.path.join(excel_path, CSV_FILES[1]), dtype = str, keep_default_na = False)
        return prices, places
    cache = None
    if cache_dir:
        stamp = f"{VERSION}|{os.path.abspath(excel_path)}|{os.path.getsize(excel_path)}|{os.path.getmtime(excel_path)}"
        cache = os.path.join(cache_dir, f"lemanpass_{hashlib.md5(stamp.encode()).hexdigest()[:10]}.pkl")
        if os.path.exists(cache):
            logger.info(f"  Léman Pass fares from the cache {cache}")
            with open(cache, "rb") as f:
                return pickle.load(f)

    logger.info(f"  Reading the Léman Pass fares of {excel_path}")
    full, frsw = _sheet(excel_path, 0), _sheet(excel_path, 2)
    full = full.rename(columns = {"Trip type (LEMAN PASS)": "trip", "Validity": "profile_raw"})
    frsw = frsw.rename(columns = {"Trip type": "trip", "Type": "profile_raw"})
    for df in (full, frsw):
        df["trip"] = df["trip"].map(trip_label)
        df["chf"] = pd.to_numeric(df["Prix (CHF)"], errors = "coerce")
    key = ["Departure", "Destination", "trip", "Details", "profile_raw", "Class", "chf"]
    euros = frsw.assign(eur = pd.to_numeric(frsw["Prix (€)"], errors = "coerce")).groupby(key)["eur"].first()

    rows = full[(full["Pricing union"] == "LEMAN PASS") & full["Details"].isin(TICKETS) & full["trip"].notna()].join(euros, on = key)
    rows = rows.assign(ticket = rows["Details"].map(TICKETS), profile = rows["profile_raw"].map(PROFILES), **{"class": rows["Class"].map(CLASSES)})
    rows = rows.dropna(subset = ["profile", "class", "chf"])
    prices = (rows[["trip", "ticket", "profile", "class", "chf", "eur"]].drop_duplicates(["trip", "ticket", "profile", "class", "chf"])
              .astype({"class": int}).sort_values(["trip", "ticket", "profile", "class", "chf"]).reset_index(drop = True))

    countries = pd.concat([full[["Departure", "Departure country"]].set_axis(["place", "country"], axis = 1),
                           full[["Destination", "Destination country"]].set_axis(["place", "country"], axis = 1)])
    # the country of each piece, by its name without "(LEX)" (the trip types write "Pringy (LEX)", the departures "Pringy")
    country_of = countries.dropna().assign(key = lambda d: d["place"].map(station_key)).drop_duplicates("key").set_index("key")["country"]
    pieces = sorted({p for t in prices["trip"].unique() for p in t.split(" + ")})
    places = pd.DataFrame({"place": pieces, "country": [country_of.get(station_key(p), "") for p in pieces]})
    places["kind"] = np.where(places["place"].str.startswith("Zone "), "zone", "station")
    places = places.sort_values(["kind", "place"])[["place", "kind", "country"]].reset_index(drop = True)
    logger.info(f"  {len(prices)} prices of {prices['trip'].nunique()} trip types, {len(places)} places")

    if cache:
        os.makedirs(cache_dir, exist_ok = True)
        with open(cache, "wb") as f:
            pickle.dump((prices, places), f, protocol = pickle.HIGHEST_PROTOCOL)
    return prices, places


def _sheet(path, index):
    """A sheet of the fare table as strings, with the column names of its header row and the cells stripped."""
    df = pd.read_excel(path, sheet_name = index, header = HEADER_ROW, dtype = str)
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


def trip_label(text):
    """A trip type with its pieces separated by " + " (the table has line breaks and missing spaces), or None."""
    if not isinstance(text, str) or not text:
        return None
    return " + ".join(part.strip() for part in re.sub(r"\s+", " ", text).split("+"))


def station_key(name):
    """A name to match the stations of the table with the stops: lower case, without accents, "(LEX)", spaces or
    punctuation."""
    name = unicodedata.normalize("NFKD", str(name).replace("(LEX)", "").lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in name if not unicodedata.combining(c)))


# ------------------------------------------------------------------ the products for a journey

class LemanPass:
    """The Léman Pass products, and the cheapest one for a journey.

    excel_path: the fare table of tpg (fare_table_path), or a folder with CSV_FILES; net: the Network (for the
    stations: its stop_ids, stop_names, stop_lat and stop_lon); zones: the zone polygons (tracks.zones), whose zones of
    NETWORK are those of the Léman Pass, or a function giving the zone of NETWORK of a station from the ids of its stops
    (a string, None if none; without the polygons, from the stop zones of the pricing); cache_dir: where what is read
    from the fare table is kept (None: it is read each time)."""

    def __init__(self, excel_path, net, zones, cache_dir = None):
        self.prices, places = read_fare_table(excel_path, cache_dir)
        self.country = dict(zip(places["place"], places["country"]))
        stations = places.loc[places["kind"] == "station", "place"]

        # the rail stops of each station of the table (by name), and the zone each station is in
        by_key = {station_key(s): s for s in stations}
        keys = pd.Series(net.stop_names).map(station_key)
        rail = pd.Series(net.stop_ids.astype(str)).str.contains(RAIL_STOP)
        match = keys.isin(by_key) & rail
        self.station_of_stop = {int(s): by_key[k] for s, k in zip(np.flatnonzero(match.to_numpy()), keys[match])}
        found = pd.DataFrame({"stop": list(self.station_of_stop), "station": list(self.station_of_stop.values())}).drop_duplicates("station")
        if callable(zones):
            stops_of = {}
            for stop, station in self.station_of_stop.items():
                stops_of.setdefault(station, []).append(str(net.stop_ids[stop]))
            self.station_zone = {station: zone for station, ids in stops_of.items() if (zone := zones(ids)) is not None}
        else:
            polygons = zones[zones["network"] == NETWORK][["zone", "geometry"]]
            points = gpd.GeoDataFrame(found, geometry = gpd.points_from_xy(net.stop_lon[found["stop"]], net.stop_lat[found["stop"]]), crs = 4326).to_crs(polygons.crs)
            inside = gpd.sjoin(points, polygons, predicate = "within").drop_duplicates("station")
            self.station_zone = dict(zip(inside["station"], inside["zone"]))
        self.stop_ids = {stop: str(net.stop_ids[stop]) for stop in self.station_of_stop}
        missing = sorted(set(stations) - set(found["station"]))
        if missing:
            logger.warning(f"  Léman Pass stations without a stop of that name: {', '.join(missing)}")

        # the zones of the networks, sold on their own (Zone 10 + Zone 210), whose tickets include the trains between
        # their stations; not the zones of the towns, sold only with their station (Zone 300 + Thonon-les-Bains (LEX))
        trips = self.prices["trip"].unique()
        self.network_zones = {p.split()[1] for t in trips if all(p.startswith("Zone ") for p in t.split(" + ")) for p in t.split(" + ")}
        # the two ends of each product: the places each holds
        self.ends = {t: self._ends(t) for t in trips}
        self.pieces = {t: set(t.split(" + ")) for t in self.ends}
        # the products of zones only (Zone 10 + Zone 230): valid on everything inside their zones, the places they hold
        self.zone_cover = {t: self._holds(t.split(" + ")) for t in self.ends if all(p.startswith("Zone ") for p in t.split(" + "))}
        self._price_lists = {}   # (ticket, travellers, class) -> {trip type: price}, made when first needed
        logger.info(f"  Léman Pass: {len(self.ends)} trip types, {len(found)} of {len(stations)} stations found among the stops")

    def _holds(self, end):
        """The places an end of a product holds: its stations, its zones and the stations inside its network zones
        (Genève-Champel in Zone 10, but not Évian-les-Bains in Zone 300, the buses of Thonon)."""
        zones = {piece.split()[1] for piece in end if piece.startswith("Zone ")} & self.network_zones
        return set(end) | {s for s, z in self.station_zone.items() if z in zones}

    def _ends(self, label):
        """The two ends of a product, (first, last), as the sets of places they hold."""
        pieces = label.split(" + ")
        stations = [k for k, piece in enumerate(pieces) if not piece.startswith("Zone ")]
        if not stations:
            start, end = ([pieces[0]], [pieces[-1]]) if len(pieces) > 1 else (pieces, pieces)
        elif stations[0] != stations[-1]:
            start, end = pieces[:stations[0] + 1], pieces[stations[-1]:]
        else:
            k = stations[0]
            before, station, after = pieces[:k], pieces[k], pieces[k + 1:]
            own = f"Zone {self.station_zone.get(station)}"
            if not before:
                start, end = [station], after
            elif not after:
                start, end = before, [station]
            elif own in before:
                start, end = before + [station], after
            else:
                start, end = before, [station] + after
        return self._holds(start), self._holds(end)

    def needs(self, journey, net, pricing):
        """What a product must hold for a journey: (first, last, through), first and last the places its two ends must
        hold and through the zones its pieces must include (empty with a train part), or None when the journey does not
        cross into the Léman Pass area abroad (nothing in France: no French station, no zone other than Zone 10).

        With a train part (rides between two stations of the Léman Pass): its first boarding station and the zones of
        the rides before it, and its last alighting station and the zones of the rides after it. Without (buses and
        trams only): the zone where the journey starts and the zone where it ends, and every zone its rides cross."""
        rides = journey.rides
        if not rides:
            return None
        zones_of = lambda rs: {z for r in rs for z in self._ride_zones(net, pricing, r)}
        train = self._train(rides, net)
        if train:
            first = {self.station_of_stop[rides[train[0]].stops[0]]} | zones_of(rides[:train[0]])
            last = {self.station_of_stop[rides[train[-1]].stops[-1]]} | zones_of(rides[train[-1] + 1:])
            through = set()
        else:
            ends = self._end_zones(rides, net, pricing)
            if ends is None:
                return None
            (first, last), through = ends, zones_of(rides)
        return (first, last, through) if self._abroad(first | last | through) else None

    def single_tickets(self, rides, wholes, net, pricing, profiles = ("adult",), travel_class = 2):
        """The Léman Pass single tickets a journey can use: a list of (start, end, trip type, price in CHF), each a
        ticket for the rides rides[start:end], the others to be priced as usual; empty when none fits or the journey
        does not cross into the Léman Pass area abroad (see needs).

        rides: the rides of the router; wholes: their RideZones (Pricing.ride), a ride being in the area when all its
        stops are in zones of NETWORK; profiles: the travellers whose tickets can be bought (single_profiles), the
        cheapest of them counting.

        With a train part (its first and last rides between two stations of the Léman Pass), a ticket covers it and the
        rides in the area right before and after it (the first mile and the last mile, as in needs), or only some of
        them: one ticket for each choice, since the rides left out may cost less on their own. The rides inside the
        train part that are not trains (a tram between two stations in Geneva) must stay in the area, and their zones
        be pieces of the ticket. Without a train part, a ticket covers all the rides, if they are all in the area."""
        if not rides:
            return []
        inside = [NETWORK in w.zones for w in wholes]
        zones_of = lambda ks: {z for k in ks for z in self._ride_zones(net, pricing, rides[k], wholes[k])}
        train = self._train(rides, net)
        choices = []   # (start, end, first, last, through)
        if train:
            a, b = train[0], train[-1] + 1
            middle = [k for k in range(a, b) if k not in train]
            if not all(inside[k] for k in middle):
                return []
            through = zones_of(middle)
            start, end = a, b
            while start > 0 and inside[start - 1]:
                start -= 1
            while end < len(rides) and inside[end]:
                end += 1
            board, alight = self.station_of_stop[rides[a].stops[0]], self.station_of_stop[rides[b - 1].stops[-1]]
            for s, e in dict.fromkeys([(start, end), (a, end), (start, b), (a, b)]):
                choices.append((s, e, {board} | zones_of(range(s, a)), {alight} | zones_of(range(b, e)), through))
        elif all(inside):
            ends = self._end_zones(rides, net, pricing)
            if ends is not None:
                choices.append((0, len(rides), *ends, zones_of(range(len(rides)))))
        tickets = []
        for s, e, first, last, through in choices:
            if self._abroad(first | last | through):
                product = self.cheapest(first, last, through, "single", profiles, travel_class)
                if product is not None:
                    tickets.append((s, e, *product))
        return tickets

    def _train(self, rides, net):
        """The indexes of the train rides between two stations of the Léman Pass."""
        rail = lambda r: net.routes["mode"].iat[net.trip_route[r.trip]] == "rail"
        return [k for k, r in enumerate(rides) if rail(r) and r.stops[0] in self.station_of_stop and r.stops[-1] in self.station_of_stop]

    @staticmethod
    def _end_zones(rides, net, pricing):
        """({the zone where the journey starts}, {the zone where it ends}) for a journey without a train part, a stop on
        the border of two zones giving one of them, or None when a stop is in no zone of NETWORK."""
        ends = []
        for stop in (rides[0].stops[0], rides[-1].stops[-1]):
            zones = {f"Zone {z}" for a, z in pricing._stop_zones(str(net.stop_ids[stop])) if a.id == NETWORK}
            if not zones:
                return None
            ends.append({min(zones)})
        return tuple(ends)

    def _abroad(self, places):
        """Whether places reach into the Léman Pass area abroad: a French station, or a zone other than Zone 10."""
        return any(self._place_abroad(p) for p in places)

    def _place_abroad(self, place):
        """Whether a place is in the Léman Pass area abroad: a French station, or a zone other than Zone 10."""
        return place != f"Zone {HOME_ZONE}" if place.startswith("Zone ") else self.country.get(place) == "FR"

    @staticmethod
    def _ride_zones(net, pricing, ride, whole = None):
        """The zones of NETWORK a ride is in, as "Zone 10": those of the pricing (whole: its RideZones, if known), and
        those its track crosses for at least MIN_ZONE_M (a ride with a stop in no zone has none in the pricing, but its
        track still crosses zones)."""
        zones = set((whole or pricing.ride(net, ride)).zones.get(NETWORK, []))
        if pricing.tracks is not None:
            zones |= {z for z, m in pricing.tracks.crossed(net, ride).get(NETWORK, {}).items() if m >= MIN_ZONE_M}
        return {f"Zone {z}" for z in zones}

    def cheapest(self, first, last, through = frozenset(), ticket = "annual", profile = "adult", travel_class = 2):
        """The cheapest product whose ends hold first and last (in either direction) and whose pieces include the zones
        through, or a product of zones only whose zones hold all of them (Genève - Lancy-Bachet by train, then the bus to
        Saint-Julien: Zone 10 + Zone 230): (trip type, price in CHF), or None. profile: a traveller, or several (the cheapest of them for each
        product). A trip type with two prices counts with the lower one."""
        fits = [(p, t) for t, p in self._prices(ticket, profile, travel_class).items() if through <= self.pieces[t]
                and ((first <= self.ends[t][0] and last <= self.ends[t][1]) or (first <= self.ends[t][1] and last <= self.ends[t][0])
                     or (t in self.zone_cover and set(first) | set(last) | set(through) <= self.zone_cover[t]))]
        if not fits:
            return None
        price, trip = min(fits)
        return trip, float(price)

    def _prices(self, ticket, profile, travel_class):
        """{trip type: its lowest price in CHF} of a ticket, for a traveller or the cheapest of several, and a class."""
        key = (ticket, (profile,) if isinstance(profile, str) else tuple(sorted(profile)), travel_class)
        if key not in self._price_lists:
            p = self.prices
            p = p[(p["ticket"] == ticket) & p["profile"].isin(key[1]) & (p["class"] == travel_class)]
            self._price_lists[key] = {t: v for t, v in p.groupby("trip")["chf"].min().items() if t in self.ends}
        return self._price_lists[key]

    def cheapest_for(self, journey, net, pricing, ticket = "annual", profile = "adult", travel_class = 2):
        """The cheapest product for a journey of the router, (trip type, price in CHF), or None when the journey does not
        need the Léman Pass or no product covers it (see needs)."""
        needed = self.needs(journey, net, pricing)
        return self.cheapest(*needed, ticket, profile, travel_class) if needed else None


    def java_tables(self):
        """The single tickets of the Léman Pass as two tables for eqasim-java (JAVA_FILES, see write_java_tables):
        {name: DataFrame}."""

        # the lowest single price of each product, traveller and class, one column each
        single = self.prices[(self.prices["ticket"] == "single") & self.prices["trip"].isin(self.ends) & self.prices["profile"].isin(JAVA_PROFILES)]
        single = single.groupby(["trip", "profile", "class"])["chf"].min()
        prices = single.unstack(["profile", "class"])
        prices.columns = [f"{profile}_{c}" for profile, c in prices.columns]
        prices = prices.reindex(columns = JAVA_COLUMNS["products"][6:]).map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

        # the places of each product, as "station:<name>" or "zone:<number>" separated by "|"
        token = lambda p: f"zone:{p.split()[1]}" if p.startswith("Zone ") else f"station:{p}"
        places = lambda ps, ordered = False: "|".join(token(p) for p in (ps if ordered else sorted(ps)))
        trips = sorted(prices.index)
        products = pd.DataFrame({"product_id": range(1, len(trips) + 1), "label": trips,
                                 "pieces": [places(t.split(" + "), True) for t in trips],
                                 "end_a": [places(self.ends[t][0]) for t in trips], "end_b": [places(self.ends[t][1]) for t in trips],
                                 "cover": [places(self.zone_cover[t]) if t in self.zone_cover else "" for t in trips]})
        products = pd.concat([products, prices.loc[trips].reset_index(drop = True)], axis = 1)[JAVA_COLUMNS["products"]]

        # the rail stops of each station, with its zone and country
        stations = pd.DataFrame({"stop_id": [self.stop_ids[s] for s in self.station_of_stop], "station": list(self.station_of_stop.values())})
        stations["zone"] = stations["station"].map(lambda st: self.station_zone.get(st, ""))
        stations["country"] = stations["station"].map(lambda st: self.country.get(st, ""))
        stations = stations.sort_values(["station", "stop_id"]).reset_index(drop = True)[JAVA_COLUMNS["stations"]]
        return {"products": products, "stations": stations}


def write_java_tables(lemanpass, folder):
    """Writes the single tickets of the Léman Pass in folder as two CSV files for eqasim-java (UTF-8, comma-separated,
    with a header) and returns their paths, {name: path} (names of JAVA_FILES). With lemanpass None (no fare table),
    the files have only their header: no Léman Pass.

    - lemanpass_products.csv: one row per product (trip type of the fare table) with a single ticket. product_id;
      label (the trip type, for reading); pieces, end_a, end_b and cover, lists of places separated by "|", a place
      being "station:<name>" (a station of lemanpass_stations.csv) or "zone:<number>" (a Unireso zone, as in
      gtfs_zones.csv): pieces are the pieces of the product, end_a and end_b the places its two ends hold (the stations
      inside its zones already listed), cover (only for a product of zones only) all the places it covers; then the
      price in CHF of each traveller and class, <profile>_<class>: adult, child (6-15), youth (16-25) and reduced (with
      the half-fare card), in 2nd and 1st class (adult_2, adult_1...), empty when the table has none (the lower price
      when it has two);
    - lemanpass_stations.csv: one row per rail stop of a station of the Léman Pass: stop_id (the ids of the GTFS feed
      and of the transit schedule), station, zone (its Unireso zone) and country (FR or SW).

    A journey needs the Léman Pass when a place of it is abroad: a station with country FR, or a zone other than 10.
    A product fits it when the zones of the rides inside its train part are pieces and either its end_a holds the
    first places and its end_b the last ones (or the reverse), or its cover holds all of them. The train part runs from
    the first to the last rail ride between two stops of lemanpass_stations.csv; the first places are its boarding
    station and the zones of the rides before it, the last places its alighting station and the zones of the rides
    after it. Without a train part (buses and trams only), the first and last places are the zones of the first and
    last stops, and every zone of the rides must be a piece. The cheapest fitting product for the traveller counts
    (see LemanPass.needs, single_tickets and cheapest)."""
    os.makedirs(folder, exist_ok = True)
    paths = {}
    tables = lemanpass.java_tables() if lemanpass is not None else {n: pd.DataFrame(columns = c) for n, c in JAVA_COLUMNS.items()}
    for name, table in tables.items():
        paths[name] = os.path.join(folder, JAVA_FILES[name])
        table.to_csv(paths[name], index = False, lineterminator = "\n")
    return paths


def single_profiles(age, halbtax = False):
    """The travellers of the single tickets someone can buy (the cheapest counts): adult, child from 6 to 15, youth from
    16 to 25, and reduced with the half-fare card (the table has reduced prices only for a few trip types)."""
    profiles = ["adult"]
    if 6 <= age <= 15:
        profiles.append("child")
    if 16 <= age <= 25:
        profiles.append("youth")
    if halbtax:
        profiles.append("reduced")
    return tuple(profiles)


def subscription_profile(age):
    """The traveller of the Léman Pass subscriptions for an age: child_youth from 6 to 25, else adult."""
    return "child_youth" if age is not None and not pd.isna(age) and 6 <= age <= 25 else "adult"


def load(folder, net, zones, cache_dir = None):
    """The LemanPass of the fare table of tpg in the pricing folder, or None (with a message) when it is not there.
    net: the Network of the whole feed (its stops are those of every date); zones: the zone polygons (tracks.zones);
    cache_dir: where what is read from the fare table is kept."""
    excel = fare_table_path(folder)
    if excel:
        return LemanPass(excel, net, zones, cache_dir)
    if folder and all(os.path.exists(os.path.join(folder, f)) for f in CSV_FILES):
        return LemanPass(folder, net, zones, cache_dir)
    logger.info(f"No Léman Pass: no fare table of tpg ({EXCEL}) nor {' and '.join(CSV_FILES)} in {folder}")
    return None
