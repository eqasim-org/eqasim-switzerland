"""The Léman Pass: the tickets and subscriptions of the trips across the border around Geneva.

The fares come from the fare table of tpg (an Excel file), turned once into two inputs of the pricing folder:

    python -m utils.gtfs_router.lemanpass ../LemanPassPrices/data/tpg_250424_Fare-system_Geneva.xlsx data/pricing

- lemanpass.csv: one row per product, traveller and class: trip (the trip type, e.g. "Zone 300 + Thonon-les-Bains
  (LEX) + Zone 10"), ticket (single, annual, monthly, annual_monthly), profile (adult, child, youth, reduced,
  child_youth), class (1 or 2), chf and eur (eur empty when the table has none). A trip type with two prices in the
  table (60 of them, between Annemasse, Saint-Julien or Bellegarde and Geneva, probably the two lines between them)
  has two rows;
- lemanpass_places.csv: the stations and zones of the trip types, with their country (FR or SW).

A Léman Pass product is a list of pieces joined by "+": train stations and zones. It goes from its first end to its
last: with stations, the first end is the pieces up to the first station and the last end those from the last
station; a product with one station puts it at the end with its own zone ("Zone 300 + Thonon-les-Bains (LEX) + Zone
10": Thonon is in Zone 300, the first end). A product of zones only goes from its first zone to its last. An end holds
its stations, its zones and every station inside these zones (Genève-Champel is in Zone 10). The price depends on the
trip type alone (not on the trip it is used for), and is the same both ways.

LemanPass(folder, net, zones) reads these inputs and finds the cheapest product for a journey of the router
(cheapest_for): the train part of the journey (its rides between two stations of the Léman Pass), and the zones of the
rides before and after it (the first mile and the last mile), which the two ends of the product must hold.
"""
import argparse
import logging
import os
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger("synpp")

FILES = ("lemanpass.csv", "lemanpass_places.csv")
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


# ------------------------------------------------------------------ the inputs, from the fare table of tpg

def prepare(excel_path, folder):
    """Writes lemanpass.csv and lemanpass_places.csv in folder from the fare table of tpg: sheet 1 (Full fares, in
    CHF) and sheet 3 (PT Fares FR-SW, for the prices in euros)."""
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
    table = (rows[["trip", "ticket", "profile", "class", "chf", "eur"]].drop_duplicates(["trip", "ticket", "profile", "class", "chf"])
             .astype({"class": int}).sort_values(["trip", "ticket", "profile", "class", "chf"]))

    countries = pd.concat([full[["Departure", "Departure country"]].set_axis(["place", "country"], axis = 1),
                           full[["Destination", "Destination country"]].set_axis(["place", "country"], axis = 1)])
    # the country of each piece, by its name without "(LEX)" (the trip types write "Pringy (LEX)", the departures "Pringy")
    country_of = countries.dropna().assign(key = lambda d: d["place"].map(station_key)).drop_duplicates("key").set_index("key")["country"]
    pieces = sorted({p for t in table["trip"].unique() for p in t.split(" + ")})
    places = pd.DataFrame({"place": pieces, "country": [country_of.get(station_key(p)) for p in pieces]})
    places["kind"] = np.where(places["place"].str.startswith("Zone "), "zone", "station")

    os.makedirs(folder, exist_ok = True)
    table.to_csv(os.path.join(folder, FILES[0]), index = False)
    places.sort_values(["kind", "place"])[["place", "kind", "country"]].to_csv(os.path.join(folder, FILES[1]), index = False)
    logger.info(f"{len(table)} prices of {table['trip'].nunique()} trip types and {len(places)} places written to {folder}")


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

    folder: the pricing folder with lemanpass.csv and lemanpass_places.csv; net: the Network (for the stations);
    zones: the zone polygons (tracks.zones), whose zones of NETWORK are those of the Léman Pass."""

    def __init__(self, folder, net, zones):
        self.prices = pd.read_csv(os.path.join(folder, FILES[0]), dtype = {"trip": str, "ticket": str, "profile": str})
        places = pd.read_csv(os.path.join(folder, FILES[1]), dtype = str, keep_default_na = False)
        self.country = dict(zip(places["place"], places["country"]))
        stations = places.loc[places["kind"] == "station", "place"]

        # the rail stops of each station of the table (by name), and the zone each station is in
        by_key = {station_key(s): s for s in stations}
        keys = pd.Series(net.stop_names).map(station_key)
        rail = pd.Series(net.stop_ids.astype(str)).str.contains(RAIL_STOP)
        match = keys.isin(by_key) & rail
        self.station_of_stop = {int(s): by_key[k] for s, k in zip(np.flatnonzero(match.to_numpy()), keys[match])}
        found = pd.DataFrame({"stop": list(self.station_of_stop), "station": list(self.station_of_stop.values())}).drop_duplicates("station")
        polygons = zones[zones["network"] == NETWORK][["zone", "geometry"]]
        points = gpd.GeoDataFrame(found, geometry = gpd.points_from_xy(net.stop_lon[found["stop"]], net.stop_lat[found["stop"]]), crs = 4326).to_crs(polygons.crs)
        inside = gpd.sjoin(points, polygons, predicate = "within").drop_duplicates("station")
        self.station_zone = dict(zip(inside["station"], inside["zone"]))
        missing = sorted(set(stations) - set(found["station"]))
        if missing:
            logger.warning(f"  Léman Pass stations without a stop of that name: {', '.join(missing)}")

        # the two ends of each product: the places each holds
        self.ends = {t: self._ends(t) for t in self.prices["trip"].unique()}
        # the products of zones only (Zone 10 + Zone 230): valid on everything inside their zones, the places they hold
        self.zone_cover = {t: self._holds(t.split(" + ")) for t in self.ends if all(p.startswith("Zone ") for p in t.split(" + "))}
        logger.info(f"  Léman Pass: {len(self.ends)} trip types, {len(found)} of {len(stations)} stations found among the stops")

    def _holds(self, end):
        """The places an end of a product holds: its stations, its zones and the stations inside these zones."""
        zones = {piece.split()[1] for piece in end if piece.startswith("Zone ")}
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
        rail = lambda r: net.routes["mode"].iat[net.trip_route[r.trip]] == "rail"
        station = self.station_of_stop.get
        zones_of = lambda rs: {f"Zone {z}" for r in rs for z in self._ride_zones(net, pricing, r)}
        stop_zones = lambda s: {f"Zone {z}" for a, z in pricing._stop_zones(str(net.stop_ids[s])) if a.id == NETWORK}
        train = [k for k, r in enumerate(rides) if rail(r) and station(r.stops[0]) and station(r.stops[-1])]
        if train:
            first = {station(rides[train[0]].stops[0])} | zones_of(rides[:train[0]])
            last = {station(rides[train[-1]].stops[-1])} | zones_of(rides[train[-1] + 1:])
        else:
            first, last = stop_zones(rides[0].stops[0]), stop_zones(rides[-1].stops[-1])
            if not first or not last:
                return None
            first, last = {min(first)}, {min(last)}   # a stop on the border of two zones: one of them
        through = zones_of(rides) if not train else set()
        places = first | last | through
        abroad = any(p.startswith("Zone ") and p != "Zone 10" for p in places) or any(self.country.get(p) == "FR" for p in places)
        return (first, last, through) if abroad else None

    @staticmethod
    def _ride_zones(net, pricing, ride):
        """The zones of NETWORK a ride is in: those of the pricing, and those its track crosses for at least
        MIN_ZONE_M (a ride with a stop in no zone has none in the pricing, but its track still crosses zones)."""
        zones = set(pricing.ride(net, ride).zones.get(NETWORK, []))
        if pricing.tracks is not None:
            zones |= {z for z, m in pricing.tracks.crossed(net, ride).get(NETWORK, {}).items() if m >= MIN_ZONE_M}
        return zones

    def cheapest(self, first, last, through = frozenset(), ticket = "annual", profile = "adult", travel_class = 2):
        """The cheapest product whose ends hold first and last (in either direction) and whose pieces include the zones
        through, or a product of zones only whose zones hold all of them (Genève - Lancy-Bachet by train, then the bus to
        Saint-Julien: Zone 10 + Zone 230): (trip type, price in CHF), or None. A trip type with two prices counts with the
        lower one."""
        prices = self.prices[(self.prices["ticket"] == ticket) & (self.prices["profile"] == profile) & (self.prices["class"] == travel_class)]
        prices = prices.groupby("trip")["chf"].min()
        fits = [(p, t) for t, p in prices.items() if t in self.ends and through <= set(t.split(" + "))
                and ((first <= self.ends[t][0] and last <= self.ends[t][1]) or (first <= self.ends[t][1] and last <= self.ends[t][0])
                     or (t in self.zone_cover and set(first) | set(last) | set(through) <= self.zone_cover[t]))]
        if not fits:
            return None
        price, trip = min(fits)
        return trip, float(price)

    def cheapest_for(self, journey, net, pricing, ticket = "annual", profile = "adult", travel_class = 2):
        """The cheapest product for a journey of the router, (trip type, price in CHF), or None when the journey does not
        need the Léman Pass or no product covers it (see needs)."""
        needed = self.needs(journey, net, pricing)
        return self.cheapest(*needed, ticket, profile, travel_class) if needed else None


def subscription_profile(age):
    """The traveller of the Léman Pass subscriptions for an age: child_youth from 6 to 25, else adult."""
    return "child_youth" if age is not None and not pd.isna(age) and 6 <= age <= 25 else "adult"


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("excel", help = "The fare table of tpg (tpg_..._Fare-system_Geneva.xlsx)")
    parser.add_argument("folder", nargs = "?", default = "data/pricing", help = "The pricing folder written to")
    args = parser.parse_args()
    logging.basicConfig(level = logging.INFO, format = "%(message)s")
    prepare(args.excel, args.folder)


if __name__ == "__main__":
    main()
