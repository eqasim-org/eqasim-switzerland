"""The price of a journey under the Swiss public transport fares.

A Swiss journey can mix SBB trains, priced by distance, the zones of regional fare networks (Verbünde: ZVV, Unireso...)
and rides in no network. There is no single price list, so the ticket is rebuilt in four steps (Pricing.price):

1. Which fare networks each ride belongs to: those shared by all its stops (gtfs_zones.csv, plus "SBB" for the
   stations of SBB_all_distances.csv). With the tracks, its zones are those its track crosses (track_zones).
2. Which network prices which rides: a network covering all the rides prices the whole journey (the one of the highest
   priority: regional networks 2, Z-Pass 1, SBB 0); otherwise each ride goes to a network of its highest priority it
   shares with another ride, or else to its own (group_rides).
3. The price in that network: by number of zones, by SBB tariff kilometres, or by distance for the rides in no network
   (the calculators: zone_price, sbb_price, default_price).
   With the Léman Pass (Pricing.lemanpass, see utils.gtfs_router.lemanpass), a journey into its area abroad can also take a
   Léman Pass single ticket for its train part and the rides around it, the other rides priced as above: the cheaper
   of the two ways is kept. With the Léman Pass, Unireso only sells the tickets of Zone 10: a way of paying with a
   Unireso ticket into its zones abroad is not kept, unless no Léman Pass ticket fits the journey (_lemanpass_zones).
4. Free travel (GA, under 6, Junior card, Gleis 7, regional and zone subscriptions), half fare, 1st class, and the
   discounts (by age, for students, for the people living or studying in a network's area). The half-fare card is a
   Swiss card: a distance-based ticket with a stop abroad (a French train or bus) has no half fare with it, nor a zone
   network ticket all abroad; a zone network ticket into Switzerland keeps it (tpg sells Unireso Zone 10 + Zone 210 at
   the reduced fare), and so does SBB, whose distance table only has places with the Swiss tariff. Under 16, half fare
   holds everywhere (and the Léman Pass has its own child and youth prices).

The fares themselves are read from the files of data/pricing (pricingDescription.xml, gtfs_zones.csv,
SBB_all_distances.csv), so they can be changed without touching the code. Pricing.fares gives them as JSON and
Pricing.with_changes applies changes to them: this is what the price editor of the map uses.

This is a port of the fare model of eqasim-java for Switzerland (ch_cmdp). For those who know the Java code:
RunComputeTransitPrices.processRequest is Pricing.price, SwissPtRoutePredictor is Pricing.ride_zones,
SwissPtVariables is Pricing.group_rides, ZoneBasedPricing, SBBPtStageCostCalculator and DefaultDistanceBasedCalculator
are the calculators, and the readers of utils/pricing/inputs are Pricing.__init__. Without the tracks, the distance of a
ride is the sum of the straight-line distances between its stops (the MATSim route distance in Java).
"""
import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .lemanpass import HOME_ZONE as LEMANPASS_HOME_ZONE, NETWORK as LEMANPASS_NETWORK, single_profiles

PT_REGIONAL_RADIUS_KM = 10.0   # a regional subscription is valid when the origin and the destination are this close to home
FIRST_CLASS_FACTOR = 1.7    # 1st class: the 2nd class price times this, rounded as usual
MIN_ZONE_M = 200            # with the tracks: a zone crossed on less than this does not count (the stops at the border of two zones)
RESIDENT_RADIUS_M = 1000    # without the zone polygons: a place is in the area of the networks of the stops this close to it
# the stops in Switzerland, by their id: the Swiss stops (8501120, 8501120:0:1, Parent8501120, ch:1:sloid:...) and the
# Swiss stations in the French feed (StopPoint:OCETrain:::TER-85162727, Genève-Champel). Not the French coaches stopping
# in Geneva (FR:...), whose rides go abroad anyway. The Swiss stops abroad (Liechtenstein, Jestetten) have the Swiss fares.
SWISS_STOP = re.compile(r"^(?:Parent)?85\d{5}(?::|$)|^ch:1:sloid:|^StopPoint:OCE\w*:::\w*-85\d{6}$")


def _java_round(value):
    """Rounded to the cent, halves up, as Java's Math.round(value * 100) / 100."""
    return math.floor(value * 100.0 + 0.5) / 100.0


def _round_up(price, step):
    """Up to the next multiple of step (0.2, 1), without going a step too far on a float like 34.00000000000001."""
    return math.ceil(round(price / step, 9)) * step


def _fare_rounding(price, km):
    """The rounding of the fares: up to 20 cents below 69 km, else up to the next franc."""
    return _round_up(price, 0.2) if km < 69 else _round_up(price, 1.0)


SLOID = re.compile(r"^(?:Parent)?ch:1:sloid:(\d+)")   # the Swiss stop ids since 2026 (ch:1:sloid:3000:2:3)


def java_station_id(stop_id):
    """The station of a stop id, as getZones in Java: the part before the first ':' or '.' (8501120:0:1 -> 8501120)."""
    return re.split(r"[:.]", stop_id)[0] if ":" in stop_id else stop_id


def station_id(stop_id):
    """The station of a stop id as a DiDok number, the key of gtfs_zones.csv and SBB_all_distances.csv: as
    java_station_id, except for the SLOIDs, whose number is the DiDok number without the country code 85 when it has at
    most 5 digits (ch:1:sloid:3000:2:3 -> 8503000) and the DiDok number itself otherwise (ch:1:sloid:1100008 -> 1100008).
    The didok column of the feed is the reference: this rule gives it for all but a few of the SLOIDs of 2026."""
    match = SLOID.match(stop_id)
    if match:
        number = match.group(1)
        return number if len(number) > 5 else "85" + number.zfill(5)
    return java_station_id(stop_id)


def _simplified(stop_id):
    """The station of a stop id (station_id): for a SLOID, its DiDok number rather than "ch" as in Java."""
    return station_id(stop_id)


@dataclass(frozen = True)
class Authority:
    """A fare network ("authority"), by its id, with the priority that decides between the networks of a ride."""
    id: str
    priority: int     # SBB and None: 0, ZPass*: 1, the other fare networks: 2

    def __eq__(self, other):
        return isinstance(other, Authority) and other.id == self.id

    def __hash__(self):
        return hash(self.id)


NONE = Authority("None", 0)   # the rides in no network, priced by distance
SBB = Authority("SBB", 0)
LABELS = {"None": "Distance-based"}   # the names of the authorities in the price breakdown


@dataclass
class Traveller:
    """Who travels: what changes the price."""
    age: int = 30
    halbtax: bool = False       # the half-fare card
    ga: bool = False            # the GA travelcard: everything is free
    verbund_abo: bool = False   # a regional subscription (free within PT_REGIONAL_RADIUS_KM of home)
    strecken_abo: bool = False  # a route subscription (the same)
    gleis7_abo: bool = False    # Gleis 7: free from 19:00 to 05:00 under 25
    junior_abo: bool = False    # the Junior card: free under 16
    first_class: bool = False
    student: bool = False
    home: tuple = None          # (lat, lon), for the regional subscriptions and the discounts for residents
    study: tuple = None         # (lat, lon) of the place of study, for the discounts for residents
    subscriptions: dict = field(default_factory = dict)   # zone subscriptions: {network: set of zones, or None for all of them}


def parse_subscriptions(text):
    """The zone subscriptions written as text: "ZVV:110,121;Mobilis:all" -> {"ZVV": {"110", "121"}, "Mobilis": None}."""
    result = {}
    for item in filter(None, (part.strip() for part in (text or "").split(";"))):
        network, _, zones = item.partition(":")
        zones = {z.strip() for z in zones.split(",") if z.strip()}
        result[network.strip()] = None if not zones or zones == {"all"} else zones
    return result


@dataclass
class Discount:
    """A share off the ticket of some networks, for some travellers. When several apply to a ticket, the largest one is
    taken: they do not add up."""
    name: str
    percent: float
    age_min: int = None         # None: no lower bound
    age_max: int = None         # inclusive; None: no upper bound
    student: bool = None        # True: students only, False: non-students only, None: everyone
    networks: tuple = ()        # the authority ids it applies to ("SBB", "None": distance-based); empty: all of them
    residents: bool = False     # only for the travellers living or studying in the area of the network (never SBB, "None")

    def applies(self, authority, traveller, lives_in = lambda authority: False):
        """Whether the discount applies to the ticket of an authority for a traveller. lives_in(authority): whether the
        traveller lives or studies in the area of the authority, asked only for a discount for residents."""
        return ((not self.networks or authority in self.networks)
                and (self.age_min is None or traveller.age >= self.age_min)
                and (self.age_max is None or traveller.age <= self.age_max)
                and (self.student is None or self.student == traveller.student)
                and (not self.residents or lives_in(authority)))


@dataclass
class RideZones:
    """A ride as the pricing sees it (SwissPtLegVariables in Java): its zones in each network and its distances."""
    zones: dict                 # {authority id: [zone ids]}; empty: in no network
    network_km: float           # the distance travelled (along the track, else straight lines between the stops)
    sbb_km: float               # the SBB tariff distance between its first and last stops (-1: not on the SBB)
    from_name: str = ""
    to_name: str = ""
    in_switzerland: bool = True # a stop in Switzerland (SWISS_STOP)
    abroad: bool = False        # a stop outside Switzerland


@dataclass
class PriceResult:
    """The price of a journey, and how it was made up."""
    price: float
    old_price: float            # the price of the older, distance-based model
    network_km: float           # the distance of the journey
    parts: list = field(default_factory = list)   # (authority, price) of each ticket (a subscription used: price 0)
    discounts: dict = field(default_factory = dict)   # {authority: the Discount applied to its ticket}

    @property
    def rule(self):
        """The breakdown as text, e.g. "Unireso 2.25 (Student discount -25 %) + SBB 24.40", or "free"."""
        def part(a, p):
            d = self.discounts.get(a)
            return f"{LABELS.get(a, a)} {p:.2f}" + (f" ({d.name} -{d.percent:g} %)" if d else "")
        return " + ".join(part(a, p) for a, p in self.parts) or "free"


class Pricing:
    """The fare model: Pricing("data/pricing").price(journey, network, traveller)."""

    def __init__(self, folder, cap = False, tracks = None):
        """folder: the pricing inputs (pricingDescription.xml, gtfs_zones.csv, SBB_all_distances.csv).

        cap: prices capped at 60 CHF (35 with half fare), as in eqasim-java. tracks (gtfs_router.tracks): if given, the
        zones of a ride are the zones its track crosses and its distance is the length of the track; else the zones of
        its stops and the straight-line distances between them."""
        self.folder, self.cap, self.tracks = folder, cap, tracks
        self.lemanpass = None   # the LemanPass (utils.gtfs_router.lemanpass), for the journeys into its area abroad; None: not priced
        self._areas = {}   # (lat, lon) rounded -> the networks whose area holds it
        self._read_description(os.path.join(folder, "pricingDescription.xml"))
        self._read_zones(os.path.join(folder, "gtfs_zones.csv"))
        distances = pd.read_csv(os.path.join(folder, "SBB_all_distances.csv"), dtype = {"origin_id": str, "destination_id": str})
        self.sbb_distances = {}   # (station, station) -> SBB tariff km, both ways
        for a, b, d in zip(distances["origin_id"].str.strip(), distances["destination_id"].str.strip(), distances["distance"].astype(float)):
            self.sbb_distances[(a, b)] = d
            self.sbb_distances[(b, a)] = d
        # the SBB is a "network" with a single zone that holds every station of the distance table
        for stop in set(distances["origin_id"].str.strip()) | set(distances["destination_id"].str.strip()):
            self.zones_by_stop.setdefault(stop, set()).add((SBB, "SBB"))

    # ------------------------------------------------------------------ inputs

    def _read_description(self, path):
        """The fares of pricingDescription.xml: the default distance-based price, the SBB distance bands and minimum
        prices, the price of each network (a list by number of zones, or a centre zone and the others), and the age
        discounts of the networks (<targetedPricing>), which become Discounts."""
        root = ET.parse(path).getroot()
        d = root.find("distanceBasedDefault")
        self.default = (float(d.find("flatRatePrice").get("value")), float(d.find("pricePerKM").get("value")), float(d.find("power2Factor").get("value")))
        s = root.find("sbb")
        self.sbb_min = float(s.find("minimumPrice").get("value"))
        self.sbb_min_halbtax = float(s.find("minimumPriceHalbtax").get("value"))
        self.sbb_bins = [(float(b.get("distance_min")), float(b.get("distance_max")), float(b.get("priceByKM"))) for b in s.findall("priceByKMByDistanceBin")]

        # the calculator of each network: full and half-fare prices by number of zones (and the zones counting twice),
        # or, for a network around a centre zone (Unireso), the price of the centre zone and of each other zone
        self.calculators, self.discounts = {}, []
        for a in root.findall("authority"):
            name = a.find("name").get("value")
            c = {"full": None, "halbtax": None, "double": set(), "center": None, "center_price": float("nan"),
                 "center_price_halbtax": float("nan"), "additional_price": float("nan"), "additional_price_halbtax": float("nan")}
            center_based = a.find("zonalBased").get("value") == "centerbased"
            if not center_based:
                for p in a.findall("priceListByZone"):
                    prices = [float(v) for v in p.get("prices").split(", ")]
                    c["halbtax" if p.get("halbtax") == "true" else "full"] = prices
                two = a.find("zonesCountingAsTwo")
                if two is not None:
                    c["double"] = {z.strip() for z in two.get("zones").split(", ")}
            else:
                c["center"] = a.find("centerZoneID").get("zone")
                # the full price and the half-fare one (halbtax="true"); without a half-fare price, the full one
                for key, tag in (("center_price", "priceCenterZone"), ("additional_price", "priceAdditionalZone")):
                    prices = {p.get("halbtax", "false"): float(p.get("price")) for p in a.findall(tag)}
                    c[key] = prices.get("false", prices.get("true"))
                    c[key + "_halbtax"] = prices.get("true", c[key])
            for t in a.findall("targetedPricing"):
                low, high = (int(x) for x in t.get("age_range").split("_"))
                self.discounts.append(Discount(f"{name}, age {low}-{high}", float(t.get("discount")), low, high, None, (name,)))
            self.calculators[name] = c

    def _read_zones(self, path):
        """The fare networks and the zones of each stop, from gtfs_zones.csv (ZonalReader in Java): the networks of the
        'tarif network' column (Z-Pass: priority 1, the others 2), and the zones of the 'zones' column ("ZVV:110, ...")."""
        df = pd.read_csv(path, dtype = str, keep_default_na = False)
        self.authorities = {}
        for text in df["tarif network"]:
            for auth in re.sub(r"[\[\]' ]", "", text).split(","):
                if auth and auth != "nan":
                    self.authorities.setdefault(auth, Authority(auth, 1 if "ZPass" in auth else 2))
        self.zones_by_stop = {}   # stop id -> {(Authority, zone)}
        for stop_id, text in zip(df["stop_id"].str.strip(), df["zones"]):
            for entry in text.split(","):
                parts = entry.strip().split(":")
                if len(parts) == 2 and parts[0].strip() in self.authorities:
                    self.zones_by_stop.setdefault(stop_id, set()).add((self.authorities[parts[0].strip()], parts[1].strip()))
        self.network_zones = {}   # network -> all its zones
        for pairs in self.zones_by_stop.values():
            for auth, zone in pairs:
                self.network_zones.setdefault(auth.id, set()).add(zone)

    def add_stop_zones(self, rows):
        """Adds zones to stops, from (stop id, network, zone) rows (the stops abroad: zones.foreign_stop_zones). A stop
        that already has a zone of the network (in gtfs_zones.csv, by its id or its station) keeps it. Returns the
        number of stops given a zone."""
        added = 0
        for stop_id, network, zone in rows:
            if any(a.id == network for a, _ in self._stop_zones(stop_id)):
                continue
            auth = self.authorities.setdefault(network, Authority(network, 1 if "ZPass" in network else 2))
            self.zones_by_stop.setdefault(stop_id, set()).add((auth, zone))
            self.network_zones.setdefault(network, set()).add(zone)
            added += 1
        self._areas.clear()
        return added

    # ------------------------------------------------------------------ rides

    def _stop_zones(self, stop_id):
        """The (Authority, zone) of a stop: those of its id and of its station (getZones in Java)."""
        return self.zones_by_stop.get(stop_id, set()) | self.zones_by_stop.get(_simplified(stop_id), set())

    def ride_zones(self, stop_ids):
        """The zones of a ride from its stops (cleanZoneInfo in Java): {authority id: [zones]} for every network shared
        by all the stops of the ride, whatever its priority: group_rides chooses between them, once it knows the other
        rides. A Z-Pass reuses the zone numbers of the networks it joins (Zürich HB: ZVV 110 and ZPassAwelle 110), so the
        zones are kept per network."""
        by_stop = {}
        for stop in dict.fromkeys(stop_ids):
            groups = {}
            for auth, zone in sorted(self._stop_zones(stop), key = lambda p: (p[0].id, len(p[1]), p[1])):
                groups.setdefault(auth, []).append(zone)
            by_stop[stop] = groups

        common = None
        for groups in by_stop.values():
            common = set(groups) if common is None else common & set(groups)
        if not common:
            return {}

        result = {}
        for groups in by_stop.values():
            for auth in sorted(common, key = lambda a: a.id):
                for zone in groups.get(auth, []):
                    if zone not in result.setdefault(auth.id, []):
                        result[auth.id].append(zone)
        return result

    def ride(self, network, ride):
        """The RideZones of a ride of the router: its networks and zones (from its track if there are tracks, else from
        its stops), its distance, and its SBB tariff distance."""
        stop_ids = [str(network.stop_ids[s]) for s in ride.stops]
        zones = self.ride_zones(stop_ids)
        km = _ride_km(network, ride.stops)
        if self.tracks is not None and ride.pos >= 0:
            zones = self.track_zones(zones, stop_ids, self.tracks.crossed(network, ride), lambda a: self.tracks.pieces(network, ride, a))
            km = self.tracks.length_m(network, ride) / 1000
        sbb_km = self.sbb_distances.get((_simplified(stop_ids[0]), _simplified(stop_ids[-1])), -1.0) if "SBB" in zones else -1.0
        swiss = [bool(SWISS_STOP.match(s)) for s in stop_ids]
        return RideZones(zones, km, sbb_km, network.stop_names[ride.stops[0]], network.stop_names[ride.stops[-1]], any(swiss), not all(swiss))

    def track_zones(self, zones, stop_ids, crossed, pieces_of):
        """The zones of a ride in each of its networks, from its track rather than only its stops.

        zones: the zones from the stops (ride_zones); crossed: {network: {zone: metres}} along the track; pieces_of(network):
        the zones of the network along the track, in order, as (zone, from, to) in metres.

        A ride is in every zone its track crosses for at least MIN_ZONE_M, and in the zone of each of its stops. A stop on
        the border of two zones (listed in both) only adds the one the track goes through most. At the first and last
        stops, when they are on a border, the stretch of track next to them inside their zones belongs to them: of those
        zones, the ride keeps the ones it needs anyway (the zone of another stop), else the one it goes through most. For
        example, Stettbach (ZVV 110 and 121) to Zürich HB goes under Dübendorf (121) in the Zürichberg tunnel, but is only
        in zone 110."""
        result = {}
        for auth, stop_zones in zones.items():
            along = crossed.get(auth)
            if not along:
                result[auth] = stop_zones
                continue
            keep, needed = {z for z, m in along.items() if m >= MIN_ZONE_M}, set()
            for stop_id in dict.fromkeys(stop_ids):
                own = {z for a, z in self._stop_zones(stop_id) if a.id == auth}
                if len(own) == 1:
                    keep |= own
                    needed |= own
                elif own and not own & keep:
                    keep.add(max(sorted(own), key = lambda z: along.get(z, 0.0)))
            pieces = pieces_of(auth)
            for stop_id, ordered in ((stop_ids[0], pieces), (stop_ids[-1], pieces[::-1])):
                own = {z for a, z in self._stop_zones(stop_id) if a.id == auth}
                if len(own) < 2:
                    continue
                near = {}   # the zones of the stop along the track next to it, and their lengths
                for k, (zone, start, end) in enumerate(ordered):
                    if zone not in own:
                        break
                    near[zone] = near.get(zone, 0.0) + end - start
                further = {zone for zone, _, _ in ordered[k + 1 if zone in own else k:]} if near else set()
                absorbed = {z for z in near if z not in further}   # the zones crossed only next to the stop
                if not absorbed:
                    continue
                needed_here = absorbed & needed
                keep -= absorbed
                keep |= needed_here or {max(sorted(absorbed), key = lambda z: near[z])}
            result[auth] = sorted(keep, key = lambda z: (len(z), z))
        return result

    def networks_at(self, network, point):
        """The fare networks whose area holds a place (lat, lon), for the discounts for residents: the networks of the
        zone polygons it is in (with the tracks), else those of the stops within RESIDENT_RADIUS_M. Never "SBB"."""
        key = (round(point[0], 4), round(point[1], 4))
        if key not in self._areas:
            if self.tracks is not None:
                import shapely
                from .tracks import TO_LV95
                inside = self.tracks.zones.contains(shapely.Point(*TO_LV95.transform(point[1], point[0])))
                found = set(self.tracks.zones.loc[inside, "network"])
            else:
                near, _ = network.stops_near(point[0], point[1], RESIDENT_RADIUS_M)
                found = {a.id for s in near for a, _ in self._stop_zones(str(network.stop_ids[s]))}
            self._areas[key] = found - {"SBB"}
        return self._areas[key]

    # ------------------------------------------------------------------ subscriptions

    def covering(self, stop_id, subscriptions):
        """The zone subscription (its network) valid at a stop, or None."""
        for auth, zone in self._stop_zones(stop_id):
            if auth.id in subscriptions and (subscriptions[auth.id] is None or zone in subscriptions[auth.id]):
                return auth.id
        return None

    def zone_covered(self, network, zone, subscriptions):
        """Whether a zone of a network is in the subscriptions. A Z-Pass (a pass across two networks) and the networks it
        joins share their zone numbers, so a zone can be covered by a subscription of another of them."""
        for other, zones in subscriptions.items():
            if other == network or "ZPass" in other or "ZPass" in network:
                zones = self.network_zones.get(other, set()) if zones is None else zones
                if zone in zones and (other == network or zone in self.network_zones.get(network, set())):
                    return True
        return False

    def uncovered_ride(self, network, ride, subscriptions):
        """The part of a ride the traveller still has to pay with their zone subscriptions, as RideZones, or None if the
        subscriptions cover it.

        If they cover all its stops, nothing is paid. If the ride leaves the subscription, it is paid from the last covered
        stop before it leaves to the first covered stop after, without the covered zones: an extension ticket. A ride that
        leaves the zones and comes back into them is paid in full, but for the covered zones."""
        stop_ids = [str(network.stop_ids[s]) for s in ride.stops]
        covered = [self.covering(s, subscriptions) is not None for s in stop_ids]
        if all(covered):
            return None
        first = covered.index(False)
        last = len(covered) - 1 - covered[::-1].index(False)
        if not any(covered[first:last + 1]):
            begin, end = max(first - 1, 0), min(last + 1, len(covered) - 1)
        else:
            begin, end = 0, len(covered) - 1
        part = self.ride(network, replace(ride, stops = ride.stops[begin:end + 1], pos = ride.pos + begin if ride.pos >= 0 else -1))
        zones = {a: [z for z in zs if not self.zone_covered(a, z, subscriptions)] for a, zs in part.zones.items()}
        if part.zones and not any(zones.values()):
            return None
        part.zones = {a: zs for a, zs in zones.items() if zs}
        return part

    def _authorities_of(self, ride):
        """The Authorities of a ride (NONE for a ride in no network)."""
        if not ride.zones:
            return {NONE}
        return {self.authorities.get(a, SBB if a == "SBB" else Authority(a, 2)) for a in ride.zones}

    def group_rides(self, rides):
        """Which network prices which rides (getPricingStrategy in Java): {authority id: [rides]}, one ticket each.

        If one network covers all the rides, it prices the whole journey (the one of the highest priority if there are
        several: an S-Bahn Zürich HB - Altstetten and a train Altstetten - Baden are in ZVV, ZPassAwelle and SBB for the
        first, ZPassAwelle and SBB for the second, so ZPassAwelle). Otherwise only the networks of the highest priority of
        each ride count: each ride goes to one it shares with another ride (the one of the lowest priority, as in Java),
        or else to its own (the one of the highest priority)."""
        sets = [self._authorities_of(r) for r in rides]
        everywhere = set.intersection(*sets)
        if everywhere:
            return {max(everywhere, key = lambda a: (a.priority, a.id)).id: list(rides)}
        sets = [{a for a in s if a.priority == max(b.priority for b in s)} for s in sets]
        frequency = {}
        for s in sets:
            for a in s:
                frequency[a] = frequency.get(a, 0) + 1
        groups = {}
        for ride, s in zip(rides, sets):
            shared = [a for a in s if frequency.get(a, 0) > 1]
            chosen = min(shared, key = lambda a: a.priority) if shared else max(s, key = lambda a: a.priority)
            groups.setdefault(chosen.id, []).append(ride)
        return groups

    # ------------------------------------------------------------------ calculators: the price of one ticket

    def default_price(self, rides, halbtax, factor = 1.0):
        """The price of rides in no network, by distance: a fixed part + a price per km + a price per km² (negative:
        cheaper per km on long rides), times factor (1st class), halved with the half-fare card, rounded up."""
        km = sum(r.network_km for r in rides)
        flat, per_km, power2 = self.default
        price = (flat + per_km * km + power2 * km * km) * factor
        if halbtax:
            price /= 2
        return _java_round(_fare_rounding(price, km))

    @staticmethod
    def _sbb_rounding(d):
        """The SBB tariff distance rounded up by steps: to 4 km up to 8 km, to 2 km up to 30, to 3 km up to 60, to 4 km
        up to 100, to 5 km up to 150, to 10 km up to 300, then to 20 km."""
        for limit, step in ((8, 4), (30, 2), (60, 3), (100, 4), (150, 5), (300, 10)):
            if d <= limit:
                return math.ceil(d / step) * step
        return math.ceil(d / 20.0) * 20

    def sbb_price(self, rides, halbtax, factor = 1.0):
        """The SBB price of rides: their tariff distances added up and rounded by steps, each kilometre at the price of
        its distance band, times factor (1st class), rounded up, halved with the half-fare card, never below the minimum."""
        distance = self._sbb_rounding(sum(r.sbb_km for r in rides))
        price = 0.0
        for low, high, rate in self.sbb_bins:
            if distance >= int(low):
                price += (min(distance, int(high)) - int(low) + 1) * rate
            else:
                break
        price = _fare_rounding(price * factor, distance)
        minimum = _round_up((self.sbb_min_halbtax if halbtax else self.sbb_min) * factor, 0.2)
        return _java_round(max(price / 2.0 if halbtax else price, minimum))

    def zone_price(self, authority, rides, halbtax, factor = 1.0):
        """The price of rides in a zone network: the price list by number of zones visited (the zones counting as two
        count twice; beyond the end of the list, its last price), or for a network around a centre zone, the price of the
        centre zone plus a price for each other zone (the default price if the centre zone is not visited).

        factor: 1st class, the 2nd class price times it rounded up to 20 cents (the SBB rule, used for the zone
        networks too until their own rules are checked)."""
        c = self.calculators[authority]
        zones = {z for r in rides for z in r.zones.get(authority, [])}
        if not zones:
            raise ValueError("At least one zone must be visited")
        if c["full"] is not None and c["halbtax"] is not None:
            prices = c["halbtax"] if halbtax else c["full"]
            n = len(zones) + sum(1 for z in zones if z in c["double"])
            price = _java_round(prices[min(n, len(prices)) - 1])
            if factor != 1.0:
                price = _round_up(price * factor, 0.2)
            if n > len(prices):
                return _java_round(price)
        elif c["center"] in zones:
            suffix = "_halbtax" if halbtax else ""
            price = c["center_price" + suffix] + c["additional_price" + suffix] * (len(zones) - 1)
            if factor != 1.0:
                price = _round_up(price * factor, 0.2)
        else:
            price = self.default_price(rides, halbtax, factor)
        return _java_round(price)

    def _calculate(self, authority, rides, halbtax, factor = 1.0):
        """The price of one ticket: the rides priced by an authority."""
        if authority == "SBB":
            return self.sbb_price(rides, halbtax, factor)
        if authority in self.calculators:
            return self.zone_price(authority, rides, halbtax, factor)
        return self.default_price(rides, halbtax, factor)   # "None" and the authorities without a price description

    # ------------------------------------------------------------------ journey

    def price(self, journey, network, traveller = None, origin = None, destination = None):
        """The PriceResult of a Journey of the router for a Traveller (processRequest in Java).

        network: the Network the journey was found on. origin and destination (lat, lon): only needed for the regional
        subscriptions (free within PT_REGIONAL_RADIUS_KM of home). A stay at a stop on the way ends a ticket: each part of
        the journey around it is priced on its own and the prices are added up."""
        t = traveller or Traveller()
        if any(leg.kind == "stay" for leg in journey.legs):
            parts, current = [], []
            for leg in journey.legs + [None]:
                if leg is None or leg.kind == "stay":
                    if any(l.kind == "ride" for l in current):
                        parts.append(self.price(type(journey)(current[0].dep, current[-1].arr, current), network, t, origin, destination))
                    current = []
                else:
                    current.append(leg)
            return PriceResult(_java_round(sum(p.price for p in parts)), _java_round(sum(p.old_price for p in parts)),
                               sum(p.network_km for p in parts), [x for p in parts for x in p.parts])
        if not journey.rides:
            return PriceResult(0.0, 0.0, 0.0)
        wholes = [self.ride(network, r) for r in journey.rides]
        km = sum(w.network_km for w in wholes)

        # free travel
        dep = journey.dep
        gleis7 = t.age < 25 and t.gleis7_abo and (dep < 5 * 3600 or dep >= 19 * 3600)
        if t.ga or t.age < 6 or (t.age < 16 and t.junior_abo) or gleis7:
            return PriceResult(0.0, 0.0, km)
        if (t.verbund_abo or t.strecken_abo) and t.home is not None and origin is not None and destination is not None:
            home_km = max(_km(origin, t.home), _km(destination, t.home))
            if home_km <= PT_REGIONAL_RADIUS_KM:
                return PriceResult(0.0, 0.0, km)

        # the zone subscriptions: what is left to pay of each ride (None: nothing), and the subscriptions used on it
        paid, used = [], []
        for r, whole in zip(journey.rides, wholes):
            part = self.uncovered_ride(network, r, t.subscriptions) if t.subscriptions else whole
            paid.append(part)
            helped = part is None or part.zones != whole.zones or part.network_km != whole.network_km
            used.append({n for s in r.stops if (n := self.covering(str(network.stop_ids[s]), t.subscriptions))} if helped else set())
        rides = [p for p in paid if p is not None]
        parts = [(f"{n} subscription", 0.0) for n in sorted(set().union(*used))]
        if not rides:
            return PriceResult(0.0, 0.0, km, parts)
        paid_km = sum(r.network_km for r in rides)

        # one ticket per network, with half fare (half-fare card or under 16), 1st class and the largest discount
        half = t.halbtax or t.age < 16
        factor = FIRST_CLASS_FACTOR if t.first_class else 1.0
        tickets, discounts = self._tickets(rides, network, t, half, factor)
        parts += tickets
        price = _java_round(sum(p for _, p in parts))

        # or a Léman Pass single ticket for some rides (with no subscription: it is bought whole), and the others as above.
        # Unireso only sells Zone 10 then: a way with a Unireso ticket abroad is kept only if no Léman Pass ticket fits
        if self.lemanpass is not None:
            profiles = single_profiles(t.age, t.halbtax)
            best = math.inf if self._lemanpass_zones(rides) else price
            for start, end, trip, lp_price in self.lemanpass.single_tickets(journey.rides, wholes, network, self, profiles, 2 - t.first_class):
                outside = [k for k in range(len(paid)) if not start <= k < end]
                rest = [paid[k] for k in outside if paid[k] is not None]
                if self._lemanpass_zones(rest):
                    continue
                rest_tickets, rest_discounts = self._tickets(rest, network, t, half, factor) if rest else ([], {})
                total = _java_round(lp_price + sum(p for _, p in rest_tickets))
                if total < best:
                    subscriptions = sorted(set().union(*(used[k] for k in outside)))
                    parts = [(f"{n} subscription", 0.0) for n in subscriptions] + [(f"Léman Pass ({trip})", lp_price)] + rest_tickets
                    price, discounts, best = total, rest_discounts, total

        # the older model: a price by distance only
        old = max(2.8, 2 * (0.21 * paid_km - 0.00015 * paid_km ** 2)) * factor
        if half:
            old /= 2.0
        old = _java_round(old)
        if self.cap:
            cap = 35.0 if half else 60.0
            price, old = min(price, cap), min(old, cap)
        return PriceResult(price, old, km, parts, discounts)

    def _lemanpass_zones(self, rides):
        """Whether the usual tickets of rides include a Unireso ticket with a zone abroad (other than Zone 10), which the
        Léman Pass sells rather than Unireso (isAbroadTicket in Java)."""
        group = self.group_rides(rides).get(LEMANPASS_NETWORK, []) if rides else []
        return any(str(z) != LEMANPASS_HOME_ZONE for r in group for z in r.zones.get(LEMANPASS_NETWORK, ()))

    @staticmethod
    def _swiss_ticket(authority, rides):
        """Whether the ticket of an authority for rides has the Swiss fares, where the half-fare card holds: SBB always,
        a zone network with a stop in Switzerland, a distance-based ticket with no stop abroad."""
        if authority == SBB.id:
            return True
        if authority == NONE.id:
            return not any(r.abroad for r in rides)
        return any(r.in_switzerland for r in rides)

    def _tickets(self, rides, network, t, half, factor):
        """The tickets of rides (RideZones), one per network: ([(authority, price)], {authority: the Discount applied}),
        with half fare (under 16, or with the half-fare card on a ticket with the Swiss fares, see _swiss_ticket), the 1st
        class factor and the largest discount of each."""
        parts, discounts, areas = [], {}, []   # areas: the networks where the traveller lives or studies, found when first needed

        def lives_in(authority):
            if not areas:
                areas.append(set().union(*(self.networks_at(network, p) for p in (t.home, t.study) if p)))
            return authority in areas[0]

        for authority, group in self.group_rides(rides).items():
            half_here = half and (t.age < 16 or self._swiss_ticket(authority, group))
            price = self._calculate(authority, group, half_here, factor)
            discount = max((d for d in self.discounts if d.applies(authority, t, lives_in)), key = lambda d: d.percent, default = None)
            if discount is not None and discount.percent > 0:
                price, discounts[authority] = _java_round(price * (1.0 - discount.percent / 100.0)), discount
            parts.append((authority, price))
        return parts, discounts

    # ------------------------------------------------------------------ fares, as the price editor shows and changes them

    def fares(self):
        """The fares as JSON, as the price editor shows them: "sbb" (minimum prices and distance bands), "default" (the
        distance-based price), "networks" (the prices of each network) and "discounts". with_changes takes a part of it."""
        networks = {}
        for name, c in sorted(self.calculators.items()):
            if c["center"] is not None:
                networks[name] = {"kind": "center", "center_zone": c["center"], "center": c["center_price"], "center_halbtax": c["center_price_halbtax"],
                                  "additional": c["additional_price"], "additional_halbtax": c["additional_price_halbtax"]}
            else:
                networks[name] = {"kind": "zones", "full": c["full"], "halbtax": c["halbtax"], "double": sorted(c["double"])}
        flat, per_km, power2 = self.default
        return {"sbb": {"minimum": self.sbb_min, "minimum_halbtax": self.sbb_min_halbtax,
                        "bands": [{"from_km": low, "to_km": high, "chf_per_km": rate} for low, high, rate in self.sbb_bins]},
                "default": {"flat": flat, "per_km": per_km, "per_km2": power2},
                "networks": networks,
                "discounts": [{"name": d.name, "percent": d.percent, "age_min": d.age_min, "age_max": d.age_max, "student": d.student,
                               "networks": list(d.networks), "residents": d.residents} for d in self.discounts]}

    def with_changes(self, changes):
        """A new Pricing with some fares changed; this one stays as it is.

        changes: a part of fares(), with "sbb", "default", single networks of "networks" (each given whole) and/or
        "discounts" (the whole list, which replaces the discounts). Raises ValueError, with a message saying which value,
        when something is not valid (a negative price, an unknown network...)."""
        if not isinstance(changes, dict) or set(changes) - {"sbb", "default", "networks", "discounts"}:
            raise ValueError('The changes are a JSON object with "sbb", "default", "networks" or "discounts"')
        other = copy.copy(self)   # the stops, zones and distances are shared; only the fares are replaced
        other.calculators = {name: dict(c) for name, c in self.calculators.items()}
        if "sbb" in changes:
            s = _object(changes["sbb"], "sbb", {"minimum", "minimum_halbtax", "bands"})
            bands = s.get("bands", [{"from_km": l, "to_km": h, "chf_per_km": r} for l, h, r in self.sbb_bins])
            if not isinstance(bands, list) or not bands:
                raise ValueError("sbb.bands: a list of distance bands")
            other.sbb_bins = sorted(((_number(b, "from_km", "sbb.bands"), _number(b, "to_km", "sbb.bands"), _number(b, "chf_per_km", "sbb.bands"))
                                     for b in bands), key = lambda b: b[0])
            if any(low > high for low, high, _ in other.sbb_bins):
                raise ValueError("sbb.bands: from_km after to_km")
            other.sbb_min = _number(s, "minimum", "sbb", self.sbb_min)
            other.sbb_min_halbtax = _number(s, "minimum_halbtax", "sbb", self.sbb_min_halbtax)
        if "default" in changes:
            d = _object(changes["default"], "default", {"flat", "per_km", "per_km2"})
            flat, per_km, power2 = self.default
            other.default = (_number(d, "flat", "default", flat), _number(d, "per_km", "default", per_km),
                             _number(d, "per_km2", "default", power2, negative = True))
        for name, n in _object(changes.get("networks", {}), "networks").items():
            if name not in self.calculators:
                raise ValueError(f"No fare network {name} with prices")
            c = other.calculators[name]
            if c["center"] is not None:
                n = _object(n, name, {"kind", "center_zone", "center", "center_halbtax", "additional", "additional_halbtax"})
                for key, price in (("center", "center_price"), ("center_halbtax", "center_price_halbtax"),
                                   ("additional", "additional_price"), ("additional_halbtax", "additional_price_halbtax")):
                    c[price] = _number(n, key, name, c[price])
            else:
                n = _object(n, name, {"kind", "full", "halbtax", "double"})
                for key in ("full", "halbtax"):
                    if key in n:
                        if not isinstance(n[key], list) or not n[key] or len(n[key]) > 100:
                            raise ValueError(f"{name}.{key}: a list of prices, by number of zones")
                        c[key] = [_number({"price": v}, "price", f"{name}.{key}") for v in n[key]]
        if "discounts" in changes:
            if not isinstance(changes["discounts"], list) or len(changes["discounts"]) > 50:
                raise ValueError("discounts: a list (of at most 50)")
            known = set(self.calculators) | {"SBB", "None"}
            other.discounts = []
            for k, d in enumerate(changes["discounts"]):
                d = _object(d, f"discounts[{k}]", {"name", "percent", "age_min", "age_max", "student", "networks", "residents"})
                percent = _number(d, "percent", f"discounts[{k}]")
                if percent > 100:
                    raise ValueError(f"discounts[{k}].percent: at most 100")
                ages = [None if d.get(key) is None else int(_number(d, key, f"discounts[{k}]")) for key in ("age_min", "age_max")]
                if d.get("student") not in (None, True, False):
                    raise ValueError(f"discounts[{k}].student: true, false or null")
                if d.get("residents", False) not in (True, False):
                    raise ValueError(f"discounts[{k}].residents: true or false")
                networks = d.get("networks") or []
                if not isinstance(networks, list) or set(networks) - known:
                    raise ValueError(f"discounts[{k}].networks: unknown {sorted(set(networks) - known) if isinstance(networks, list) else networks}")
                other.discounts.append(Discount(str(d.get("name") or f"Discount {k + 1}")[:60], percent, ages[0], ages[1], d.get("student"), tuple(networks),
                                                d.get("residents", False)))
        return other


def _object(value, where, keys = None):
    """value if it is a JSON object (with only the keys given, if any), else a ValueError naming where it is."""
    if not isinstance(value, dict) or (keys is not None and set(value) - keys):
        raise ValueError(f"{where}: an object" + (f" with {', '.join(sorted(keys))}" if keys else ""))
    return value


def _number(values, key, where, default = None, negative = False):
    """values[key] as a float (default if missing), or a ValueError naming where it is: it must be a finite number, and
    at least 0 unless negative."""
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 and not negative):
        raise ValueError(f"{where}.{key}: a number" + ("" if negative else " of at least 0"))
    return float(value)


def _ride_km(network, stops):
    """The distance of a ride without its track: the sum of the straight-line distances between its stops, in km."""
    lat, lon = network.stop_lat[stops], network.stop_lon[stops]
    return float(np.sum(np.hypot(np.diff(lat) * 111.32, np.diff(lon) * 111.32 * np.cos(np.radians(lat[:-1])))))


def _km(a, b):
    """The straight-line distance between two (lat, lon), in km."""
    return float(np.hypot((a[0] - b[0]) * 111.32, (a[1] - b[1]) * 111.32 * np.cos(np.radians(a[0]))))
