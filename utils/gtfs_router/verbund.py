"""The zones of the Verbundabos of a synthetic population of eqasim-switzerland.

The population gives who has a Verbundabo (pt_subscription 2: Verbundabo, 4: Verbundabo and half-fare card), where
they live and where they work or study, but not which zones their subscription covers. These are guessed from the
commute, the trip the subscription is most likely bought for:

1. Each agent gets a commute destination: the place of education for the students, else the workplace (the workplace
   of a remote worker is their home). The other agents have none.
2. Their home and destination are placed in the fare zones (the zone polygons of zones.py, drawn from the
   Tarifverbundkarte and the zones of the stops). The home network is the regional network the home is in (not a
   Z-Pass). A place in no zone but within SNAP_M of one (a home on a lake shore) is in the nearest one.
3. The agents are grouped by the zones of their home and destination (the zone OD matrix, od_matrix.csv):
   - no destination, or the same zone: the home zone;
   - two zones of the same network: the zones of that network along the commute;
   - two networks joined by a Z-Pass (Baden - Zürich: ZPassAwelle): the Z-Pass zones along the commute;
   in these two cases, the home and destination zones are always added;
   - two networks (Zürich - Bern): the zones of the home network along the commute until the first ride that leaves
     it, where the SBB (or another network) takes over; of that ride, only the zone of its boarding stop;
   - a home in no network and a destination in one (Sion - Lausanne): the other way round, the zones of the
     destination network along the commute back from the destination, until the ride that enters it; of that ride,
     only the zone of its alighting stop.
   Agents outside every network with a destination in none, or with no route, keep what can be found: their home
   (or destination) zone, or nothing.
4. The exact commute of every agent of a group that needs a route is routed on the timetable (gtfs_router), and the
   agent gets the zones of its own journey (with samples > 0, only that many agents per group are routed, and the
   others get the zones of one of their journeys, drawn at random). An agent without a journey gets the zones of the
   journey of another agent of its group, drawn at random; a group where no journey is found gets the home and
   destination zones of the network.

5. A commute that goes into the Léman Pass area abroad (a French station of the Léman Pass, or a Unireso zone other
   than Zone 10: Geneva - Annemasse, Thonon - Geneva) is covered by a Léman Pass subscription rather than a
   Verbundabo: the cheapest annual one (2nd class, for 6-25 year olds or adults) whose two ends hold the stations of
   the train part of the journey and the zones of the rides before and after it (see lemanpass). Its
   Unireso zones stay in verbund_zones. This needs the fare table of tpg (tpg_<date>_Fare-system_Geneva.xlsx) in the
   pricing folder (lemanpass.load).

verbund_zones() returns one row per agent with a Verbundabo: its operator and zones, and verbund_zones in the format
of the subscriptions of the Swiss PT explorer ("ZVV:110,121"); subscription (Verbundabo or Léman Pass), lemanpass (the
trip type of the Léman Pass subscription) and lemanpass_annual_chf; od_matrix() gives the agents by group with the zones
found. The stage synthesis.population.models.subscription_zones calls it on the synthetic population.

Copied from verbund_zones.py of the Swiss PT explorer (gtfs_router, github.com/AuroreSallard/swiss-pt-explorer), without
its command line.
"""
import logging
import multiprocessing
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from .lemanpass import subscription_profile
from .pricing import MIN_ZONE_M
from .tracks import TO_LV95, TO_WGS84

logger = logging.getLogger("synpp")

VERBUND_CODES = (2, 4)   # pt_subscription of synthesis.population.models.subscriptions: 2 Verbundabo, 4 with the half-fare card
OUTSIDE_CELL_M = 10_000     # the homes or destinations in no zone are grouped by squares of this size
# A place in no regional zone but this close to one is in it: the outlines of the municipalities, hence of the zones,
# leave out the lakes and are coarse at the borders, so homes on the shore (Zollikon on the Zürichsee) fall just outside.
# In the 5% population of 2026-11-10, all the homes in no municipality were within 100 m of a zone, and all the homes
# farther away were in a municipality of no network.
SNAP_M = 100
ROUTED = ("intra", "zpass", "inter", "destination")   # the cases whose zones come from the routed commute

_shared = {}   # what the routing workers use: the router, the network, the pricing, the Léman Pass (set before they fork)


# ------------------------------------------------------------------ the population

def commutes(persons, work, education):
    """The agents with a Verbundabo and their commute: person_id, age (if the persons have it), home_x, home_y,
    purpose (education, work or none), destination_x, destination_y (NaN without a destination), in LV95."""
    columns = ["person_id"] + (["age"] if "age" in persons else []) + ["home_x", "home_y", "pt_subscription"]
    agents = persons.loc[persons["pt_subscription"].isin(VERBUND_CODES), columns].copy()
    places = {name: pd.DataFrame({"person_id": df["person_id"].to_numpy(), "x": df.geometry.x.to_numpy(), "y": df.geometry.y.to_numpy()}).set_index("person_id")
              for name, df in (("education", education), ("work", work))}
    agents["purpose"] = "none"
    agents["destination_x"], agents["destination_y"] = np.nan, np.nan
    for name in ("work", "education"):   # education last: it wins for the students who also work
        place = places[name].reindex(agents["person_id"])
        has = place["x"].notna().to_numpy()
        agents.loc[has, "purpose"] = name
        agents.loc[has, "destination_x"] = place["x"].to_numpy()[has]
        agents.loc[has, "destination_y"] = place["y"].to_numpy()[has]
    return agents.reset_index(drop = True)


# ------------------------------------------------------------------ the zones of the places

def locate(x, y, zones):
    """The zones of points (LV95): (network, zone) as two Series aligned with x, for the regional network the point is
    in (not a Z-Pass; NaN in none), and zpass, a Series of {Z-Pass network: zone} for the Z-Passes the point is in."""
    n = len(x)
    network, zone = pd.Series(np.nan, index = range(n), dtype = object), pd.Series(np.nan, index = range(n), dtype = object)
    zpass = pd.Series([{} for _ in range(n)], index = range(n), dtype = object)
    valid = np.isfinite(x) & np.isfinite(y)
    points = gpd.GeoDataFrame(geometry = gpd.points_from_xy(x[valid], y[valid]), index = np.flatnonzero(valid), crs = 2056)
    zones = zones[["network", "zone", "geometry"]]
    hits = gpd.sjoin(points, zones, predicate = "within").reset_index(names = "point")[["point", "network", "zone"]]
    hits = pd.concat([hits, snapped(points, zones, hits)], ignore_index = True)
    # the polygons of a network can overlap a little at their borders: one zone per point and network
    hits = hits.sort_values(["point", "network", "zone"]).drop_duplicates(["point", "network"])
    is_zpass = hits["network"].str.startswith("ZPass")
    regional = hits[~is_zpass].drop_duplicates("point")   # a point in two regional networks: the first by name
    network[regional["point"].to_numpy()] = regional["network"].to_numpy()
    zone[regional["point"].to_numpy()] = regional["zone"].to_numpy()
    for point, net, z in zip(hits.loc[is_zpass, "point"], hits.loc[is_zpass, "network"], hits.loc[is_zpass, "zone"]):
        zpass[point][net] = z
    return network, zone, zpass


def snapped(points, zones, hits, max_distance = SNAP_M):
    """The zones of the points in no regional zone but within max_distance of one, as rows of hits (point, network,
    zone): the nearest regional zone, and the Z-Pass zones about as close (they share its outline) for the Z-Passes the
    point is not already in."""
    regional = zones[~zones["network"].str.startswith("ZPass")]
    loose = points[~points.index.isin(hits.loc[~hits["network"].str.startswith("ZPass"), "point"])]
    if len(loose) == 0:
        return hits.iloc[0:0]
    near = gpd.sjoin_nearest(loose, regional, max_distance = max_distance, distance_col = "distance").reset_index(names = "point")
    near = near.sort_values(["point", "distance", "network", "zone"]).drop_duplicates("point")
    if len(near) == 0:
        return hits.iloc[0:0]
    reach = gpd.GeoDataFrame({"point": near["point"].to_numpy()}, crs = points.crs,
                             geometry = points.geometry[near["point"]].buffer(near["distance"].to_numpy() + 1.0).to_numpy())
    zpass = gpd.sjoin(reach, zones[zones["network"].str.startswith("ZPass")], predicate = "intersects")[["point", "network", "zone"]]
    zpass = zpass[~zpass.set_index(["point", "network"]).index.isin(hits.set_index(["point", "network"]).index)]
    logger.info(f"  {len(near)} places within {max_distance:.0f} m of a zone but in none (lake shores, borders): given the nearest zone")
    return pd.concat([near[["point", "network", "zone"]], zpass], ignore_index = True)


def classify(agents, zones):
    """Adds to the agents the zones of their home and destination and how their subscription is found: case (outside:
    the home and the destination in no network; destination: the home in no network, the destination in one; home: no
    destination; same_zone; intra: two zones of the home network; zpass: a Z-Pass joins both; inter: two networks),
    operator (the network of the subscription: of the home, of the destination for destination), home_zone and
    destination_zone (in the operator's network, the destination's own network for inter; no home_zone for
    destination), and od, the key of the group."""
    agents = agents.copy()
    home_net, home_zone, home_zpass = locate(agents["home_x"].to_numpy(), agents["home_y"].to_numpy(), zones)
    dest_net, dest_zone, dest_zpass = locate(agents["destination_x"].to_numpy(), agents["destination_y"].to_numpy(), zones)
    agents["home_network"], agents["destination_network"] = home_net.to_numpy(), dest_net.to_numpy()

    cases, operators, origins, destinations = [], [], [], []
    for k, purpose in enumerate(agents["purpose"]):
        hn, hz, dn, dz = home_net[k], home_zone[k], dest_net[k], dest_zone[k]
        shared = sorted(set(home_zpass[k]) & set(dest_zpass[k]))
        if not isinstance(hn, str) and purpose != "none" and isinstance(dn, str):
            case, operator, o, d = "destination", dn, None, dz
        elif not isinstance(hn, str):
            case, operator, o, d = "outside", None, None, None
        elif purpose == "none":
            case, operator, o, d = "home", hn, hz, None
        elif dn == hn and dz == hz:
            case, operator, o, d = "same_zone", hn, hz, dz
        elif dn == hn:
            case, operator, o, d = "intra", hn, hz, dz
        elif shared:
            zp = shared[0]
            case, operator, o, d = "zpass", zp, home_zpass[k][zp], dest_zpass[k][zp]
        else:
            case, operator, o, d = "inter", hn, hz, dz
        cases.append(case)
        operators.append(operator)
        origins.append(o)
        destinations.append(d)
    agents["case"], agents["operator"], agents["home_zone"], agents["destination_zone"] = cases, operators, origins, destinations

    # the group of each agent: its case, operator and zones; the places in no zone by squares (the destinations of
    # inter abroad, the homes of destination)
    cell = lambda prefix: ("cell:" + np.floor(agents[f"{prefix}_x"] / OUTSIDE_CELL_M).astype("Int64").astype(str) + "_"
                           + np.floor(agents[f"{prefix}_y"] / OUTSIDE_CELL_M).astype("Int64").astype(str))
    operator = agents["operator"].fillna("").astype(str)
    where = np.where(agents["destination_network"].notna(), agents["destination_network"].astype(str) + ":" + agents["destination_zone"].astype(str), cell("destination"))
    origin = np.where(agents["case"] == "destination", cell("home"), operator + ":" + agents["home_zone"].fillna("").astype(str))
    agents["od"] = (agents["case"] + "|" + origin
                    + "|" + np.where(agents["case"].isin(["intra", "zpass", "destination"]), operator + ":" + agents["destination_zone"].astype(str),
                                     np.where(agents["case"] == "inter", where, "")))
    return agents


# ------------------------------------------------------------------ the zones along a journey

def zones_along(journey, net, pricing, operator, stop_at_exit, detail = None, from_end = False):
    """The zones of the network operator a journey goes through, in order.

    A ride whose stops are all in the network adds its zones (as the pricing finds them, from its track). A ride that
    leaves it: with stop_at_exit (inter), the journey has left the home network, the SBB or another network takes over;
    only the zone of its boarding stop is added, and the rest of the journey is not looked at. Without (intra, zpass),
    the zones it crosses in the network (for at least MIN_ZONE_M) and those of its stops are added, and the journey
    goes on. With from_end (destination), the journey is read back from its end: the ride where it enters the network
    gives the zone of its alighting stop, and the rides before it are not looked at.

    detail: a list to which a dict per ride is added, in the order of the journey, {role, zones}: how the ride counted
    (in_network, leaves_network, after_exit; enters_network, before_entry with from_end; partly_in_network) and the
    zones it gave."""
    found, notes = [], []
    note = lambda role, zones: notes.append({"role": role, "zones": list(zones)})
    rides = journey.rides[::-1] if from_end else journey.rides
    edge, entered, before = (1, "enters_network", "before_entry") if from_end else (0, "leaves_network", "after_exit")

    def add(zones):
        new = [z for z in dict.fromkeys(zones) if z not in found]
        found.extend(new)
        return new

    for k, ride in enumerate(rides):
        ride_zones = pricing.ride(net, ride).zones
        if operator in ride_zones:
            add(ride_zones[operator])
            note("in_network", ride_zones[operator])
            continue
        stop_zones = [sorted(z for a, z in pricing._stop_zones(str(net.stop_ids[s])) if a.id == operator) for s in (ride.stops[0], ride.stops[-1])]
        if stop_at_exit:
            board = stop_zones[edge]   # where the ride leaves the network (or, from the end, enters it)
            board = [z for z in board if z in found][:1] or board[:1]
            add(board)
            note(entered, board)
            for _ in rides[k + 1:]:
                note(before, [])
            break
        crossed = pricing.tracks.crossed(net, ride).get(operator, {}) if pricing.tracks is not None else {}
        partly = [z for z, m in crossed.items() if m >= MIN_ZONE_M] + stop_zones[0][:1] + stop_zones[1][:1]
        add(partly)
        note("partly_in_network", dict.fromkeys(partly))
    if detail is not None:
        detail.extend(notes[::-1] if from_end else notes)
    return found


def _route_one(task):
    """The commute of one agent (a task: index, operator, case, home and destination (lat, lon)): (zones, needs),
    zones a tuple (empty for a journey on foot) and needs what a Léman Pass product must hold for it (see
    LemanPass.needs; None without the Léman Pass), or None when no journey is found."""
    k, operator, case, home, destination = task
    router, net, pricing, departure = _shared["router"], _shared["net"], _shared["pricing"], _shared["departure"]
    lemanpass = _shared.get("lemanpass")
    try:
        journey = router.route(home, destination, departure)
    except Exception as error:   # a place the router cannot handle: as if no journey
        logger.debug(f"No journey for {k}: {error}")
        return k, None
    if journey is None:
        return k, None
    zones = tuple(zones_along(journey, net, pricing, operator, stop_at_exit = case in ("inter", "destination"),
                              from_end = case == "destination"))
    return k, (zones, lemanpass.needs(journey, net, pricing) if lemanpass is not None else None)


def route_samples(agents, router, net, pricing, departure, samples, seed, workers = 1, lemanpass = None):
    """What is found by routing the commute of the agents of each group that needs a route: {agent index: (zones,
    needs) or None}, see _route_one. samples: the agents routed per group, drawn at random (seed); 0 or None: all."""
    needs = agents[agents["case"].isin(ROUTED)]
    rng = np.random.default_rng(seed)
    drawn = []
    for _, group in needs.groupby("od", sort = True):
        index = group.index.to_numpy()
        drawn.extend(index if not samples or len(index) <= samples else rng.choice(index, samples, replace = False))
    drawn = np.sort(np.asarray(drawn, dtype = int))
    lon, lat = TO_WGS84.transform(agents["home_x"].to_numpy()[drawn], agents["home_y"].to_numpy()[drawn])
    dlon, dlat = TO_WGS84.transform(agents["destination_x"].to_numpy()[drawn], agents["destination_y"].to_numpy()[drawn])
    tasks = [(int(k), agents.at[k, "operator"], agents.at[k, "case"], (float(a), float(b)), (float(c), float(d)))
             for k, a, b, c, d in zip(drawn, lat, lon, dlat, dlon)]
    logger.info(f"Routing {len(tasks)} commutes of {needs['od'].nunique()} zone ODs ({len(needs)} agents), {workers} worker(s)")

    _shared.update(router = router, net = net, pricing = pricing, departure = departure, lemanpass = lemanpass)
    started, results = time.time(), {}
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:   # the workers share the network of this process
            for n, (k, zones) in enumerate(pool.imap_unordered(_route_one, tasks, chunksize = 64), 1):
                results[k] = zones
                if n % 5000 == 0:
                    logger.info(f"  {n} of {len(tasks)} in {time.time() - started:.0f} s")
    else:
        for n, task in enumerate(tasks, 1):
            k, zones = _route_one(task)
            results[k] = zones
            if n % 2000 == 0:
                logger.info(f"  {n} of {len(tasks)} in {time.time() - started:.0f} s")
    logger.info(f"  {len(tasks)} commutes in {time.time() - started:.0f} s, {sum(v is None for v in results.values())} without a journey")
    return results


# ------------------------------------------------------------------ the whole

def verbund_zones(persons, work, education, zones, router, net, pricing, departure = "07:30", samples = 0, seed = 0, workers = 1,
                  lemanpass = None):
    """The zones of the Verbundabo of each agent who has one, as a DataFrame: person_id, purpose, case, home_network,
    destination_network, operator, zones (e.g. "110,121"), verbund_zones ("ZVV:110,121", empty when none is found),
    method (own_route: the zones of the agent's journey; group_route: of the journey of another agent of its group;
    home_zone, same_zone; fallback: the home and destination zones, no journey found; none: no network), subscription
    (Verbundabo, or Léman Pass for a commute into the Léman Pass area abroad), lemanpass (its trip type) and
    lemanpass_annual_chf.

    persons, work, education: the persons of synthesis.population.models.subscriptions and the work and education places of
    synthesis.population.spatial.primary.locations; zones: the zone polygons (tracks.zones);
    router, net, pricing: the router on the network of a date, that network, and the Pricing with the tracks;
    samples: the commutes routed per group (0: every agent's own commute);
    lemanpass: the LemanPass (None: no Léman Pass subscriptions)."""

    # The commutes and their zones.

    agents = classify(commutes(persons, work, education), zones)
    logger.info("Agents with a Verbundabo by case: " + ", ".join(f"{c} {n}" for c, n in agents["case"].value_counts().items()))

    # The zones along the commutes of the agents of each group (all of them unless samples is set).

    routed = route_samples(agents, router, net, pricing, departure, samples, seed, workers, lemanpass)

    # The zones of each agent: its own journey, else one of its group, else the zones of its home and destination.

    rng = np.random.default_rng(seed + 1)
    found = {}
    for od, group in agents[agents["case"].isin(ROUTED)].groupby("od", sort = True):
        options = [routed[k] for k in group.index if routed.get(k) is not None]
        for k in group.index:
            if routed.get(k) is not None:
                found[k] = (routed[k], "own_route")
            elif options:
                found[k] = (options[rng.integers(len(options))], "group_route")
    zone_lists, methods, needs = [], [], []
    for k, case, home_zone, dest_zone in zip(agents.index, agents["case"], agents["home_zone"], agents["destination_zone"]):
        (journey_zones, needed), method = found.get(k, ((None, None), None))
        zones_k, method = final_zones(case, home_zone, dest_zone, journey_zones, method)
        zone_lists.append(zones_k)
        methods.append(method)
        needs.append(needed)

    # The Léman Pass subscription of the commutes into its area abroad: the cheapest for the age of the agent.

    ages = agents["age"] if "age" in agents else pd.Series(np.nan, index = agents.index)
    products = [lemanpass.cheapest(*n, ticket = "annual", profile = subscription_profile(a)) if lemanpass is not None and n else None
                for n, a in zip(needs, ages)]
    agents["subscription"] = ["Léman Pass" if p else "Verbundabo" for p in products]
    agents["lemanpass"] = [p[0] if p else "" for p in products]
    agents["lemanpass_annual_chf"] = [p[1] if p else np.nan for p in products]
    uncovered = sum(1 for n, p in zip(needs, products) if n and not p)
    if lemanpass is not None:
        logger.info(f"Léman Pass subscriptions: {sum(1 for p in products if p)} agents" + (f", {uncovered} more with no product covering their commute" if uncovered else ""))

    # Return the zones of each agent, in the format of the subscriptions.

    agents["zones"] = [",".join(z) for z in zone_lists]
    agents["n_zones"] = [len(z) for z in zone_lists]
    agents["method"] = methods
    agents["verbund_zones"] = np.where(agents["n_zones"] > 0, agents["operator"].fillna("").astype(str) + ":" + agents["zones"], "")
    return agents[["person_id", "pt_subscription", "purpose", "case", "home_network", "destination_network", "operator", "home_zone",
                   "destination_zone", "od", "zones", "n_zones", "verbund_zones", "method", "subscription", "lemanpass", "lemanpass_annual_chf"]]


def final_zones(case, home_zone, destination_zone, routed = None, method = None):
    """The zones of a subscription, sorted, and how they were found, from the case of the agent, its home and
    destination zones, and the zones along a journey (routed, None without one) found by method (own_route,
    group_route)."""
    if case == "outside":
        zones, method = (), "none"
    elif case in ("home", "same_zone"):
        zones, method = (home_zone,), "home_zone" if case == "home" else "same_zone"
    elif case == "destination":
        # the destination zone is always in it, as the home zone in the other cases
        zones = (destination_zone,) + tuple(routed or ())
        method = method if routed is not None else "fallback"
    elif routed is not None:
        # the home zone is always in the subscription (the journey can start with a walk to a stop of the next zone), and
        # so is the destination zone of the same network (intra, zpass: the journey can end with a walk from a stop of
        # the zone before); for inter, the destination is in another network
        zones = (home_zone,) + tuple(routed) + ((destination_zone,) if case != "inter" and isinstance(destination_zone, str) else ())
    else:
        zones, method = (home_zone,) + ((destination_zone,) if case != "inter" and isinstance(destination_zone, str) else ()), "fallback"
    return tuple(sorted(dict.fromkeys(zones), key = lambda v: (len(v), v))), method


def commute_zones(home, destination, zones, router, net, pricing, departure = "07:30", cost = None, lemanpass = None, age = None):
    """The zones of the Verbundabo of one person, as verbund_zones finds them, with what led to them: home and
    destination as (lat, lon), destination None for a person without a commute; cost: the Cost of the routing (default:
    that of the router); lemanpass: the LemanPass (None: no Léman Pass subscriptions); age: for its price (none: adult).

    A dict: case, operator, home_network, destination_network, home_zone, destination_zone, zones (sorted list),
    verbund_zones ("ZVV:110,121"), method, subscription, lemanpass (the trip type of the Léman Pass subscription, or
    None), lemanpass_annual_chf, lemanpass_needs (what its two ends must hold: [first, last], or None), the journey of
    the commute (a Journey, or None when it is not routed) and rides (for each ride of the journey: how it counted, see
    zones_along)."""
    (hx, dx), (hy, dy) = TO_LV95.transform([home[1], destination[1] if destination else np.nan], [home[0], destination[0] if destination else np.nan])
    agent = pd.DataFrame({"person_id": [0], "home_x": [hx], "home_y": [hy], "pt_subscription": [VERBUND_CODES[0]],
                          "purpose": ["work" if destination else "none"], "destination_x": [dx], "destination_y": [dy]})
    agent = classify(agent, zones).iloc[0]
    journey, rides, routed, needs, product = None, [], None, None, None
    if agent["case"] in ROUTED:
        journey = router.route(tuple(home), tuple(destination), departure, cost = cost)
        if journey is not None:
            routed = zones_along(journey, net, pricing, agent["operator"], stop_at_exit = agent["case"] in ("inter", "destination"),
                                 detail = rides, from_end = agent["case"] == "destination")
            needs = lemanpass.needs(journey, net, pricing) if lemanpass is not None else None
            product = lemanpass.cheapest(*needs, ticket = "annual", profile = subscription_profile(age)) if needs else None
    found, method = final_zones(agent["case"], agent["home_zone"], agent["destination_zone"], routed, "own_route")
    clean = lambda v: v if isinstance(v, str) else None
    return {"case": agent["case"], "operator": clean(agent["operator"]), "home_network": clean(agent["home_network"]),
            "destination_network": clean(agent["destination_network"]), "home_zone": clean(agent["home_zone"]),
            "destination_zone": clean(agent["destination_zone"]), "zones": list(found), "method": method,
            "verbund_zones": f"{agent['operator']}:{','.join(found)}" if found else "",
            "subscription": "Léman Pass" if product else "Verbundabo", "lemanpass": product[0] if product else None,
            "lemanpass_annual_chf": product[1] if product else None, "lemanpass_needs": [sorted(n) for n in needs[:2]] if needs else None,
            "journey": journey, "rides": rides}


def od_matrix(result):
    """The agents by group (zone OD): their number, the share of each method and the most frequent zones."""
    return (result.groupby(["od", "case"], dropna = False)
            .agg(agents = ("person_id", "size"), operator = ("operator", "first"), home_zone = ("home_zone", "first"),
                 destination_network = ("destination_network", "first"), destination_zone = ("destination_zone", "first"),
                 routed = ("method", lambda m: int((m == "own_route").sum())),
                 variants = ("zones", "nunique"), zones = ("zones", lambda z: z.mode().iloc[0] if len(z) else ""),
                 mean_zones = ("n_zones", "mean"))
            .reset_index().sort_values("agents", ascending = False))
