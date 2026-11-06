"""The polygons of the fare zones, drawn from the municipalities.

No official map of the zones of the Swiss fare networks is used. Instead, the zones are rebuilt from two sources: the
map of the municipalities and the fare networks they belong to (tarifverbundkarte.geojson), and the zone of each stop
(gtfs_zones.csv of the pricing inputs). For each fare network:

- a municipality belongs to the zone of most of its stops (at least MAJORITY of them), not counting the stops on the
  border of two zones (those listed in both);
- a municipality whose stops are in several zones (Bern, Lausanne...) is split between them: each place goes to the
  zone of its nearest stop;
- a municipality whose stops are all on the border of zones (Wichtrach: Libero 626 and 710) is split between these
  zones, each part going to the nearest one;
- a municipality without any stop of the network takes the zone of its neighbour with the longest common border, for
  the networks the map says it is in (and their Z-Pass).

The map has only Swiss municipalities. The zones abroad (the French zones of Unireso) come from a list of communes and
their zones (unireso_france_zones.csv: insee, commune, zone, and optionally network, else Unireso) and the outlines of
these communes (communes_france.gpkg: insee, name, geometry). Only the listed communes are in these zones; a commune
listed in several zones (Pougny: 240 and 250) is split by the nearest of its stops in them.

The polygons are in the Swiss projection (LV95, EPSG:2056), in metres. They are used to find the zones a ride goes
through (tracks.py), to draw the zones on the map, and to know whether a traveller lives in a network's area.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

logger = logging.getLogger("synpp")

# the names of the networks of the map, as the pricing inputs call them (the others are the same)
NETWORK_NAMES = {"A-Welle": "Awelle", "ONDE VERTE": "OndeVerte", "engadin mobil": "EngadinMobil", "unireso": "Unireso",
                 "Tarifverbund Schwyz": "TVSZ", "Tarifverbund Zug": "ZVB", "OSTWIND": "Ostwind", "Tarifverbund Davos": "Davos",
                 "Tarifverbund Klosters": "Klosters"}
MAJORITY = 0.9   # the share of its stops a zone needs for a whole municipality to be in it (else it is split)
FOREIGN_NETWORK = "Unireso"   # the network of the communes abroad when their list has no network column


def zone_polygons(pricing, net, municipalities_path, foreign_communes_path = None, foreign_zones_path = None):
    """The zone polygons, as a GeoDataFrame with one row per zone of each network: network, zone, geometry,
    municipalities (the names of the municipalities in it, the one with the most stops first) and name (that first one).

    foreign_communes_path, foreign_zones_path: the outlines of communes abroad and the list of their zones, for the
    zones outside Switzerland (both needed; see the top of this module)."""
    towns = gpd.read_file(municipalities_path).to_crs(2056)[["id", "name", "partners", "geometry"]]
    towns["networks"] = towns["partners"].map(lambda p: {NETWORK_NAMES.get(x.strip(), x.strip()) for x in (p if isinstance(p, str) else "").split(",") if x.strip()})

    # the stops with one zone in a network, and the stops at the border of zones (several zones in the same network)
    rows, border_rows = [], []
    for s, stop_id in enumerate(net.stop_ids):
        by_network = {}
        for auth, zone in pricing._stop_zones(str(stop_id)):
            if auth.id != "SBB":
                by_network.setdefault(auth.id, set()).add(zone)
        rows += [(s, n, next(iter(z))) for n, z in by_network.items() if len(z) == 1]
        border_rows += [(s, n, z) for n, z in by_network.items() if len(z) > 1]

    def placed(rows, columns):
        """The rows (one per stop) with the position of their stop."""
        table = pd.DataFrame(rows, columns = columns)
        points = gpd.GeoSeries(gpd.points_from_xy(net.stop_lon[table["stop"]], net.stop_lat[table["stop"]]), crs = 4326).to_crs(2056)
        return gpd.GeoDataFrame(table, geometry = points.values, crs = 2056)

    def located(rows, columns):
        """The rows (one per stop) with the position of their stop and the municipality it is in (column id)."""
        return gpd.sjoin(placed(rows, columns), towns[["id", "geometry"]], predicate = "within").drop(columns = "index_right")
    stops = located(rows, ["stop", "network", "zone"])
    border = located(border_rows, ["stop", "network", "zones"])

    # the zones of the communes abroad only come from their list: a Swiss stop with such a zone (Collonges VS, named
    # like Collonges in zone 230 of Unireso) does not put its municipality in it
    abroad = _foreign_communes(foreign_communes_path, foreign_zones_path)
    stops, border = _without_foreign_zones(stops, border, set(zip(abroad["network"], abroad["zone"])))

    # each municipality in the zone most of its stops are in, or split by nearest stop (Voronoi cells of its stops)
    pieces = []   # (town id, network, zone, geometry)
    votes = stops.groupby(["id", "network", "zone"]).size().rename("n").reset_index()
    votes["share"] = votes["n"] / votes.groupby(["id", "network"])["n"].transform("sum")
    best = votes.sort_values("n", ascending = False).drop_duplicates(["id", "network"])
    shapes = towns.set_index("id").geometry
    split = 0
    for town, network, zone, share in zip(best["id"], best["network"], best["zone"], best["share"]):
        if share >= MAJORITY:
            pieces.append((town, network, zone, shapes[town]))
            continue
        split += 1   # the town by the nearest stop of the network
        inside = stops[(stops["id"] == town) & (stops["network"] == network)]
        pieces += [(town, network, z, cell) for z, cell in _split_by_stops(shapes[town], inside)]

    # the communes abroad, in the zones of their list: all of a commune in its zone, or, listed in several zones, split
    # by the nearest of its stops in them; when these stops are all in one of the zones (Pougny: 240), all of it in that
    # zone, and without such stops, in the first zone listed
    if len(abroad):
        every = placed(rows, ["stop", "network", "zone"])
        for (town, network), group in abroad.groupby(["id", "network"], sort = False):
            shape, listed = group.geometry.iloc[0], sorted(group["zone"])
            inside = every[(every["network"] == network) & every["zone"].isin(listed)]
            inside = inside[inside.within(shape)]
            if len(listed) == 1 or inside["zone"].nunique() < 2:
                pieces.append((town, network, inside["zone"].iloc[0] if len(inside) else listed[0], shape))
            else:
                pieces += [(town, network, z, cell) for z, cell in _split_by_stops(shape, inside)]
        logger.info(f"  {abroad['id'].nunique()} communes abroad in {abroad[['network', 'zone']].drop_duplicates().shape[0]} zones")

    # the towns with only stops at the border of zones (Wichtrach, Libero 626 and 710): the border goes through them.
    # A grid of points covers the town, each point goes to the nearest of these zones (as drawn so far), and the town is
    # cut along the cells of these points
    have = {(t, n) for t, n, _, _ in pieces}
    between = 0
    for (town, network), group in border.groupby(["id", "network"]):
        if (town, network) in have:
            continue
        candidates = set().union(*group["zones"])
        known = {z: shapely.union_all([g for t, n, zz, g in pieces if n == network and zz == z]) for z in candidates}
        known = {z: g for z, g in known.items() if not g.is_empty}
        if not known:
            continue
        shape = shapes[town]
        step = max(100.0, np.sqrt(shape.area) / 40)
        x0, y0, x1, y1 = shape.bounds
        grid = shapely.points(*np.meshgrid(np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)))
        grid = grid.ravel()[shapely.contains(shape, grid.ravel())]
        if len(grid) == 0:
            grid = np.array([shape.representative_point()])
        names = sorted(known)
        nearest = np.array(names)[np.argmin([shapely.distance(known[z], grid) for z in names], axis = 0)]
        cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(grid), extend_to = shape))
        cells = gpd.GeoDataFrame(geometry = list(cells), crs = 2056)
        cells = gpd.sjoin(cells, gpd.GeoDataFrame({"zone": nearest}, geometry = list(grid), crs = 2056), predicate = "contains")
        for z, cell in cells.dissolve("zone").geometry.items():
            pieces.append((town, network, z, cell.intersection(shape)))
        have.add((town, network))
        between += 1

    # the towns without stops of a network: the zone of their neighbour with the longest common border (the neighbours
    # are tried from the longest border down), for the networks the map gives them
    zone_of = {}   # town -> {network: its first zone}
    for t, n, z, _ in pieces:
        zone_of.setdefault(t, {}).setdefault(n, z)
    neighbours = gpd.sjoin(towns[["id", "geometry"]], towns[["id", "geometry"]], predicate = "touches")
    border = shapely.length(shapely.intersection(shapes[neighbours["id_left"]].values, shapes[neighbours["id_right"]].values))
    neighbours = neighbours.assign(border = border).sort_values("border", ascending = False)
    networks_of = towns.set_index("id")["networks"]
    filled = 0
    for town, group in neighbours.groupby("id_left", sort = False):
        wanted = networks_of[town]
        if not wanted:
            continue
        for other in group["id_right"]:
            for network, zone in zone_of.get(other, {}).items():
                if (town, network) in have or not (network in wanted or network.startswith("ZPass")):
                    continue
                pieces.append((town, network, zone, shapes[town]))
                have.add((town, network))
                filled += 1

    # the pieces merged into one polygon per zone, named after their municipalities (the one with the most stops first)
    result = gpd.GeoDataFrame(pd.DataFrame(pieces, columns = ["id", "network", "zone", "geometry"]), crs = 2056)
    names_of = pd.concat([towns.set_index("id")["name"], abroad.drop_duplicates("id").set_index("id")["name"]])
    result["town"] = result["id"].map(names_of)
    counts = stops.groupby(["network", "zone", "id"]).size().rename("stops")
    result = result.join(counts, on = ["network", "zone", "id"]).fillna({"stops": 0})
    names = result.sort_values("stops", ascending = False).groupby(["network", "zone"])["town"].agg(lambda t: list(dict.fromkeys(t)))
    result = result.dissolve(["network", "zone"]).reset_index()[["network", "zone", "geometry"]]
    result["geometry"] = result.geometry.buffer(0)   # repairs the small overlaps and gaps of the merged pieces
    result["municipalities"] = [names[(n, z)] for n, z in zip(result["network"], result["zone"])]
    result["name"] = result["municipalities"].map(lambda m: m[0])
    logger.info(f"  {len(result)} zone polygons: {len(best)} municipality-network pairs from their stops ({split} split by stop), "
                f"{between} split between the zones of their border stops, {filled} from a neighbour")
    return result


def _without_foreign_zones(located, border, foreign):
    """The stops in Swiss municipalities (located: one zone per network; border: several) without the zones of the
    communes abroad (foreign: {(network, zone)}). A border stop left with one zone is moved to located."""
    located = located[[(n, z) not in foreign for n, z in zip(located["network"], located["zone"])]]
    border = border.assign(zones = [{z for z in zones if (n, z) not in foreign} for n, zones in zip(border["network"], border["zones"])])
    single = border["zones"].map(len) == 1
    moved = border[single].assign(zone = border.loc[single, "zones"].map(lambda z: next(iter(z)))).drop(columns = "zones")
    return pd.concat([located, moved[located.columns]]), border[border["zones"].map(len) > 1]


def _split_by_stops(shape, stops):
    """A place split by the zone of its nearest stop (the Voronoi cells of the stops): [(zone, geometry)]. stops: a
    GeoDataFrame of the stops with their zone (EPSG:2056)."""
    stops = stops.drop_duplicates("geometry")
    cells = shapely.voronoi_polygons(shapely.multipoints(stops.geometry.values), extend_to = shape)
    cells = gpd.GeoDataFrame(geometry = list(shapely.get_parts(cells)), crs = 2056)
    cells = gpd.sjoin(cells, stops[["zone", "geometry"]], predicate = "contains")
    return [(z, cell.intersection(shape)) for z, cell in cells.dissolve("zone").geometry.items()]


def _foreign_communes(communes_path, zones_path):
    """The communes abroad and their zones, one row per commune and zone: id ("FR" and the INSEE code), name, network,
    zone and geometry (EPSG:2056); empty without both files. A listed commune without an outline is left out, with a
    warning."""
    if not (communes_path and zones_path):
        return gpd.GeoDataFrame({"id": [], "name": [], "network": [], "zone": []}, geometry = [], crs = 2056)
    listed = pd.read_csv(zones_path, dtype = str).apply(lambda s: s.str.strip())
    if "network" not in listed:
        listed["network"] = FOREIGN_NETWORK
    communes = gpd.read_file(communes_path).to_crs(2056)[["insee", "name", "geometry"]]
    missing = sorted(set(listed["insee"]) - set(communes["insee"]))
    if missing:
        logger.warning(f"  {len(missing)} communes of {zones_path} have no outline in {communes_path}: {', '.join(missing)}")
    result = communes.merge(listed[["insee", "network", "zone"]], on = "insee")
    result["id"] = "FR" + result["insee"]
    return result[["id", "name", "network", "zone", "geometry"]]


def foreign_stop_zones(net, zones, communes_path, zones_path):
    """The zones of the stops in the communes abroad, as a DataFrame stop_id, network, zone: the zone of the commune of
    each stop, or, for a commune listed in several zones, the one whose polygon (of zones, the zone polygons) holds the
    stop (else the first listed). Empty without both files."""
    abroad = _foreign_communes(communes_path, zones_path)
    if not len(abroad):
        return pd.DataFrame({"stop_id": [], "network": [], "zone": []})
    points = gpd.GeoDataFrame({"stop_id": net.stop_ids.astype(str)},
                              geometry = gpd.GeoSeries(gpd.points_from_xy(net.stop_lon, net.stop_lat), crs = 4326).to_crs(2056).values, crs = 2056)
    hits = gpd.sjoin(points, abroad[["network", "zone", "geometry"]], predicate = "within").drop(columns = "index_right")
    several = hits.duplicated(["stop_id", "network"], keep = False)
    if several.any():   # a commune in several zones: the zone whose polygon holds the stop
        inside = gpd.sjoin(hits[several].drop(columns = "zone"), zones[["network", "zone", "geometry"]], predicate = "within")
        inside = inside[inside["network_left"] == inside["network_right"]].rename(columns = {"network_left": "network"})
        chosen = hits[several].merge(inside[["stop_id", "network", "zone"]], on = ["stop_id", "network", "zone"])
        first = hits[several].sort_values("zone").drop_duplicates(["stop_id", "network"])
        hits = pd.concat([hits[~several], chosen, first]).drop_duplicates(["stop_id", "network"])
    return pd.DataFrame(hits[["stop_id", "network", "zone"]]).reset_index(drop = True)
