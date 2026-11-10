"""The polygons of the fare zones, rebuilt from the municipalities and the zones of the stops.

No official map of the zones of the Swiss fare networks is used. The zones are rebuilt from two sources: the map of
the municipalities and the fare networks they belong to (tarifverbundkarte.geojson, SBB open data), and the zone of
each stop (gtfs_zones.csv of data.pt_pricing.pt_pricing). For each fare network:

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

The polygons are in the Swiss projection (LV95, EPSG:2056). Ported from gtfs_router/zones.py of the Swiss PT explorer.
"""
import html
import logging
import re
from collections import Counter

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from branca.element import MacroElement
from jinja2 import Template

from utils.gtfs_router.pricing import station_id
from utils.gtfs_router.zones import foreign_communes

logger = logging.getLogger("synpp")

# the names of the networks of the map, as the pricing inputs call them (the others are the same)
NETWORK_NAMES = {"A-Welle": "Awelle", "ONDE VERTE": "OndeVerte", "engadin mobil": "EngadinMobil", "unireso": "Unireso",
                 "Tarifverbund Schwyz": "TVSZ", "Tarifverbund Zug": "ZVB", "OSTWIND": "Ostwind", "Tarifverbund Davos": "Davos",
                 "Tarifverbund Klosters": "Klosters"}
MAJORITY = 0.9   # the share of its stops a zone needs for a whole municipality to be in it (else it is split)

# the background of the interactive map, and the colours of the zones (cycled within each network)
ESRI_TILES = {"Esri World Topo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
              "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"}
ESRI_ATTRIBUTION = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, OpenStreetMap contributors and the GIS user community"
ZONE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MAP_SIMPLIFY_M = 20   # the tolerance of the simplified outlines of the interactive map, in metres
MAP_OVERLAP_MIN_M2 = 10_000   # the smallest overlap of two networks shown on the map (smaller ones are slivers)


def configure(context):
    context.config("data_path")
    context.config("pt_zones_municipalities_file", default = "pt_pricing/tarifverbundkarte.geojson")
    context.config("pt_zones_foreign_communes_file", default = "pt_pricing/communes_france.gpkg")
    context.config("pt_zones_foreign_zones_file", default = "pt_pricing/unireso_france_zones.csv")

    context.stage("data.pt_pricing.pt_pricing")
    context.stage("data.gtfs.cleaned")


def execute(context):

    # Load the municipalities with their fare networks.

    data_path = context.config("data_path")
    towns = gpd.read_file(f"{data_path}/{context.config('pt_zones_municipalities_file')}").to_crs(2056)[["id", "name", "partners", "geometry"]]
    towns["networks"] = towns["partners"].map(lambda p: {NETWORK_NAMES.get(x.strip(), x.strip()) for x in (p if isinstance(p, str) else "").split(",") if x.strip()})
    shapes = towns.set_index("id").geometry

    # Load the stops with one zone in a network, and the stops at the border of zones (several zones in the same network).

    _, zones_path, _ = context.stage("data.pt_pricing.pt_pricing")
    zones_by_stop = read_stop_zones(zones_path)
    stops = read_stops(f"{context.path('data.gtfs.cleaned')}/output/stops.txt")

    rows, border_rows = [], []
    for stop_id, geometry in zip(stops["stop_id"], stops.geometry):
        by_network = {}
        for network, zone in zones_by_stop.get(stop_id, set()) | zones_by_stop.get(simplified(stop_id), set()):
            by_network.setdefault(network, set()).add(zone)
        rows += [(stop_id, n, next(iter(z)), geometry) for n, z in by_network.items() if len(z) == 1]
        border_rows += [(stop_id, n, z, geometry) for n, z in by_network.items() if len(z) > 1]
    if not rows:
        raise ValueError(f"No stop of {context.path('data.gtfs.cleaned')}/output/stops.txt has a zone in {zones_path}: their stop ids "
                         "do not match (see the check of data.pt_pricing.pt_pricing)")

    every = gpd.GeoDataFrame(pd.DataFrame(rows, columns = ["stop", "network", "zone", "geometry"]), crs = 2056)
    border = gpd.GeoDataFrame(pd.DataFrame(border_rows, columns = ["stop", "network", "zones", "geometry"]), crs = 2056)
    located = gpd.sjoin(every, towns[["id", "geometry"]], predicate = "within").drop(columns = "index_right")
    border = gpd.sjoin(border, towns[["id", "geometry"]], predicate = "within").drop(columns = "index_right")

    # The zones of the communes abroad only come from their list: a Swiss stop with such a zone (Collonges VS, named
    # like Collonges in zone 230 of Unireso) does not put its municipality in it.

    abroad = foreign_communes(f"{data_path}/{context.config('pt_zones_foreign_communes_file')}",
                                   f"{data_path}/{context.config('pt_zones_foreign_zones_file')}")
    foreign = set(zip(abroad["network"], abroad["zone"]))
    swiss_stops = set(located["stop"]) | set(border["stop"])
    located, border = without_foreign_zones(located, border, foreign)

    # Place each municipality in the zone most of its stops are in, or split it by nearest stop.

    pieces = []   # (town id, network, zone, geometry)
    votes = located.groupby(["id", "network", "zone"]).size().rename("n").reset_index()
    votes["share"] = votes["n"] / votes.groupby(["id", "network"])["n"].transform("sum")
    best = votes.sort_values("n", ascending = False).drop_duplicates(["id", "network"])
    split = 0
    for town, network, zone, share in zip(best["id"], best["network"], best["zone"], best["share"]):
        if share >= MAJORITY:
            pieces.append((town, network, zone, shapes[town]))
            continue
        split += 1
        inside = located[(located["id"] == town) & (located["network"] == network)]
        pieces += [(town, network, z, cell) for z, cell in split_by_stops(shapes[town], inside)]

    # Place the communes abroad in the zones of their list: all of a commune in its zone or, listed in several zones,
    # split by the nearest of its stops in them. When these stops are all in one of the zones (Pougny: 240), all of it
    # goes to that zone, and without such stops, to the first zone listed.

    for (town, network), group in abroad.groupby(["id", "network"], sort = False):
        shape, listed = group.geometry.iloc[0], sorted(group["zone"])
        inside = every[(every["network"] == network) & every["zone"].isin(listed)]
        inside = inside[inside.within(shape)]
        if len(listed) == 1 or inside["zone"].nunique() < 2:
            pieces.append((town, network, inside["zone"].iloc[0] if len(inside) else listed[0], shape))
        else:
            pieces += [(town, network, z, cell) for z, cell in split_by_stops(shape, inside)]
    logger.info(f"  {abroad['id'].nunique()} communes abroad in {abroad[['network', 'zone']].drop_duplicates().shape[0]} zones")

    # Split the towns with only stops at the border of zones (Wichtrach, Libero 626 and 710): a grid of points covers
    # the town, each point goes to the nearest of these zones (as drawn so far), and the town is cut along their cells.

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

    # Give the towns without stops of a network the zone of their neighbour with the longest common border (tried
    # from the longest border down), for the networks the map gives them.

    zone_of = {}   # town -> {network: its first zone}
    for t, n, z, _ in pieces:
        zone_of.setdefault(t, {}).setdefault(n, z)
    neighbours = gpd.sjoin(towns[["id", "geometry"]], towns[["id", "geometry"]], predicate = "touches")
    lengths = shapely.length(shapely.intersection(shapes[neighbours["id_left"]].values, shapes[neighbours["id_right"]].values))
    neighbours = neighbours.assign(border = lengths).sort_values("border", ascending = False)
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

    # Merge the pieces into one polygon per zone, named after its municipalities (the one with the most stops first).

    result = gpd.GeoDataFrame(pd.DataFrame(pieces, columns = ["id", "network", "zone", "geometry"]), crs = 2056)
    names_of = pd.concat([towns.set_index("id")["name"], abroad.drop_duplicates("id").set_index("id")["name"]])
    result["town"] = result["id"].map(names_of)
    counts = located.groupby(["network", "zone", "id"]).size().rename("stops")
    result = result.join(counts, on = ["network", "zone", "id"]).fillna({"stops": 0})
    names = result.sort_values("stops", ascending = False).groupby(["network", "zone"])["town"].agg(lambda t: list(dict.fromkeys(t)))
    result = result.dissolve(["network", "zone"]).reset_index()[["network", "zone", "geometry"]]
    result["geometry"] = result.geometry.buffer(0)   # repairs the small overlaps and gaps of the merged pieces
    result["municipalities"] = [names[(n, z)] for n, z in zip(result["network"], result["zone"])]
    result["name"] = result["municipalities"].map(lambda m: m[0])
    logger.info(f"  {len(result)} zone polygons: {len(best)} municipality-network pairs from their stops ({split} split by stop), "
                f"{between} split between the zones of their border stops, {filled} from a neighbour")

    # Write the zones and their interactive map, and return the zones.

    result.assign(municipalities = result["municipalities"].map(", ".join)).to_file(f"{context.path()}/pt_zones.gpkg", driver = "GPKG")
    write_map(result, border_stops(stops, zones_by_stop, swiss_stops, foreign), f"{context.path()}/pt_zones.html")

    return result


def read_stop_zones(path):
    """The zones of each stop of gtfs_zones.csv, as {stop id: {(network, zone)}}: the networks of the 'tarif network'
    column, and the zones of the 'zones' column ("ZVV:110, ..."), without the SBB."""
    df = pd.read_csv(path, dtype = str, keep_default_na = False)
    networks = {n for text in df["tarif network"] for n in re.sub(r"[\[\]' ]", "", text).split(",") if n and n != "nan"}
    zones_by_stop = {}
    for stop_id, text in zip(df["stop_id"].str.strip(), df["zones"]):
        for entry in text.split(","):
            parts = [p.strip() for p in entry.split(":")]
            if len(parts) == 2 and parts[0] in networks and parts[0] != "SBB":
                zones_by_stop.setdefault(stop_id, set()).add((parts[0], parts[1]))
    return zones_by_stop


def simplified(stop_id):
    """The station of a stop id, as getZones in eqasim-java (8501120:0:1 -> 8501120), with the DiDok number of a SLOID
    (utils.gtfs_router.pricing.station_id)."""
    return station_id(stop_id)


def read_stops(path):
    """The GTFS stops: stop_id, stop_name and geometry (EPSG:2056)."""
    stops = pd.read_csv(path, dtype = {"stop_id": str, "stop_name": str}, usecols = ["stop_id", "stop_name", "stop_lat", "stop_lon"])
    stops["stop_id"] = stops["stop_id"].str.strip()
    points = gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"])
    return gpd.GeoDataFrame(stops[["stop_id", "stop_name"]], geometry = points, crs = 4326).to_crs(2056)


def border_stops(stops, zones_by_stop, swiss_stops, foreign):
    """The stops at the border of zones or networks, for the map: a GeoDataFrame of stop_name, zones (sorted (network,
    zone) pairs) and geometry, one row per name and zones. A stop is at a border when it is in two zones of a network or
    in two regional networks (not counting the Z-Pass). The Swiss stops (swiss_stops) do not have the zones abroad
    (foreign), as in the zone polygons."""
    rows = []
    for stop_id, name, geometry in zip(stops["stop_id"], stops["stop_name"], stops.geometry):
        pairs = zones_by_stop.get(stop_id, set()) | zones_by_stop.get(simplified(stop_id), set())
        if stop_id in swiss_stops:
            pairs = {p for p in pairs if p not in foreign}
        regional = {n for n, _ in pairs if not n.startswith("ZPass")}
        if len(regional) > 1 or any(c > 1 for c in Counter(n for n, _ in pairs).values()):
            rows.append((name, tuple(sorted(pairs)), geometry))
    result = gpd.GeoDataFrame(pd.DataFrame(rows, columns = ["stop_name", "zones", "geometry"]), crs = 2056)
    return result.drop_duplicates(["stop_name", "zones"]).reset_index(drop = True)


def without_foreign_zones(located, border, foreign):
    """The stops in Swiss municipalities (located: one zone per network; border: several) without the zones of the
    communes abroad (foreign: {(network, zone)}). A border stop left with one zone is moved to located."""
    located = located[np.array([(n, z) not in foreign for n, z in zip(located["network"], located["zone"])], dtype = bool)]
    border = border.assign(zones = [{z for z in zones if (n, z) not in foreign} for n, zones in zip(border["network"], border["zones"])])
    single = border["zones"].map(len) == 1
    moved = border[single].assign(zone = border.loc[single, "zones"].map(lambda z: next(iter(z)))).drop(columns = "zones")
    return pd.concat([located, moved[located.columns]]), border[border["zones"].map(len) > 1]


def split_by_stops(shape, stops):
    """A place split by the zone of its nearest stop (the Voronoi cells of the stops): [(zone, geometry)]. stops: a
    GeoDataFrame of the stops with their zone (EPSG:2056)."""
    stops = stops.drop_duplicates("geometry")
    cells = shapely.voronoi_polygons(shapely.multipoints(stops.geometry.values), extend_to = shape)
    cells = gpd.GeoDataFrame(geometry = list(shapely.get_parts(cells)), crs = 2056)
    cells = gpd.sjoin(cells, stops[["zone", "geometry"]], predicate = "contains")
    return [(z, cell.intersection(shape)) for z, cell in cells.dissolve("zone").geometry.items()]


def write_map(zones, stops, path):
    """An interactive map of the zone polygons on an Esri background: one layer per network (the Z-Pass hidden at
    first), each zone with its network, number and name as a tooltip; the places in several regional networks; and the
    stops at the border of zones or networks (stops, see border_stops). A click anywhere lists all the zones of all the
    networks at that place (also those of the hidden layers) and outlines them."""
    names = {(n, z): name for n, z, name in zip(zones["network"], zones["zone"], zones["name"])}
    overlaps = network_overlaps(zones)
    zones = zones.assign(geometry = zones.geometry.simplify(MAP_SIMPLIFY_M)).to_crs(4326)
    x0, y0, x1, y1 = zones.total_bounds
    m = folium.Map(location = [(y0 + y1) / 2, (x0 + x1) / 2], zoom_start = 8, tiles = None, prefer_canvas = True)
    for i, (name, url) in enumerate(ESRI_TILES.items()):
        folium.TileLayer(tiles = url, attr = ESRI_ATTRIBUTION, name = name, max_zoom = 19, show = i == 0).add_to(m)

    # One layer per network: the click on a zone goes to the map, which lists every zone at that place

    zone_layers = []
    for network, group in sorted(zones.groupby("network"), key = lambda g: (g[0].startswith("ZPass"), g[0])):
        group = group.sort_values("zone").reset_index(drop = True)
        group["colour"] = [ZONE_COLOURS[i % len(ZONE_COLOURS)] for i in range(len(group))]
        group["label"] = [html.escape(f"{network} {z}: {n}") for z, n in zip(group["zone"], group["name"])]
        group["towns"] = group["municipalities"].map(lambda t: html.escape(", ".join(t)))
        group = group[["network", "zone", "label", "towns", "colour", "geometry"]]
        layer = folium.FeatureGroup(name = f"{network} ({len(group)} zones)", show = not network.startswith("ZPass"))
        geojson = folium.GeoJson(
            group,
            style_function = lambda f: {"color": "#333333", "weight": 1, "fillColor": f["properties"]["colour"], "fillOpacity": 0.45},
            highlight_function = lambda f: {"weight": 3, "fillOpacity": 0.7},
            tooltip = folium.GeoJsonTooltip(fields = ["label"], labels = False),
        )
        geojson.add_to(layer)
        layer.add_to(m)
        zone_layers.append(geojson)

    # The places in several regional networks, hatched in black

    if len(overlaps):
        overlaps = overlaps.assign(geometry = overlaps.geometry.simplify(MAP_SIMPLIFY_M)).to_crs(4326)
        overlaps["label"] = [html.escape(" / ".join(f"{n} {z}: {names[(n, z)]}" for n, z in pairs)) for pairs in overlaps["pairs"]]
        layer = folium.FeatureGroup(name = f"Places in several networks ({len(overlaps)})", show = True)
        folium.GeoJson(
            overlaps[["label", "geometry"]],
            style_function = lambda f: {"color": "#000000", "weight": 2, "dashArray": "6 4", "fillColor": "#000000", "fillOpacity": 0.15},
            tooltip = folium.GeoJsonTooltip(fields = ["label"], labels = False),
        ).add_to(layer)
        layer.add_to(m)

    # The stops at the border of zones or networks, with all their zones

    layer = folium.FeatureGroup(name = f"Stops at zone or network borders ({len(stops)})", show = True)
    for name, pairs, point in zip(stops["stop_name"], stops["zones"], stops.to_crs(4326).geometry):
        lines = "<br>".join(html.escape(f"{n} {z}" + (f": {names[(n, z)]}" if (n, z) in names else "")) for n, z in pairs)
        folium.CircleMarker(
            [point.y, point.x], radius = 5, color = "#000000", weight = 2, fill = True, fill_color = "#ffffff", fill_opacity = 1,
            tooltip = html.escape(f"{name} ({len(pairs)} zones)"), popup = folium.Popup(f"<b>{html.escape(name)}</b><br>{lines}", max_width = 450),
            bubbling_mouse_events = False,
        ).add_to(layer)
    layer.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)
    ClickZones(m, zone_layers).add_to(m)
    m.fit_bounds([[y0, x0], [y1, x1]])
    m.save(path)


def network_overlaps(zones):
    """The places in zones of several regional networks (not the Z-Pass), larger than MAP_OVERLAP_MIN_M2: a GeoDataFrame
    of pairs (the sorted (network, zone) of the place) and geometry (EPSG:2056)."""
    regional = zones.loc[~zones["network"].str.startswith("ZPass"), ["network", "zone", "geometry"]]
    pieces = gpd.overlay(regional, regional, how = "intersection", keep_geom_type = True)
    pieces = pieces[pieces["network_1"] < pieces["network_2"]]
    pieces = pieces[pieces.area >= MAP_OVERLAP_MIN_M2]
    pairs = [tuple(sorted([(a, b), (c, d)])) for a, b, c, d in zip(pieces["network_1"], pieces["zone_1"], pieces["network_2"], pieces["zone_2"])]
    return gpd.GeoDataFrame({"pairs": pairs}, geometry = pieces.geometry.values, crs = 2056).reset_index(drop = True)


class ClickZones(MacroElement):
    """A click on the map opens a popup with every zone of the zone layers that holds the place, and outlines them."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this.map }};
            var layers = [{{ this.layers }}];
            var outlines = L.layerGroup().addTo(map);
            function inRing(x, y, ring) {
                var inside = false;
                for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
                    var xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
                    if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
                }
                return inside;
            }
            function inPolygon(x, y, rings) {
                if (!inRing(x, y, rings[0])) return false;
                for (var k = 1; k < rings.length; k++) if (inRing(x, y, rings[k])) return false;
                return true;
            }
            function holds(geometry, x, y) {
                if (geometry.type === "Polygon") return inPolygon(x, y, geometry.coordinates);
                if (geometry.type === "MultiPolygon") return geometry.coordinates.some(function (p) { return inPolygon(x, y, p); });
                return false;
            }
            map.on("click", function (e) {
                var found = [];
                outlines.clearLayers();
                layers.forEach(function (layer) {
                    layer.eachLayer(function (zone) {
                        if (zone.getBounds().contains(e.latlng) && holds(zone.feature.geometry, e.latlng.lng, e.latlng.lat)) {
                            found.push(zone.feature.properties);
                            outlines.addLayer(L.geoJSON(zone.feature, {interactive: false, style: {color: "#000000", weight: 3, fill: false}}));
                        }
                    });
                });
                if (!found.length) return;
                var regional = found.filter(function (p) { return p.network.indexOf("ZPass") !== 0; });
                var title = regional.length > 1 ? "In " + regional.length + " regional networks" : "Zones here";
                var html = "<b>" + title + "</b>" + found.map(function (p) {
                    return "<br><b>" + p.label + "</b><br><small>" + p.towns + "</small>";
                }).join("");
                L.popup({maxWidth: 450}).setLatLng(e.latlng).setContent(html).openOn(map);
            });
            map.on("popupclose", function () { outlines.clearLayers(); });
        })();
        {% endmacro %}
    """)

    def __init__(self, m, layers):
        super().__init__()
        self._name = "ClickZones"
        self.map = m.get_name()
        self.layers = ", ".join(layer.get_name() for layer in layers)
