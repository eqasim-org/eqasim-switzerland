"""The zones of the regional PT subscriptions (Verbundabos) of the synthetic population.

synthesis.population.models.subscriptions gives who has a Verbundabo (pt_subscription 2, or 4 with the half-fare card)
but not which zones it covers. They are found from the commute, as in the Swiss PT explorer (utils.gtfs_router.verbund):
the home and the place of education (else of work) are placed in the fare zones, and the commute of every agent whose
home and destination are in different zones is routed on the timetable to find the zones in between. A commute into the Léman Pass area abroad gets a Léman Pass
subscription when the fare table of tpg (tpg_<date>_Fare-system_Geneva.xlsx) is in subscription_zones_lemanpass_path
(else the lemanpass.csv and lemanpass_places.csv of older versions, if they are there).

The persons are returned with the columns of the subscriptions stage and, for the agents with a Verbundabo,
subscription_operator (the fare network), subscription_zones ("ZVV:110,121"; empty when none is found),
subscription_zones_method, subscription_product (Verbundabo or Léman Pass), lemanpass (its trip type) and
lemanpass_annual_chf. verbund_zones.csv (one row per agent) and od_matrix.csv (by zone OD) are written in the stage
folder, with subscription_zones_map.html: the zones and commutes of a sample of agents, to check the results.
"""
import html
import json
import logging
import os

import folium
import numpy as np
import pandas as pd
from branca.element import Element, MacroElement
from jinja2 import Template

from utils.gtfs_router.csa import Cost, Router
from utils.gtfs_router.feed import build_network, day_of
from utils.gtfs_router.lemanpass import load as load_lemanpass
from utils.gtfs_router.pricing import Pricing
from utils.gtfs_router.tracks import TO_WGS84, build_tracks
from utils.gtfs_router.verbund import ROUTED, VERBUND_CODES, commute_zones, commutes, od_matrix, verbund_zones
from utils.gtfs_router.zones import foreign_stop_zones

logger = logging.getLogger("synpp")

ZONE_COLUMNS = {"operator": "subscription_operator", "verbund_zones": "subscription_zones", "method": "subscription_zones_method",
                "subscription": "subscription_product", "lemanpass": "lemanpass", "lemanpass_annual_chf": "lemanpass_annual_chf"}

# the control map: its background, the colours of the agents by method and of the rides by how they count
ESRI_TILES = {"Esri World Topo": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
              "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"}
ESRI_ATTRIBUTION = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, USGS, OpenStreetMap contributors and the GIS user community"
METHOD_COLOURS = {"own_route": "#1f77b4", "group_route": "#17becf", "same_zone": "#2ca02c", "home_zone": "#bcbd22",
                  "fallback": "#d62728", "none": "#000000"}
RIDE_COLOURS = {"in_network": "#2ca02c", "partly_in_network": "#9467bd", "leaves_network": "#ff7f0e", "enters_network": "#ff7f0e",
                "after_exit": "#7f7f7f", "before_entry": "#7f7f7f"}
MAP_SIMPLIFY_M = 20   # the tolerance of the simplified zone outlines of the map, in metres


def configure(context):
    context.config("data_path")
    context.config("random_seed")
    context.config("threads", volatile = True)
    context.config("gtfs_date", "dayWithMostServices")   # the same default as data.gtfs.cleaned

    # The communes abroad and the rail lines of the tracks, relative to data_path (the same as data.pt_pricing.zones)
    context.config("pt_zones_foreign_communes_file", default = "pt_pricing/communes_france.gpkg")
    context.config("pt_zones_foreign_zones_file", default = "pt_pricing/unireso_france_zones.csv")
    context.config("pt_zones_lines_file", default = "pt_pricing/linie-mit-polygon.geojson")

    # The agents on the control map (subscription_zones_map.html), drawn by case; 0: no map
    context.config("subscription_zones_map_agents", default = 400)
    # The date of the routed commutes (YYYYMMDD); empty: gtfs_date if it is a date, else the first Tuesday of the feed
    context.config("subscription_zones_date", default = "")
    context.config("subscription_zones_departure", default = "07:30")
    # The commutes routed in each zone OD; 0: every agent's own commute. With n > 0, the other agents of the OD get the
    # zones of one of the n journeys (faster, for tests, but 16% of these agents then miss a zone of their own journey)
    context.config("subscription_zones_samples", default = 0)
    # The folder of the fare table of tpg (tpg_<date>_Fare-system_Geneva.xlsx), relative to data_path; without it, no Léman Pass
    context.config("subscription_zones_lemanpass_path", default = "pt_pricing/output")
    # The buses follow the roads of their MATSim routes; false: straight lines between their stops
    context.config("subscription_zones_bus_tracks", default = True)

    context.stage("synthesis.population.models.subscriptions")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("data.gtfs.cleaned")
    context.stage("data.pt_pricing.pt_pricing")
    context.stage("data.pt_pricing.zones")
    context.stage("calibration.pt_routing.pt_routing_parameters")   # the route choice of MATSim
    if context.config("subscription_zones_bus_tracks"):
        context.stage("matsim.scenario.network.pt")


def execute(context):

    # Load the population and its commute destinations.

    persons = context.stage("synthesis.population.models.subscriptions")
    work, education = context.stage("synthesis.population.spatial.primary.locations")[:2]

    # Load the timetable, the fares, the zone polygons (data.pt_pricing.zones) and the tracks of the rides with the zones they cross.

    data_path = context.config("data_path")
    data_file = lambda option: os.path.join(data_path, context.config(option))
    pricing_path = context.path("data.pt_pricing.pt_pricing")
    net = build_network("%s/output" % context.path("data.gtfs.cleaned"), cache_dir = context.path())

    schedule_path, network_path = None, None
    if context.config("subscription_zones_bus_tracks"):
        paths = context.stage("matsim.scenario.network.pt")
        schedule_path, network_path = paths["schedule"], paths["network"]

    zones = context.stage("data.pt_pricing.zones")
    zones_path = "%s/pt_zones.gpkg" % context.path("data.pt_pricing.zones")
    tracks = build_tracks(net, data_file("pt_zones_lines_file"), zones, zones_path, schedule_path, network_path, cache_dir = context.path())
    foreign = [data_file("pt_zones_foreign_communes_file"), data_file("pt_zones_foreign_zones_file")]
    pricing = Pricing(pricing_path, tracks = tracks)
    rows = foreign_stop_zones(net, tracks.zones, *foreign)
    added = pricing.add_stop_zones(zip(rows["stop_id"], rows["network"], rows["zone"]))
    logger.info(f"Zones of the stops abroad: {added} stops of {len(rows)} in the listed communes given the zone of their commune")

    # Route the commutes on the network of one date, weighed as by the PT router of MATSim.

    date = routing_date(context.config("subscription_zones_date") or context.config("gtfs_date"), net)
    logger.info(f"Routing the commutes of the Verbundabos on {date}")
    day = net.on(date)
    router = Router(day, cost = Cost.from_eqasim(context.stage("calibration.pt_routing.pt_routing_parameters")))
    lemanpass = load_lemanpass(data_file("subscription_zones_lemanpass_path"), day, tracks.zones, cache_dir = context.path())
    pricing.lemanpass = lemanpass   # the Léman Pass single tickets in the prices of the journeys

    result = verbund_zones(persons, work, education, tracks.zones, router, day, pricing,
                           departure = context.config("subscription_zones_departure"),
                           samples = context.config("subscription_zones_samples"), seed = context.config("random_seed"),
                           workers = max(1, int(context.config("threads"))), lemanpass = lemanpass)

    # Write the zones of the agents and their summary by zone OD.

    result.to_csv("%s/verbund_zones.csv" % context.path(), index = False)
    od_matrix(result).to_csv("%s/od_matrix.csv" % context.path(), index = False)
    logger.info("Verbundabos by method: " + ", ".join(f"{m} {n}" for m, n in result["method"].value_counts().items()))
    logger.info("Verbundabos by product: " + ", ".join(f"{m} {n}" for m, n in result["subscription"].value_counts().items()))

    # Draw the zones and commutes of a sample of agents, to check the results.

    if context.config("subscription_zones_map_agents") > 0:
        sample = map_sample(result, context.config("subscription_zones_map_agents"), context.config("random_seed"))
        agents = map_agents(commutes(persons[persons["person_id"].isin(sample["person_id"])], work, education), sample, tracks.zones, router,
                            day, pricing, context.config("subscription_zones_departure"), lemanpass)
        write_map("%s/subscription_zones_map.html" % context.path(), agents, result, tracks.zones, date)

    # Return the persons with the zones of their subscription.

    zones = result[["person_id"] + list(ZONE_COLUMNS)].rename(columns = ZONE_COLUMNS)
    persons = persons.merge(zones, on = "person_id", how = "left")
    for column in ("subscription_operator", "subscription_zones", "subscription_zones_method", "subscription_product", "lemanpass"):
        persons[column] = persons[column].fillna("")
    assert persons.loc[persons["pt_subscription"].isin(VERBUND_CODES), "subscription_product"].ne("").all()
    return persons


def routing_date(value, net):
    """The date of the routing (YYYYMMDD): value if it is a date covered by the feed (gtfs_date can also be a rule such as
    dayWithMostServices), else the first Tuesday of the feed."""
    first, last = net.dates
    try:
        date = day_of(value)
        if first <= date <= last:
            return date
    except ValueError:
        pass
    first_day = pd.Timestamp(first)
    return (first_day + pd.Timedelta(days = (1 - first_day.dayofweek) % 7)).strftime("%Y%m%d")


def map_sample(result, n, seed):
    """The agents of the control map: n agents drawn at random (with seed), as evenly as possible across the cases."""
    rng = np.random.default_rng(seed)
    groups = {case: group.index.to_numpy() for case, group in result.groupby("case")}
    per_case = max(1, int(np.ceil(n / max(1, len(groups)))))
    drawn = [rng.choice(index, min(per_case, len(index)), replace = False) for index in groups.values()]
    return result.loc[np.sort(np.concatenate(drawn))] if drawn else result.iloc[0:0]


def map_agents(agents, sample, zones, router, net, pricing, departure, lemanpass):
    """The agents of the control map, as dicts for its script: their home and destination, assigned zones, popup, and
    the legs of their own commute routed again (also for the agents whose zones come from another agent of their OD)."""
    agents = agents.merge(sample, on = ["person_id", "purpose"], suffixes = ("_commute", ""))
    lon, lat = TO_WGS84.transform(agents["home_x"].to_numpy(), agents["home_y"].to_numpy())
    dlon, dlat = TO_WGS84.transform(agents["destination_x"].to_numpy(), agents["destination_y"].to_numpy())
    result = []
    for k, row in enumerate(agents.itertuples(index = False)):
        home = (float(lat[k]), float(lon[k]))
        destination = (float(dlat[k]), float(dlon[k])) if np.isfinite(dlat[k]) else None
        own = commute_zones(home, destination, zones, router, net, pricing, departure, lemanpass = lemanpass,
                            age = getattr(row, "age", None)) if row.case in ROUTED else None
        legs = journey_legs(own["journey"], own["rides"], net, pricing.tracks, home, destination) if own and own["journey"] else []
        assigned = row.verbund_zones.split(":", 1)[1].split(",") if row.verbund_zones else []
        result.append({"home": home, "destination": destination, "colour": METHOD_COLOURS.get(row.method, "#000000"),
                       "zones": [f"{row.operator}:{z}" for z in assigned], "legs": legs, "popup": agent_popup(row, own)})
    return result


def journey_legs(journey, rides, net, tracks, home, destination):
    """The legs of a journey to draw: {coords, kind (walk or ride), role (how the ride counted, see zones_along)}."""
    where = lambda stop, end: [float(net.stop_lat[stop]), float(net.stop_lon[stop])] if stop >= 0 else list(end) if end else None
    legs, k = [], 0
    for leg in journey.legs:
        if leg.kind == "ride":
            role = rides[k]["role"] if k < len(rides) else ""
            k += 1
            legs.append({"kind": "ride", "role": role, "colour": RIDE_COLOURS.get(role, "#000000"), "coords": tracks.path(net, leg)})
        elif leg.kind == "walk":
            ends = [where(leg.from_stop, home), where(leg.to_stop, destination)]
            if None not in ends and ends[0] != ends[1]:
                legs.append({"kind": "walk", "role": "walk", "colour": "#555555", "coords": ends})
    return legs


def agent_popup(row, own):
    """The popup of an agent of the control map: its commute, its zones, and the zones of its own routed journey."""
    text = lambda v: html.escape(str(v)) if isinstance(v, str) and v else "–"
    lines = [f"<b>Agent {row.person_id}</b> ({text(row.purpose)}, age {getattr(row, 'age', '–')})",
             f"Case: <b>{text(row.case)}</b>, method: <b>{text(row.method)}</b>",
             f"Home: {text(row.home_network)} {text(row.home_zone)}; destination: {text(row.destination_network)} {text(row.destination_zone)}",
             f"Product: <b>{text(row.subscription)}</b>" + (f": {text(row.lemanpass)}, {row.lemanpass_annual_chf:.0f} CHF/year" if row.lemanpass else ""),
             f"Assigned zones (red): <b>{text(row.verbund_zones)}</b>"]
    if own is not None:
        same = own["verbund_zones"] == row.verbund_zones
        lines.append(f"Zones of its own journey: <b>{text(own['verbund_zones'])}</b> " + ("(same)" if same else "<b style='color:#d62728'>(different)</b>"))
        lines += [f"&nbsp;· {html.escape(r['role'])}: {html.escape(', '.join(r['zones'])) or '–'}" for r in own["rides"]]
        if own["journey"] is None:
            lines.append("<b style='color:#d62728'>No journey found</b>")
    return "<br>".join(lines)


def write_map(path, agents, result, zones, date):
    """The control map: the zone polygons, the homes of the sampled agents (by method), a click showing the zones, the
    destination and the routed journey of an agent, and a summary of all the agents."""
    zones = zones.assign(geometry = zones.geometry.simplify(MAP_SIMPLIFY_M)).to_crs(4326)
    zones = zones.assign(key = zones["network"] + ":" + zones["zone"], label = zones["network"] + " " + zones["zone"] + ": " + zones["name"])
    x0, y0, x1, y1 = zones.total_bounds
    m = folium.Map(location = [(y0 + y1) / 2, (x0 + x1) / 2], zoom_start = 8, tiles = None, prefer_canvas = True)
    for i, (name, url) in enumerate(ESRI_TILES.items()):
        folium.TileLayer(tiles = url, attr = ESRI_ATTRIBUTION, name = name, max_zoom = 19, show = i == 0).add_to(m)
    folium.LayerControl(collapsed = True).add_to(m)
    m.get_root().html.add_child(Element(summary_panel(result, len(agents), date)))
    ControlMap(m, json.loads(zones[["key", "label", "geometry"]].to_json()), agents).add_to(m)
    m.fit_bounds([[y0, x0], [y1, x1]])
    m.save(path)


def summary_panel(result, shown, date):
    """The summary of all the agents with a Verbundabo, and the legends of the map, as a fixed HTML box."""
    table = lambda title, counts: (f"<b>{html.escape(title)}</b><table>" + "".join(
        f"<tr><td>{html.escape(str(k))}</td><td style='text-align:right'>{v:,}</td><td style='text-align:right'>{v / max(1, len(result)):.1%}</td></tr>"
        for k, v in counts.items()) + "</table>")
    sizes = result["n_zones"].clip(upper = 6).map(lambda n: f"{n}+" if n == 6 else str(n)).value_counts().sort_index()
    methods = "".join(f"<span style='color:{c}'>&#9679;</span> {m} " for m, c in METHOD_COLOURS.items())
    rides = "".join(f"<span style='color:{c}'>&#9644;</span> {r} " for r, c in RIDE_COLOURS.items() if r not in ("enters_network", "before_entry"))
    return f"""<div style="position:fixed; top:10px; left:55px; z-index:1000; max-height:90vh; overflow:auto; width:300px; background:white;
        padding:8px; font:12px/1.35 system-ui, sans-serif; border:1px solid #999; border-radius:4px">
        <b>Zones of the Verbundabos</b>: {len(result):,} agents, commutes routed on {date}; {shown} drawn (click a home)<br>
        <small>Homes by method: {methods}<br>Rides: {rides} <span style='color:#555'>&#9476;</span> walk; destination: black ring</small>
        <details><summary>Summary of all the agents</summary>
        {table("By case", result["case"].value_counts())}{table("By method", result["method"].value_counts())}
        {table("By product", result["subscription"].value_counts())}{table("Zones per subscription", sizes)}
        {table("By network (top 15)", result["operator"].fillna("none").value_counts().head(15))}</details></div>"""


class ControlMap(MacroElement):
    """The zones and agents of the control map, drawn by its script (the zone outlines are indexed to be highlighted)."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this.map }};
            var zones = {{ this.zones }}, agents = {{ this.agents }}, index = {};
            var zoneLayer = L.geoJSON(zones, {style: {color: "#666666", weight: 0.6, fill: false},
                onEachFeature: function (f, l) { index[f.properties.key] = f; l.bindTooltip(f.properties.label, {sticky: true}); }}).addTo(map);
            var agentLayer = L.layerGroup().addTo(map), selection = L.layerGroup().addTo(map), current = null;
            function select(a) {
                current = null;
                selection.clearLayers();
                a.zones.forEach(function (k) {
                    if (index[k]) selection.addLayer(L.geoJSON(index[k], {interactive: false, style: {color: "#d62728", weight: 3, fillColor: "#d62728", fillOpacity: 0.15}}));
                });
                a.legs.forEach(function (leg) {
                    selection.addLayer(L.polyline(leg.coords, {color: leg.colour, weight: leg.kind === "ride" ? 5 : 3,
                        dashArray: leg.kind === "walk" ? "4 6" : null, interactive: false}));
                });
                if (a.destination) selection.addLayer(L.circleMarker(a.destination, {radius: 8, color: "#000000", weight: 3, fill: false, interactive: false}));
                var missing = a.zones.filter(function (k) { return !index[k]; });
                current = L.popup({maxWidth: 480}).setLatLng(a.home)
                    .setContent(a.popup + (missing.length ? "<br><b>Zones without a polygon: " + missing.join(", ") + "</b>" : ""));
                current.openOn(map);
            }
            agents.forEach(function (a) {
                L.circleMarker(a.home, {radius: 6, color: "#ffffff", weight: 1, fillColor: a.colour, fillOpacity: 1, bubblingMouseEvents: false})
                    .on("click", function () { select(a); }).addTo(agentLayer);
            });
            // only the popup of the selected agent clears it (opening another one closes the previous popup first)
            map.on("popupclose", function (e) { if (e.popup === current) { selection.clearLayers(); current = null; } });
            L.control.layers(null, {"Fare zones": zoneLayer, "Agents": agentLayer}, {collapsed: false, position: "bottomright"}).addTo(map);
        })();
        {% endmacro %}
    """)

    def __init__(self, m, zones, agents):
        super().__init__()
        self._name = "ControlMap"
        self.map = m.get_name()
        self.zones = json.dumps(zones)
        self.agents = json.dumps(agents)
