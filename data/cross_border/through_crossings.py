"""Preserve surveyed road entries and place through-trip destinations abroad.

A+GQPV road-border interviews observe entry into Switzerland. Reported exits
constrain the route where the crossing reference identifies their location;
otherwise the directed route supplies a provisional exit for diagnostics.
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from sklearn.neighbors import KDTree
from shapely.geometry import LineString, Point

from data.cross_border.network_projection import (
    CANDIDATE_COUNT, MIN_ENTRY_FREESPEED_THRESHOLD, MOTORWAY_CANDIDATE_COUNT,
    MOTORWAY_FREESPEED_THRESHOLD, MOTORWAY_SEARCH_RADIUS, find_ideal_crossing,
)
from data.cross_border.travel_times import estimate_travel_time
from data.cross_border.road_exit_points import crossing_name, read_liechtenstein_border
from matsim.readers import read_network
from matsim.simulation.cross_border_links import build_directional_border_link_table

logger = logging.getLogger("synpp")
UNREACHABLE_TIME = np.iinfo(np.uint32).max / 1000.0
RESULT_COLUMNS = [
    "cross_border_person_id", "crossing_direction", "direction_source", "direction_ambiguous",
    "observed_facility_id", "entry_link_id", "entry_match_type",
    "exit_facility_id", "exit_geometry", "exit_link_id", "exit_source", "exit_mapping_source", "exit_border_scope",
    "survey_exit_code", "survey_exit_name", "destination_point_x", "destination_point_y",
    "entry_to_destination_travel_time", "route_exit_link_id", "route_exit_x", "route_exit_y",
    "route_reentries", "reported_exit_is_final", "reported_exit_equals_entry", "destination_node_id", "destination_proxy_offset",
]


def configure(context):
    context.config("border_offset", default = 50000)
    context.config("data_path")
    context.stage("data.cross_border.sample")
    context.stage("data.cross_border.interview_places")
    context.stage("data.cross_border.road_exit_points")
    context.stage("data.spatial.swiss_border")
    context.stage("matsim.scenario.network.pt")


def execute(context):

    # Work on sampled road through-agents before destination facilities are built.

    population = context.stage("data.cross_border.sample")
    through = population[population["label"].eq("Through") & population["trip_mode"].isin(["car", "car_passenger"])].copy()
    if through.empty:
        return pd.DataFrame(columns = RESULT_COLUMNS)
    if not through["direction_crossing"].eq("entering CH").all():
        raise ValueError("Road through-trips must retain the entry direction of the A+GQPV border survey.")
    points = context.stage("data.cross_border.interview_places").to_crs(2056)
    points = points[points["label"].eq("road")].copy()
    exit_points = context.stage("data.cross_border.road_exit_points")
    border = context.stage("data.spatial.swiss_border").iloc[0]

    # Reviewed offices have distinct exit-only anchors; entry sampling keeps its
    # original reference. Their actual coordinates also feed route construction.

    reviewed = exit_points.loc[exit_points["exit_mapping_source"].eq("reviewed_official_office")]
    points = gpd.GeoDataFrame(pd.concat([points, reviewed], ignore_index = True), geometry = "geometry", crs = 2056)
    points = points.drop_duplicates("border_crossing_point_id")
    needs_liechtenstein = points.get("border_scope", pd.Series(index = points.index, dtype = object)).eq("swiss_liechtenstein").any()
    liechtenstein = read_liechtenstein_border(context.config("data_path")) if needs_liechtenstein else None

    # Match crossings with the same directional links later used in preparation.

    network_path = context.stage("matsim.scenario.network.pt")["network"]
    assignments = build_crossing_link_assignments(points, network_path, border, liechtenstein)
    checked = assignments[assignments["facility_id"].isin(reviewed["border_crossing_point_id"] + "_exit")]
    if (checked["facility_id"].nunique() != reviewed["border_crossing_point_id"].nunique()
            or not (checked["from_inside_ch"] & ~checked["to_inside_ch"]).all()):
        raise ValueError("A reviewed road exit has no outward link across its declared border scope.")
    network = read_network(network_path, only_car_links = True, skip_attributes = True, context = context)
    nodes = network.nodes.reset_index(drop = True)
    node_indices = pd.Series(nodes.index, index = nodes["node_id"])
    links = network.links.copy()
    links["from_index"] = links["from_node"].map(node_indices)
    links["to_index"] = links["to_node"].map(node_indices)
    links["travel_time"] = links["length"] / links["freespeed"]
    links = links.set_index("link_id", drop = False)
    facility_links = assignments.set_index("facility_id")
    through["observed_facility_id"] = through["interview_point_id"] + "_entry"
    through["entry_link_id"] = through["observed_facility_id"].map(facility_links["link_id"])
    through["entry_match_type"] = through["observed_facility_id"].map(facility_links["match_type"])
    if through["entry_link_id"].isna().any():
        raise ValueError("A surveyed road entry has no directional network link.")

    # Deduplicate expanded occupants and weights before routing each distinct OD.

    keys = ["entry_link_id", "destination_residence_x", "destination_residence_y", "survey_exit_code", "survey_exit_name"]
    requests = through[keys].drop_duplicates().reset_index(drop = True)
    requests["request_id"] = requests.index
    prepared_border = shapely.from_wkb(border.wkb)
    shapely.prepare(prepared_border)
    inside = shapely.intersects_xy(prepared_border, nodes["x"].to_numpy(), nodes["y"].to_numpy())
    graph = build_routing_graph(nodes, links)
    routes = route_destinations(requests, points, facility_links, nodes, links, inside, graph, border,
                                context.config("border_offset"), exit_points = exit_points)

    # Broadcast routing results without changing sampled agents, weights or IDs.

    result = through[["cross_border_person_id", "observed_facility_id", "entry_match_type"] + keys].merge(
        requests, on = keys, how = "left", validate = "many_to_one",
    ).merge(routes, on = "request_id", how = "left", validate = "many_to_one")
    result["crossing_direction"] = "entry"
    result["direction_source"] = "survey_design"
    result["direction_ambiguous"] = False
    result["reported_exit_equals_entry"] = result["exit_facility_id"].eq(
        result["observed_facility_id"].str.replace(r"_entry$", "_exit", regex = True),
    )
    result = result[RESULT_COLUMNS]
    logger.info("Road through-trips all retain surveyed entry; exit sources: %s", result["exit_source"].value_counts().to_dict())
    unmapped = result["exit_source"].eq("reported_unmapped_network")
    if unmapped.any():
        logger.warning("%d road through-agents report an exit absent from the named crossing reference. Codes/names are retained; routing supplies a provisional exit.", int(unmapped.sum()))
    if result["route_reentries"].gt(0).any():
        logger.warning("%d road through-routes re-enter Switzerland; inspect through_crossings.csv before interpreting their final exit.", int(result["route_reentries"].gt(0).sum()))
    later_exit = result["reported_exit_is_final"].eq(False)
    if later_exit.any():
        logger.warning("%d road through-routes visit their reported exit and later leave Switzerland again; the reported stop is retained and the later exit is flagged.", int(later_exit.sum()))
    same_crossing = result["reported_exit_equals_entry"]
    if same_crossing.any():
        logger.warning("%d road through-agents explicitly report their entry crossing as their exit too. These survey reports are retained and flagged separately from inferred exits.", int(same_crossing.sum()))
    result.drop(columns = "exit_geometry").to_csv(os.path.join(context.path(), "through_crossings.csv"), index = False)
    assignments.to_csv(os.path.join(context.path(), "directional_links.csv"), index = False)
    return result


def build_crossing_link_assignments(points, network_path, border, liechtenstein = None):
    """Keep the national matcher; use CH+LI only for reviewed LI–Austria exits."""
    points = points.copy()
    if "border_scope" not in points:
        points["border_scope"] = "swiss"
    points["border_scope"] = points["border_scope"].fillna("swiss")
    results = []
    for scope, group in points.groupby("border_scope", sort = False):
        if scope == "swiss_liechtenstein":
            if liechtenstein is None:
                raise ValueError("Liechtenstein exit matching requires its national boundary.")
            boundary = border.union(liechtenstein)
        elif scope == "swiss":
            boundary = border
        else:
            raise ValueError("Unknown road-exit border scope: %s" % scope)
        facilities = group.rename(columns = {"border_crossing_point_id": "point_id"}).copy()
        facilities["entry_interview_point_id"] = facilities["point_id"] + "_entry"
        facilities["exit_interview_point_id"] = facilities["point_id"] + "_exit"
        facilities["entry_interview_geometry_point"] = facilities["geometry"]
        facilities["exit_interview_geometry_point"] = facilities["geometry"]
        facilities["trip_mode"] = "car"
        assignments = build_directional_border_link_table(facilities, None, network_path, boundary)
        # Inside flags describe the declared scope here. Swiss route-exit
        # diagnostics below continue to use the Swiss national polygon alone.
        assignments["border_scope"] = scope
        results.append(assignments)
    return pd.concat(results, ignore_index = True)


def route_times(graph, sources, targets):
    """Pandana uses a finite sentinel for unreachable pairs."""
    times = np.asarray(graph.shortest_path_lengths(list(sources), list(targets), imp_name = "travel_time"), dtype = float)
    return np.where(np.isfinite(times) & (times < UNREACHABLE_TIME), times, np.inf)


def build_routing_graph(nodes, links):
    """Build only the directed travel-time hierarchy used by this stage."""
    import pandana
    # Arrays avoid aligning string link IDs with the weights' integer row index.
    return pandana.Network(nodes["x"], nodes["y"], links["from_index"].to_numpy(dtype = "int32"),
                           links["to_index"].to_numpy(dtype = "int32"),
                           edge_weights = links[["travel_time"]].reset_index(drop = True), twoway = False)


def destination_candidates(requests, points, nodes, links, inside, border, border_offset, exit_points = None):
    """Place distant destinations on the destination side of the network perimeter.

    The original foreign destination controls the direction. A reported exit can
    anchor the line to the perimeter; without it the known entry is the anchor.
    Nodes inside Switzerland are never foreign endpoint candidates.
    """
    outside = np.flatnonzero(~inside)
    if len(outside) == 0:
        raise ValueError("The car network has no foreign nodes for through-trip destinations.")
    all_tree = KDTree(nodes.loc[outside, ["x", "y"]].values)
    fast = links[links["freespeed"].ge(MIN_ENTRY_FREESPEED_THRESHOLD)]
    eligible = np.intersect1d(pd.unique(fast[["from_index", "to_index"]].values.ravel()), outside)
    eligible = eligible if len(eligible) else outside
    fast_tree = KDTree(nodes.loc[eligible, ["x", "y"]].values)
    motorway = links[links["freespeed"].ge(MOTORWAY_FREESPEED_THRESHOLD)]
    motorway_nodes = np.intersect1d(pd.unique(motorway[["from_index", "to_index"]].values.ravel()), outside)
    motorway_tree = KDTree(nodes.loc[motorway_nodes, ["x", "y"]].values) if len(motorway_nodes) else None
    # Production uses audited survey-code mappings. The exact-name fallback
    # keeps this helper usable with an explicitly supplied standalone reference.
    grouped = ({code: group for code, group in exit_points.groupby("survey_exit_code")}
               if exit_points is not None else
               {name: group for name, group in points.groupby(points["interview_place"].map(crossing_name))})
    boundary = border.simplify(50).buffer(border_offset).boundary
    rows = []
    for row in requests.itertuples(index = False):
        end = Point(row.destination_residence_x, row.destination_residence_y)
        entry = links.loc[row.entry_link_id]
        entry_anchor = Point(nodes.loc[int(entry.to_index), ["x", "y"]])
        key = row.survey_exit_code if exit_points is not None else crossing_name(row.survey_exit_name)
        reported = grouped.get(key) if pd.notna(key) else None
        anchor = reported.geometry.iloc[0] if reported is not None else entry_anchor
        far = end.distance(border) > border_offset
        ideal = find_ideal_crossing((end.x, end.y), (anchor.x, anchor.y), boundary) if far else end
        pool, tree = (eligible, fast_tree) if far else (outside, all_tree)
        k = min(CANDIDATE_COUNT, len(pool))
        candidates = pool[tree.query([[ideal.x, ideal.y]], k = k)[1].ravel()]
        if far and motorway_tree is not None:
            distance, positions = motorway_tree.query([[ideal.x, ideal.y]], k = min(MOTORWAY_CANDIDATE_COUNT, len(motorway_nodes)))
            candidates = np.concatenate([candidates, motorway_nodes[positions.ravel()[distance.ravel() <= MOTORWAY_SEARCH_RADIUS]]])
        if far:
            # A straight ray from the entry can miss the road corridor towards a
            # distant city. Also consider nodes nearest the original destination,
            # including motorway access, without restricting the exit country.
            candidates = np.concatenate([candidates, eligible[fast_tree.query([[end.x, end.y]], k = k)[1].ravel()]])
            if motorway_tree is not None:
                positions = motorway_tree.query([[end.x, end.y]], k = min(MOTORWAY_CANDIDATE_COUNT, len(motorway_nodes)))[1]
                candidates = np.concatenate([candidates, motorway_nodes[positions.ravel()]])
        exits = reported["border_crossing_point_id"].tolist() if reported is not None else [None]
        mapping_source = None
        if reported is not None:
            mapping_source = reported["exit_mapping_source"].iloc[0] if exit_points is not None else "exact_name"
        scope = reported["border_scope"].iloc[0] if reported is not None and "border_scope" in reported else "swiss"
        scope = scope if pd.notna(scope) else "swiss"
        for exit_point in exits:
            for node in np.unique(candidates):
                offset = np.hypot(nodes.iloc[int(node)].x - ideal.x, nodes.iloc[int(node)].y - ideal.y)
                onward_distance = np.hypot(nodes.iloc[int(node)].x - end.x, nodes.iloc[int(node)].y - end.y)
                rows.append((row.request_id, int(entry.to_index), int(node), exit_point, offset, onward_distance, mapping_source, scope))
    return pd.DataFrame(rows, columns = ["request_id", "entry_node", "destination_node", "exit_point_id", "proxy_offset", "onward_distance", "exit_mapping_source", "exit_border_scope"])


def route_destinations(requests, points, facility_links, nodes, links, inside, graph, border, border_offset, exit_points = None):
    """Choose reachable foreign proxies and inspect the last outbound crossing.

    Costs are directed free-flow travel times. MATSim still performs final routing
    with its configured travel costs; a missing exit remains an inference.
    """
    candidates = destination_candidates(requests, points, nodes, links, inside, border, border_offset, exit_points = exit_points)
    candidates["exit_facility_id"] = candidates["exit_point_id"].map(lambda value: value + "_exit" if pd.notna(value) else None)
    candidates["exit_link_id"] = candidates["exit_facility_id"].map(facility_links["link_id"])
    if (candidates["exit_point_id"].notna() & candidates["exit_link_id"].isna()).any():
        raise ValueError("A located reported road exit has no directional network link.")
    candidates["travel_time"] = np.inf
    constrained = candidates["exit_link_id"].notna()
    if (~constrained).any():
        free = candidates.loc[~constrained]
        candidates.loc[~constrained, "travel_time"] = route_times(graph, free["entry_node"], free["destination_node"])
    if constrained.any():
        fixed = candidates.loc[constrained]
        exits = links.loc[fixed["exit_link_id"]]
        candidates.loc[constrained, "travel_time"] = (
            route_times(graph, fixed["entry_node"], exits["from_index"])
            + exits["travel_time"].to_numpy()
            + route_times(graph, exits["to_index"], fixed["destination_node"])
        )
    # Account for the remaining displacement to the ideal endpoint so that a
    # faster route cannot win simply by ending farther short of the destination.
    # Reuse the existing 50 km/h endpoint-access scale, without a fitted parameter.
    candidates["selection_cost"] = (candidates["travel_time"]
        + estimate_travel_time(candidates["onward_distance"].to_numpy(), "car")
        + candidates["proxy_offset"] / MIN_ENTRY_FREESPEED_THRESHOLD)
    best = candidates.sort_values(["selection_cost", "destination_node", "exit_facility_id"], kind = "stable").drop_duplicates("request_id")
    failed = ~np.isfinite(best["travel_time"])
    if failed.any():
        raise ValueError("No reachable foreign destination through the surveyed entry/reported exit for %d unique routes: %s" % (
            int(failed.sum()), requests[requests["request_id"].isin(best.loc[failed, "request_id"])].head(3).to_dict("records"),
        ))

    # Only boundary links are needed to identify exits in the selected node paths.

    from_inside = inside[links["from_index"].to_numpy(dtype = int)]
    to_inside = inside[links["to_index"].to_numpy(dtype = int)]
    outbound = links[from_inside & ~to_inside].sort_values("travel_time").drop_duplicates(["from_index", "to_index"])
    outbound_lookup = {(int(link.from_index), int(link.to_index)): link.link_id for link in outbound.itertuples()}
    exit_locations = {}
    point_geometry = points.set_index("border_crossing_point_id")["geometry"]
    requests_by_id = requests.set_index("request_id")
    records = []
    for row in best.itertuples(index = False):
        path = list(graph.shortest_path(int(row.entry_node), int(row.destination_node), imp_name = "travel_time")) if pd.isna(row.exit_link_id) else None
        if path is None:
            exit_link = links.loc[row.exit_link_id]
            before = list(graph.shortest_path(int(row.entry_node), int(exit_link.from_index), imp_name = "travel_time"))
            after = list(graph.shortest_path(int(exit_link.to_index), int(row.destination_node), imp_name = "travel_time"))
            path = before + after
        summary = inspect_route_exit(path, inside, outbound_lookup, nodes)
        if summary["route_exit_link_id"] is None:
            raise ValueError("Through-route %s has no outbound Swiss crossing." % row.request_id)
        exit_id = summary["route_exit_link_id"]
        if exit_id not in exit_locations:
            exit_link = links.loc[exit_id]
            line = LineString(nodes.loc[[int(exit_link.from_index), int(exit_link.to_index)], ["x", "y"]].values)
            cut = line.intersection(border.boundary)
            cuts = [cut] if cut.geom_type == "Point" else [part for part in getattr(cut, "geoms", []) if part.geom_type == "Point"]
            exit_locations[exit_id] = max(cuts, key = line.project) if cuts else line.interpolate(0.5, normalized = True)
        summary["route_exit_x"] = exit_locations[exit_id].x
        summary["route_exit_y"] = exit_locations[exit_id].y
        request = requests_by_id.loc[row.request_id]
        source = "survey" if pd.notna(row.exit_point_id) else "reported_unmapped_network" if pd.notna(request.survey_exit_code) else "network"
        destination = nodes.iloc[int(row.destination_node)]
        records.append(dict(
            request_id = row.request_id, exit_facility_id = row.exit_facility_id,
            exit_geometry = point_geometry.loc[row.exit_point_id] if pd.notna(row.exit_point_id) else None,
            exit_link_id = row.exit_link_id, exit_source = source,
            exit_mapping_source = row.exit_mapping_source,
            exit_border_scope = row.exit_border_scope,
            destination_point_x = destination.x, destination_point_y = destination.y,
            destination_node_id = destination.node_id, entry_to_destination_travel_time = row.travel_time,
            destination_proxy_offset = row.proxy_offset,
            reported_exit_is_final = (summary["route_exit_link_id"] == row.exit_link_id
                if pd.notna(row.exit_link_id) and row.exit_border_scope == "swiss" else None),
            **summary,
        ))
    return pd.DataFrame.from_records(records).sort_values("request_id").reset_index(drop = True)


def inspect_route_exit(path, inside, outbound_lookup, nodes):
    """Use the last exit; retain a count of later re-entries for inspection."""
    transitions = [(int(a), int(b)) for a, b in zip(path[:-1], path[1:]) if inside[int(a)] and not inside[int(b)]]
    reentries = sum(not inside[int(a)] and inside[int(b)] for a, b in zip(path[:-1], path[1:]))
    if not transitions:
        return dict(route_exit_link_id = None, route_exit_x = np.nan, route_exit_y = np.nan, route_reentries = reentries)
    a, b = transitions[-1]
    return dict(route_exit_link_id = outbound_lookup[(a, b)],
                route_exit_x = float((nodes.iloc[a].x + nodes.iloc[b].x) / 2),
                route_exit_y = float((nodes.iloc[a].y + nodes.iloc[b].y) / 2),
                route_reentries = reentries)
