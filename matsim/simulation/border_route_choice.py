"""Conservative route repair within grouped road interview places.

This runs on prepared copies, after directional facility matching and before
foreign terminals/final routing. Survey weights, people and Swiss destinations
remain fixed. The Java helper uses the configured JAR, including car turn bans.
"""
import gzip
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from lxml import etree as ET

from matsim.runtime import eqasim
from matsim.simulation.border_endpoints import select_visitors

logger = logging.getLogger("synpp")
FOREIGN_SNAP_DISTANCE = 1000.0


def repair_grouped_crossings(context, network_path, population_path, facilities_path,
                             geometry_path, border_links, destinations=None):
    """Return a local destination copy for subsequent boundary-endpoint creation."""
    root = Path(context.path()) / "border_route_choice"
    root.mkdir(exist_ok=True)
    points = context.stage("data.cross_border.road_candidates")
    candidates = candidate_groups(points)
    if not candidates:
        return destinations
    border = context.stage("data.spatial.swiss_border").to_crs(2056).union_all()
    people, audit = collect_people(
        population_path, context.stage("data.cross_border.swiss_residents_od"),
        candidates, border_links,
        foreign_records(context, destinations) if destinations is not None else {})
    if not people:
        pd.DataFrame(audit).to_csv(root / "persons.csv", index=False)
        return destinations

    # Evaluate existing routes first; do not redistribute already working trips.

    car_links, crossing_times = network_metadata(network_path, border, root / "inside_nodes.txt")
    baseline_pairs = {pair for person in people.values() for pair in person["pairs"]}
    baseline = route_queries(context, root, "baseline", network_path, baseline_pairs)
    pending = {}
    for pid, person in people.items():
        metrics = journey_metrics(person, person["pairs"], baseline)
        person["baseline"] = metrics
        if metrics["valid"] and not person["same_transit_crossing"]:
            audit.append(audit_row(person, "kept_valid", metrics))
        elif person["category"] == "transit" and person["exit_source"] != "survey":
            audit.append(audit_row(person, "unresolved_unobserved_exit", metrics))
        else:
            pending[pid] = person
    logger.info("Grouped border routes: %d checked, %d require candidate evaluation.", len(people), len(pending))

    # Compare only alternative physical crossings in the same surveyed group.

    choices = {}
    pairs = set()
    for pid, person in pending.items():
        choices[pid] = []
        for candidate in candidates[person["group"]]:
            # A corridor match can fall back to a nearby non-crossing road. Do
            # not choose it merely because its onward route crosses elsewhere:
            # the terminal writer requires a real directed boundary link.
            if (str(candidate["entry_link_id"]) not in crossing_times
                    or str(candidate["exit_link_id"]) not in crossing_times):
                continue
            candidate_pairs = journey_pairs(person, candidate)
            if (person["category"] == "transit"
                    and (candidate["exit_link_id"] == person["pairs"][0][1]
                         or candidate["entry_link_id"] == person["pairs"][0][1])):
                continue
            choices[pid].append((candidate, candidate_pairs))
            pairs.update(candidate_pairs)
    routes = route_queries(context, root, "candidates", network_path, pairs)
    feasible = {}
    for pid, options in choices.items():
        feasible[pid] = [(candidate, journey_metrics(pending[pid], query, routes))
                         for candidate, query in options]
        feasible[pid] = [(candidate, metrics) for candidate, metrics in feasible[pid] if metrics["valid"]]

    # Foreign coordinates are independent of the sampled border projection.
    # Unsupported origins fall back to the Swiss journey for visitors; transit
    # requires both foreign ends as well as an observed exit before changing it.

    locations = {coordinate for pid in pending if feasible[pid]
                 for coordinate in reported_locations(pending[pid])}
    snaps = snap_foreign_locations(locations, geometry_path, car_links, border)
    del car_links
    exterior_pairs = set()
    for pid, options in feasible.items():
        for candidate, _ in options:
            exterior_pairs.update(foreign_pairs(pending[pid], candidate, snaps))
    exterior = route_queries(context, root, "exterior", network_path, exterior_pairs, exterior=True)
    replacements, candidate_audit = {}, []
    for pid, person in pending.items():
        ranked = []
        for candidate, metrics in feasible[pid]:
            total = whole_journey_time(person, candidate, metrics["time_s"], snaps, exterior, crossing_times)
            candidate_audit.append(dict(person_id=pid, point_id=candidate["border_crossing_point_id"],
                                        domestic_time_s=metrics["time_s"], whole_time_s=total))
            ranked.append((candidate, metrics, total))
        supported = [option for option in ranked if option[2] is not None]
        if supported:
            selected = min(supported, key=lambda option: (option[2], option[0]["border_crossing_point_id"]))
            basis = "whole_journey"
        elif ranked and person["category"] != "transit":
            selected = min(ranked, key=lambda option: (option[1]["time_s"], option[0]["border_crossing_point_id"]))
            basis = "domestic_journey"
        else:
            reason = "unresolved_foreign_approach" if ranked else "unresolved_no_valid_candidate"
            audit.append(audit_row(person, reason, person["baseline"]))
            continue
        candidate, metrics, total = selected
        replacements[pid] = dict(person=person, candidate=candidate)
        audit.append(audit_row(person, "repaired", metrics, candidate, basis, total))

    # Change only each repaired person's border activities, never shared original
    # facilities. Local destination overrides keep subsequent terminals aligned.

    destinations = apply_replacements(population_path, facilities_path, replacements,
                                      border_links, destinations)
    pd.DataFrame(audit).to_csv(root / "persons.csv", index=False)
    pd.DataFrame(candidate_audit).to_csv(root / "candidates.csv", index=False)
    pd.DataFrame([dict(x=coord[0], y=coord[1], link_id=link, distance_m=distance)
                  for coord, values in snaps.items() for link, distance in values]).to_csv(
                      root / "foreign_attachments.csv", index=False)
    logger.info("Grouped border route repair: %s", pd.Series([r["status"] for r in audit]).value_counts().to_dict())
    return destinations


def candidate_groups(points):
    """Only explicit groups are ambiguous; a named observed office stays fixed."""
    roads = points.loc[points.label.eq("road") & points.interview_place.str.match(r"^Gruppe\s+[A-Z](?:\s|:)")
                       & points.is_candidate.eq(True) & points.selection_weight.gt(0)].copy()
    roads = roads.dropna(subset=["entry_link_id", "exit_link_id"])
    roads = roads[roads.entry_link_id.ne(roads.exit_link_id)]
    return {group: rows.sort_values("border_crossing_point_id").to_dict("records")
            for group, rows in roads.groupby("interview_place") if len(rows) > 1}


def foreign_records(context, destinations):
    """Map generated IDs explicitly; retain provenance from the survey sample."""
    selected = select_visitors(destinations, context.stage("data.cross_border.population"),
                              context.stage("data.cross_border.generate_cross_border_traffic")[0],
                              context.stage("data.cross_border.through_crossings"))
    columns = ["cross_border_person_id", "interview_place", "interview_point_id"]
    selected = selected.merge(destinations[columns], on="cross_border_person_id", validate="one_to_one")
    sample = context.stage("data.cross_border.sample")
    # destinations rounds coordinates when creating facilities. Keep the original
    # survey positions here, together with their provenance, for exterior routing.
    selected = selected.merge(sample[["cross_border_person_id", "start_location_source", "end_location_source",
                                      "residence_x", "residence_y", "destination_residence_x", "destination_residence_y"]],
                              on="cross_border_person_id", validate="one_to_one")
    return selected.set_index("person_id").to_dict("index")


def root_children(path):
    """Bound memory while reading large MATSim XML files in binary mode."""
    with gzip.open(path, "rb") as stream:
        parser = ET.iterparse(stream, events=["start", "end"], resolve_entities=False, no_network=True)
        _, root = next(parser)
        for event, element in parser:
            if event == "end" and element.getparent() is root:
                yield element
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]


def collect_people(path, swiss, groups, border_links, foreign):
    """Recognise supported survey plans without touching ordinary Swiss plans."""
    swiss = swiss.loc[swiss.trip_mode.eq("car") & swiss.interview_place.isin(groups)]
    swiss = swiss.set_index("cross_border_person_id")[["interview_place", "interview_point_id"]].to_dict("index")
    people, audit = {}, []
    for element in root_children(path):
        if element.tag != "person":
            continue
        pid = element.get("id")
        plans = element.findall("plan")
        if len(plans) != 1:
            # Preparation supplies one plan. Never guess which of multiple plans
            # should determine a single person's crossing assignment.
            if pid in foreign and foreign[pid]["interview_place"] in groups:
                audit.append(dict(person_id=pid, status="unresolved_multiple_plans"))
            continue
        plan = plans[0]
        acts = [act for act in plan.findall("activity") if not act.get("type", "").endswith(" interaction")]
        car_legs = [leg for leg in plan.findall("leg") if leg.get("mode") == "car"]
        row = foreign.get(pid)
        if row is not None and row["interview_place"] in groups:
            row = dict(row)
            category = "visitor" if row["label"] == "From-To" else "transit"
            entry, exit = row["entry_interview_point_id"], row["exit_interview_point_id"]
            expected = 5 if category == "visitor" else (4 if pd.notna(exit) else 3)
            if (len(acts) != expected or len(car_legs) != expected - 1
                    or acts[1].get("facility") != entry
                    or acts[1].get("link") != border_links.get(entry)
                    or (pd.notna(exit) and acts[-2].get("facility") != exit)):
                audit.append(dict(person_id=pid, status="unresolved_plan_shape", category=category))
                continue
            exit_link = border_links.get(exit) if pd.notna(exit) else row["inferred_exit_link"]
            if category == "visitor":
                pairs = [(acts[1].get("link"), acts[2].get("link")), (acts[2].get("link"), exit_link)]
            else:
                pairs = [(acts[1].get("link"), exit_link)]
            same = category == "transit" and (
                border_links.get(str(entry).removesuffix("_entry") + "_exit") == exit_link
                or any(str(candidate["entry_link_id"]) == pairs[0][0]
                       and str(candidate["exit_link_id"]) == exit_link
                       for candidate in groups[row["interview_place"]]))
            row.update(person_id=pid, category=category, group=row["interview_place"], pairs=pairs,
                       current_point=row["interview_point_id"], same_transit_crossing=same,
                       activity_indices=[1, len(acts)-2] if category == "visitor" else [1])
        else:
            border_acts = [(index, act) for index, act in enumerate(acts)
                           if str(act.get("facility", "")).rsplit("_", 1)[0] in swiss]
            if not border_acts:
                continue
            if len(border_acts) != 1 or len(acts) != 2 or len(car_legs) != 1:
                audit.append(dict(person_id=pid, status="unresolved_plan_shape", category="swiss"))
                continue
            index, act = border_acts[0]
            record = swiss[act.get("facility").rsplit("_", 1)[0]]
            direction = "entry" if index == 0 else "exit"
            if not act.get("facility").endswith("_" + direction):
                audit.append(dict(person_id=pid, status="unresolved_direction", category="swiss"))
                continue
            row = dict(person_id=pid, category="swiss", group=record["interview_place"],
                       pairs=[(acts[0].get("link"), acts[1].get("link"))], direction=direction,
                       current_point=record["interview_point_id"], same_transit_crossing=False,
                       activity_indices=[index])
        if any(pd.isna(link) for pair in row["pairs"] for link in pair):
            audit.append(dict(person_id=pid, category=row["category"], status="unresolved_missing_link"))
        else:
            people[pid] = row
    return people, audit


def network_metadata(path, border, inside_path):
    """Classify all nodes, including crossings absent from the reference layer."""
    shapely.prepare(border)
    inside, car_links, times = set(), set(), {}
    ids, xs, ys = [], [], []
    with gzip.open(path, "rb") as stream, open(inside_path, "w") as target:
        for _, element in ET.iterparse(stream, events=["end"], tag=["node", "link"],
                                       resolve_entities=False, no_network=True):
            if element.tag == "node":
                ids.append(element.get("id")); xs.append(float(element.get("x"))); ys.append(float(element.get("y")))
                if len(ids) == 10000:
                    inside.update(i for i, yes in zip(ids, shapely.intersects_xy(border, np.array(xs), np.array(ys))) if yes)
                    ids, xs, ys = [], [], []
            else:
                if ids:
                    inside.update(i for i, yes in zip(ids, shapely.intersects_xy(border, np.array(xs), np.array(ys))) if yes)
                    ids, xs, ys = [], [], []
                if "car" in element.get("modes", "").split(","):
                    link = element.get("id")
                    car_links.add(link)
                    if (element.get("from") in inside) != (element.get("to") in inside):
                        times[link] = float(element.get("length")) / float(element.get("freespeed"))
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
        if ids:
            inside.update(i for i, yes in zip(ids, shapely.intersects_xy(border, np.array(xs), np.array(ys))) if yes)
        target.writelines(node + "\n" for node in sorted(inside))
    return car_links, times


def route_queries(context, root, name, network_path, pairs, exterior=False):
    """Deduplicate OD queries and launch the small helper with the existing JAR."""
    pairs = sorted(pairs)
    if not pairs:
        return {}
    queries = root / (name + "_queries.csv")
    result = root / (name + "_routes.csv.gz")
    pd.DataFrame([(index, *pair) for index, pair in enumerate(pairs)],
                 columns=["query_id", "start_link", "end_link"]).to_csv(queries, index=False)
    if exterior:
        # Exterior times rank foreign approaches approximately; domestic route
        # validity still uses the same turn-aware router as scenario preparation.
        exterior_route_queries(context, network_path, root / "inside_nodes.txt", pairs).to_csv(result, index=False)
    else:
        # JDK source launch uses the pinned MATSim/SpeedyALT classes without a
        # second JAR build. It requires a JDK, as does the Maven stage.
        eqasim.run(context, str(Path(__file__).with_name("BorderRouteQueries.java")),
                   [str(network_path), str(root / "inside_nodes.txt"), str(queries), str(result), "full"])
    records = pd.read_csv(result).set_index("query_id")
    if len(records) != len(pairs) or not records.index.is_unique or set(records.index) != set(range(len(pairs))):
        raise ValueError("Incomplete or duplicate grouped-border routing results")
    return {pair: records.loc[index].to_dict() for index, pair in enumerate(pairs)}


def exterior_route_queries(context, network_path, inside_path, pairs):
    """Approximate foreign approaches with the existing directed Pandana model.

    Crossing links are endpoints only; the graph between them contains foreign
    roads exclusively. This prevents even a two-link shortcut through a shared
    Swiss border node without constructing synthetic nodes or changing files.
    """
    from matsim.readers import read_network
    from data.cross_border.through_crossings import build_routing_graph, route_times

    network = read_network(network_path, skip_attributes=True, only_car_links=True, context=context)
    inside = set(Path(inside_path).read_text().splitlines())
    return exterior_times(network.nodes, network.links, inside, pairs, build_routing_graph, route_times)


def exterior_times(nodes, links, inside, pairs, build_graph, get_times):
    """Keep endpoint traversal times while querying only exterior intermediate roads."""
    links = links.set_index("link_id", drop=False).copy()
    links["from_inside"] = links.from_node.isin(inside)
    links["to_inside"] = links.to_node.isin(inside)
    links["travel_time"] = links.length / links.freespeed
    foreign = nodes.loc[~nodes.node_id.isin(inside), ["node_id", "x", "y"]].reset_index(drop=True)
    index = pd.Series(foreign.index, index=foreign.node_id)
    edges = links.loc[~links.from_inside & ~links.to_inside].copy()
    edges["from_index"] = edges.from_node.map(index).astype("int32")
    edges["to_index"] = edges.to_node.map(index).astype("int32")
    pending, rows = [], []
    for i, (start, end) in enumerate(pairs):
        # Pandana does not check turns or return distance for these time-only
        # queries. Leave those metrics unknown rather than reporting false zeros.
        row = dict(query_id=i, found=False, time_s=0., distance_m=np.nan, entries=0, exits=0,
                   restriction_violations=np.nan, routing_method="pandana_exterior")
        rows.append(row)
        if start not in links.index or end not in links.index:
            continue
        a, b = links.loc[start], links.loc[end]
        endpoint_links = [a] if start == end else [a, b]
        row["entries"] = sum(not bool(link.from_inside) and bool(link.to_inside) for link in endpoint_links)
        row["exits"] = sum(bool(link.from_inside) and not bool(link.to_inside) for link in endpoint_links)
        if start == end:
            row.update(found=True, time_s=float(a.travel_time))
        elif not a.to_inside and not b.from_inside:
            pending.append((i, int(index[a.to_node]), int(index[b.from_node]), float(a.travel_time + b.travel_time)))
    if pending:
        source = np.array([row[1] for row in pending], dtype=np.int32)
        target = np.array([row[2] for row in pending], dtype=np.int32)
        supported = directed_reachable(len(foreign), edges.from_index.to_numpy(), edges.to_index.to_numpy(), source, target)
        logger.info("Foreign approach queries: %d; disconnected intermediate paths rejected: %d.",
                    len(pairs), int((~supported).sum()))
        # Equal endpoint nodes need no intermediate road and no routing query.
        adjacent = source == target
        times = np.full(len(pending), np.inf)
        times[adjacent] = 0.
        route = supported & ~adjacent
        if route.any():
            graph = build_graph(foreign, edges)
            times[route] = get_times(graph, source[route], target[route])
        for (i, _, _, endpoint_time), time in zip(pending, times):
            if np.isfinite(time):
                rows[i].update(found=True, time_s=float(time + endpoint_time))
    return pd.DataFrame(rows)


def directed_reachable(size, sources, targets, query_sources, query_targets):
    """Check the component DAG; different components may still connect one way."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, breadth_first_order

    graph = csr_matrix((np.ones(len(sources), dtype=bool), (sources, targets)), shape=(size, size))
    count, labels = connected_components(graph, directed=True, connection="strong")
    start, end = labels[query_sources], labels[query_targets]
    reachable = start == end
    a, b = labels[sources], labels[targets]
    cross = a != b
    dag = csr_matrix((np.ones(int(cross.sum()), dtype=bool), (a[cross], b[cross])), shape=(count, count))
    for component in np.unique(start[~reachable]):
        visited = np.zeros(count, dtype=bool)
        visited[breadth_first_order(dag, int(component), directed=True, return_predecessors=False)] = True
        mask = start == component
        reachable[mask] = visited[end[mask]]
    return reachable


def journey_pairs(person, candidate):
    entry, exit = str(candidate["entry_link_id"]), str(candidate["exit_link_id"])
    current = person["pairs"]
    if person["category"] == "swiss":
        return [(entry, current[0][1])] if person["direction"] == "entry" else [(current[0][0], exit)]
    if person["category"] == "visitor":
        return [(entry, current[0][1]), (current[1][0], exit)]
    return [(entry, current[0][1])]


def journey_metrics(person, pairs, routes):
    """Check each leg's direction, not just an aggregate crossing count."""
    expected = ([(1, 0), (0, 1)] if person["category"] == "visitor" else
                [(1, 1)] if person["category"] == "transit" else
                [(1, 0)] if person["direction"] == "entry" else [(0, 1)])
    values = [routes[pair] for pair in pairs]
    valid = all(value["found"] and value["restriction_violations"] == 0
                and (value["entries"], value["exits"]) == counts for value, counts in zip(values, expected))
    return dict(valid=valid, time_s=sum(v["time_s"] for v in values),
                crossings=sum(v["entries"] + v["exits"] for v in values))


def reported_locations(person):
    locations = []
    if person["category"] == "swiss":
        return locations
    for prefix, source in [("residence", "start_location_source"), ("destination_residence", "end_location_source")]:
        if prefix.startswith("destination") and person["category"] != "transit":
            continue
        x, y = person.get(prefix + "_x"), person.get(prefix + "_y")
        if person.get(source) == "reported" and pd.notna(x) and pd.notna(y) and math.isfinite(x) and math.isfinite(y):
            locations.append((float(x), float(y)))
    return locations


def snap_foreign_locations(locations, geometry_path, car_links, border):
    """Find actual nearby road geometry, keeping tied opposite directions."""
    coordinates = sorted(coord for coord in locations if not border.covers(shapely.Point(coord)))
    if not coordinates:
        return {}
    points = shapely.points(coordinates)
    hits = {coord: [] for coord in coordinates}
    best = {coord: float("inf") for coord in coordinates}
    for chunk in pd.read_csv(geometry_path, chunksize=50000, dtype={"LinkId": str}):
        chunk = chunk[chunk.LinkId.isin(car_links)].reset_index(drop=True)
        if chunk.empty:
            continue
        tree = shapely.STRtree(shapely.from_wkt(chunk.Geometry.to_numpy()))
        indices, distances = tree.query_nearest(points, max_distance=FOREIGN_SNAP_DISTANCE,
                                                all_matches=True, return_distance=True)
        for i, j, distance in zip(*indices, distances):
            coord = coordinates[i]
            if distance < best[coord] - 1e-5:
                hits[coord] = []
                best[coord] = distance
            if distance <= best[coord] + 1e-5:
                hits[coord].append((str(chunk.LinkId.iloc[j]), float(distance)))
    return {coord: sorted(values) for coord, values in hits.items() if values}


def foreign_pairs(person, candidate, snaps):
    """Use both foreign ends for transit and the same residence for a visitor."""
    locations = reported_locations(person)
    if person["category"] == "swiss" or not locations:
        return []
    # A transit origin cannot be substituted by its destination if the former
    # lacks provenance. Require both endpoints explicitly in that case.
    if person["category"] == "transit" and (person.get("start_location_source") != "reported"
            or person.get("end_location_source") != "reported" or len(locations) != 2):
        return []
    entry = str(candidate["entry_link_id"])
    exit = str(candidate["exit_link_id"]) if person["category"] == "visitor" else person["pairs"][0][1]
    return [(link, entry) for link, _ in snaps.get(locations[0], [])] + [
        (exit, link) for link, _ in snaps.get(locations[-1], [])]


def whole_journey_time(person, candidate, domestic_time, snaps, routes, crossing_times):
    pairs = foreign_pairs(person, candidate, snaps)
    incoming, outgoing = [], []
    entry = str(candidate["entry_link_id"])
    exit = str(candidate["exit_link_id"]) if person["category"] == "visitor" else person["pairs"][0][1]
    for pair in pairs:
        route = routes.get(pair)
        if (not route or not route["found"]
                or (pd.notna(route["restriction_violations"]) and route["restriction_violations"] > 0)):
            continue
        if pair[1] == entry and (route["entries"], route["exits"]) == (1, 0):
            incoming.append(route["time_s"])
        if pair[0] == exit and (route["entries"], route["exits"]) == (0, 1):
            outgoing.append(route["time_s"])
    if not incoming or not outgoing or entry not in crossing_times or exit not in crossing_times:
        return None
    # Link-to-link paths include both endpoints. The border links occur in the
    # domestic and exterior queries, so count each only once in the whole trip.
    return domestic_time + min(incoming) + min(outgoing) - crossing_times[entry] - crossing_times[exit]


def audit_row(person, status, metrics, candidate=None, basis=None, whole_time=None):
    return dict(person_id=person["person_id"], category=person["category"], interview_place=person["group"],
                original_point=person["current_point"],
                selected_point=candidate["border_crossing_point_id"] if candidate else person["current_point"],
                status=status, choice_basis=basis, before_crossings=person["baseline"]["crossings"],
                after_crossings=metrics["crossings"], before_domestic_time_s=person["baseline"]["time_s"],
                after_domestic_time_s=metrics["time_s"], whole_time_s=whole_time,
                exit_source=person.get("exit_source"))


def apply_replacements(population_path, facilities_path, replacements, border_links, destinations):
    """New local facilities avoid moving other users of a shared survey record."""
    if not replacements:
        return destinations
    facilities, overrides, seen = {}, {}, set()
    tmp = str(population_path) + ".route-choice.tmp"
    with gzip.open(population_path, "rb") as source, gzip.open(tmp, "wb") as target:
        parser = ET.iterparse(source, events=["start", "end"], resolve_entities=False, no_network=True)
        _, root = next(parser)
        with ET.xmlfile(target, encoding="utf-8") as writer:
            writer.write_declaration()
            if root.getroottree().docinfo.doctype:
                writer.write_doctype(root.getroottree().docinfo.doctype)
            with writer.element(root.tag, dict(root.attrib)):
                for event, element in parser:
                    if event != "end" or element.getparent() is not root:
                        continue
                    pid = element.get("id")
                    if element.tag == "person" and pid in replacements:
                        row = replacements[pid]; person, candidate = row["person"], row["candidate"]
                        plan = element.find("plan")
                        acts = [act for act in plan.findall("activity") if not act.get("type", "").endswith(" interaction")]
                        directions = ([person["direction"]] if person["category"] == "swiss" else
                                      ["entry", "exit"] if person["category"] == "visitor" else ["entry"])
                        for index, direction in zip(person["activity_indices"], directions):
                            act = acts[index]
                            fid = "borderRouteChoice:" + candidate["border_crossing_point_id"] + "_" + direction
                            link = str(candidate[direction + "_link_id"])
                            point = candidate["geometry"]
                            act.set("facility", fid); act.set("link", link)
                            act.set("x", str(point.x)); act.set("y", str(point.y))
                            facility = facilities.setdefault(fid, dict(link_id=link, x=point.x, y=point.y, types=set()))
                            facility["types"].add(act.get("type"))
                            border_links[fid] = link
                            if person["category"] != "swiss":
                                overrides.setdefault(person["cross_border_person_id"], {})[direction] = (fid, point)
                        # These prepared plans will be routed immediately afterwards.
                        for leg in plan.findall("leg"):
                            for route in leg.findall("route"):
                                leg.remove(route)
                        seen.add(pid)
                    writer.write(element)
                    element.clear()
                    while element.getprevious() is not None:
                        del element.getparent()[0]
    if seen != set(replacements):
        os.unlink(tmp)
        raise ValueError("Not all grouped-border replacements were found in the prepared population")
    append_facilities(facilities_path, facilities)
    os.replace(tmp, population_path)
    if destinations is not None:
        destinations = destinations.copy()
        for direction in ["entry", "exit"]:
            ids = {pid: values[direction][0] for pid, values in overrides.items() if direction in values}
            points = {pid: values[direction][1] for pid, values in overrides.items() if direction in values}
            mask = destinations.cross_border_person_id.isin(ids)
            destinations.loc[mask, direction + "_interview_point_id"] = destinations.loc[mask, "cross_border_person_id"].map(ids)
            destinations.loc[mask, direction + "_interview_geometry_point"] = destinations.loc[mask, "cross_border_person_id"].map(points)
    return destinations


def append_facilities(path, facilities):
    tmp = str(path) + ".route-choice.tmp"
    with gzip.open(path, "rb") as source, gzip.open(tmp, "wb") as target:
        parser = ET.iterparse(source, events=["start", "end"], resolve_entities=False, no_network=True)
        _, root = next(parser)
        with ET.xmlfile(target, encoding="utf-8") as writer:
            writer.write_declaration()
            if root.getroottree().docinfo.doctype:
                writer.write_doctype(root.getroottree().docinfo.doctype)
            with writer.element(root.tag, dict(root.attrib)):
                for event, element in parser:
                    if event != "end" or element.getparent() is not root:
                        continue
                    if element.tag == "facility" and element.get("id") in facilities:
                        raise ValueError("Grouped-border facilities already exist; use fresh prepared copies")
                    writer.write(element)
                    writer.write("\n")
                    element.clear()
                    while element.getprevious() is not None:
                        del element.getparent()[0]
                for fid, row in sorted(facilities.items()):
                    element = ET.Element("facility", id=fid, linkId=row["link_id"], x=str(row["x"]), y=str(row["y"]))
                    for activity_type in sorted(row["types"]):
                        ET.SubElement(element, "activity", type=activity_type)
                    writer.write(element)
                    writer.write("\n")
    os.replace(tmp, path)
