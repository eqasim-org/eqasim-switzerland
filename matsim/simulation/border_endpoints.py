"""Local preparation of directional boundary endpoints for foreign car visitors and transit.

Only prepared copies are edited. Survey crossings and Swiss activities stay fixed;
these terminals replace foreign approaches regardless of origin distance.
"""
import csv
import gzip
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from lxml import etree as ET
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from matsim.readers import read_network

logger = logging.getLogger("synpp")

# These are injection/removal buffers, not extra road bottlenecks. Values are
# unscaled, so QSim still applies the scenario's flow/storage capacity factors.
# Storage has no independent link-XML field: lane-metres provide space for about
# 10,000 cars at MATSim's usual 7.5 m cell size, even for shorter fallback links.
TERMINAL_FLOW_CAPACITY = 1_000_000.0
TERMINAL_STORAGE_LANE_METRES = 75_000.0


def select_visitors(destinations, original_population, generated_population, through_crossings=None):
    """Recover survey IDs from the existing, order-preserving ID renumbering."""
    # generate_cross_border_traffic copies population rows without reordering and
    # replaces their IDs. Check that contract instead of guessing an ID offset.
    old = original_population.reset_index(drop=True)
    new = generated_population.reset_index(drop=True)
    if len(old) != len(new) or not old.mz_person_id.equals(new.mz_person_id):
        raise ValueError("Cross-border population order changed during ID renumbering.")
    for column in ["home_x", "home_y", "age", "sex"]:
        if not old[column].equals(new[column]):
            raise ValueError("Cross-border population alignment changed: " + column)
    if not old.person_id.is_unique or not new.person_id.is_unique:
        raise ValueError("Cross-border person identifiers must be unique.")
    mapping = pd.DataFrame({"cross_border_person_id": old.person_id,
                            "person_id": new.person_id.astype(str)})
    selected = destinations.loc[
        destinations.label.isin(["From-To", "Through"]) & destinations.trip_mode.eq("car"),
        ["cross_border_person_id", "label", "entry_interview_point_id", "exit_interview_point_id"],
    ]
    selected = selected.copy()
    selected["exit_source"] = "visitor_assigned"
    selected["inferred_exit_link"] = None
    transit = selected.label.eq("Through")
    if transit.any():
        if through_crossings is None:
            raise ValueError("Transit boundary endpoints require through-crossing assignments.")
        crossings = through_crossings.set_index("cross_border_person_id")
        ids = selected.loc[transit, "cross_border_person_id"]
        selected.loc[transit, "exit_source"] = ids.map(crossings.exit_source).to_numpy()
        # destinations defaults to the entry's exit facility when no reported
        # exit exists. Never mistake that fallback for an assigned transit exit.
        selected.loc[transit, "exit_interview_point_id"] = ids.map(crossings.exit_facility_id).to_numpy()
        inferred = transit & selected.exit_interview_point_id.isna()
        selected.loc[inferred, "inferred_exit_link"] = selected.loc[
            inferred, "cross_border_person_id"].map(crossings.route_exit_link_id)
    return selected.merge(mapping, on="cross_border_person_id", validate="one_to_one")


def exterior_endpoint(exterior, interior, border):
    """Find a short terminal segment wholly outside CH, including near bends."""
    exterior, interior = np.asarray(exterior, float), np.asarray(interior, float)
    if border.covers(shapely.Point(exterior)):
        raise ValueError("assigned crossing has no exterior endpoint")
    base = math.atan2(*(exterior - interior)[::-1])
    for length in [100.0, 50.0, 20.0, 10.0, 5.0, 1.0]:
        for offset in [0, math.pi / 8, -math.pi / 8, math.pi / 4, -math.pi / 4,
                       math.pi / 2, -math.pi / 2, math.pi]:
            point = exterior + length * np.array([math.cos(base + offset), math.sin(base + offset)])
            if not border.intersects(shapely.LineString([exterior, point])):
                return point, length
    raise ValueError("no wholly exterior terminal segment")


def make_terminal(direction, link, nodes, border):
    """Use separate source/sink links; neither permits through traffic."""
    incoming = direction == "in"
    outside = link.from_node if incoming else link.to_node
    inside = link.to_node if incoming else link.from_node
    a = nodes.loc[outside, ["x", "y"]].to_numpy(float)
    b = nodes.loc[inside, ["x", "y"]].to_numpy(float)
    if not border.covers(shapely.Point(b)):
        raise ValueError("assigned link does not cross into/out of CH")
    point, length = exterior_endpoint(a, b, border)
    for field in ["freespeed", "capacity", "permlanes", "length"]:
        if not math.isfinite(float(link[field])) or float(link[field]) <= 0:
            raise ValueError("nonpositive crossing link " + field)
    identifier = "borderEndpoint:%s:%s" % (direction, link.link_id)
    return dict(link_id=identifier, node_id=identifier + ":node", x=point[0], y=point[1],
                from_node=identifier + ":node" if incoming else outside,
                to_node=outside if incoming else identifier + ":node",
                length=length, freespeed=float(link.freespeed),
                capacity=max(float(link.capacity), TERMINAL_FLOW_CAPACITY),
                permlanes=max(float(link.permlanes), TERMINAL_STORAGE_LANE_METRES / length),
                reference_link=str(link.link_id),
                crossing_time=float(link.length) / float(link.freespeed),
                geometry=shapely.LineString([point, a] if incoming else [a, point]).wkt)


def _seconds(value):
    if value is None or value == "undefined":
        raise ValueError("missing planned border time")
    parts = value.split(":")
    seconds = float(value) if len(parts) == 1 else sum(float(v) * f for v, f in zip(parts, [3600, 60, 1]))
    if not math.isfinite(seconds):
        raise ValueError("nonfinite planned border time")
    return seconds


def patch_person(person, entry, exit, entry_facility, exit_facility, transit=False):
    """Retain the interior itinerary and nominal border time of a single plan."""
    plans = person.findall("plan")
    if len(plans) != 1:
        raise ValueError("expected one prepared visitor plan")
    plan = plans[0]
    acts, legs = plan.findall("activity"), plan.findall("leg")
    expected = (3 if pd.isna(exit_facility) else 4) if transit else 5
    if len(acts) != expected or len(legs) != expected - 1 or any(l.get("mode") != "car" for l in legs):
        raise ValueError("unexpected car transit/visitor activity chain")
    explicit_exit = not transit or pd.notna(exit_facility)
    exit_activity = acts[-2] if explicit_exit else None
    if (acts[1].get("facility") != entry_facility
            or acts[1].get("link") != entry["reference_link"]
            or (explicit_exit and (exit_activity.get("facility") != exit_facility
                or exit_activity.get("link") != exit["reference_link"]))):
        raise ValueError("prepared crossing differs from directional assignment")
    arrival = _seconds(acts[1].get("start_time"))
    approach = entry["length"] / entry["freespeed"] + entry["crossing_time"]
    departure = arrival - approach
    return_departure = _seconds(exit_activity.get("end_time")) if explicit_exit else None
    # Without an explicit exit activity there is no planned exit time to retain.
    # Keep the original final activity time; actual arrival is determined by routing/QSim.
    returned = (return_departure + exit["crossing_time"] + exit["length"] / exit["freespeed"]
                if explicit_exit else _seconds(acts[-1].get("start_time")))
    if departure < 0 or departure < _seconds(acts[0].get("start_time", "0")):
        raise ValueError("planned entry too early for the short approach")
    # Validate everything before modifying the person, so unsupported plans can
    # retain their original treatment and receive an explicit diagnostic reason.
    old_departure = acts[0].get("end_time")
    for activity, terminal in [(acts[0], entry), (acts[-1], exit)]:
        activity.set("facility", terminal["link_id"] + ":facility")
        activity.set("link", terminal["link_id"])
        activity.set("x", str(terminal["x"]))
        activity.set("y", str(terminal["y"]))
        activity.attrib.pop("max_dur", None)
    acts[0].set("end_time", str(departure))
    acts[-1].set("start_time", str(returned))
    # Remove stale exterior routing information; preparation routes these legs
    # on the new network. Border and Swiss activity times remain unchanged.
    for leg, time in [(legs[0], departure), (legs[-1], return_departure)]:
        for route in leg.findall("route"):
            leg.remove(route)
        leg.attrib.pop("trav_time", None)
        if "dep_time" in leg.attrib and time is not None:
            leg.set("dep_time", str(time))
    return dict(original_departure=old_departure, departure_s=departure,
                planned_entry_s=arrival, approach_freeflow_s=approach,
                return_arrival_s=returned)


def _patch_population(path, selected, terminals, errors, border_links):
    """Stream one root child at a time, so national populations stay bounded."""
    selected = selected.set_index("person_id").to_dict("index")
    rows, used, seen = [], set(), set()
    tmp = str(path) + ".boundary.tmp"
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
                    pid = element.get("id")
                    if element.tag == "person" and pid in selected:
                        row = dict(person_id=pid, **selected[pid]);seen.add(pid)
                        entry_id, exit_id = row["entry_interview_point_id"], row["exit_interview_point_id"]
                        keys = [("in", border_links.get(entry_id)), ("out", row["inferred_exit_link"] if pd.isna(exit_id) else border_links.get(exit_id))]
                        try:
                            for key in keys:
                                if key in errors:
                                    raise ValueError(errors[key])
                                if key not in terminals:
                                    raise ValueError("assigned car crossing link missing")
                            entry, exit = [terminals[key] for key in keys]
                            row.update(patch_person(element, entry, exit, entry_id, exit_id, transit=row["label"] == "Through"))
                            acts = element.find("plan").findall("activity")
                            for terminal, activity in [(entry, acts[0]), (exit, acts[-1])]:
                                terminal.setdefault("activity_types", set()).add(activity.get("type"))
                            row.update(status="patched", entry_terminal=entry["link_id"], exit_terminal=exit["link_id"])
                            used.update(keys)
                        except ValueError as error:
                            row.update(status="unchanged", reason=str(error))
                        rows.append(row)
                    writer.write(element)
                    element.clear()
                    while element.getprevious() is not None:
                        del root[0]
    for pid in selected.keys() - seen:
        rows.append(dict(person_id=pid, status="absent_from_prepared_population"))
    os.replace(tmp, path)
    return pd.DataFrame(rows), [terminals[key] for key in sorted(used)]


def _append_network(path, terminals):
    """Keep existing network XML intact and add only terminal nodes and links."""
    tmp = str(path) + ".boundary.tmp"
    nodes_written = links_written = False
    with gzip.open(path, "rb") as source, gzip.open(tmp, "wb") as target:
        for line in source:
            if line.strip() == b"</nodes>":
                for row in terminals:
                    target.write(ET.tostring(ET.Element("node", id=row["node_id"], x=str(row["x"]), y=str(row["y"]))) + b"\n")
                nodes_written = True
            if line.strip() == b"</links>":
                for row in terminals:
                    attrs = {key: str(row[key]) for key in ["length", "freespeed", "capacity", "permlanes"]}
                    attrs.update(id=row["link_id"], modes="car", oneway="1")
                    attrs.update({"from": row["from_node"], "to": row["to_node"]})
                    element = ET.Element("link", attrs)
                    attributes = ET.SubElement(element, "attributes")
                    # Artificial approaches carry no original road's tolls,
                    # signals or turn prohibitions. Existing links are untouched.
                    for name, value in [("borderEndpoint", "true"), ("osm:way:highway", "service"),
                                        ("municipalityType", "outside")]:
                        ET.SubElement(attributes, "attribute", name=name, **{"class": "java.lang.String"}).text = value
                    for name, value in [("penalty", "0.0"), ("speedFactor", "1.0")]:
                        ET.SubElement(attributes, "attribute", name=name, **{"class": "java.lang.Double"}).text = value
                    target.write(ET.tostring(element) + b"\n")
                links_written = True
            target.write(line)
    if not nodes_written or not links_written:
        os.unlink(tmp)
        raise ValueError("Expected MATSim network nodes/links sections on separate lines.")
    os.replace(tmp, path)


def patch_border_endpoints(destinations, original_population, generated_population,
                           border_links, swiss_border, network_path, population_path,
                           geometry_path, context, facilities_path, network_modes=("car",), through_crossings=None):
    """Patch local prepared files and export every candidate's outcome."""
    selected = select_visitors(destinations, original_population, generated_population, through_crossings)
    if selected.empty:
        return 0
    border = swiss_border.unary_union if hasattr(swiss_border, "unary_union") else swiss_border
    shapely.prepare(border)
    network = read_network(network_path, skip_attributes=True, context=context)
    links, nodes = network.links.set_index("link_id", drop=False), network.nodes.set_index("node_id")
    keys = {(direction, border_links.get(facility))
            for direction, column in [("in", "entry_interview_point_id"), ("out", "exit_interview_point_id")]
            for facility in selected[column]}
    keys.update(("out", link) for link in selected.inferred_exit_link.dropna())
    terminals, errors = {}, {}
    for direction, link_id in sorted(keys, key=str):
        try:
            if link_id not in links.index or "car" not in links.loc[link_id, "modes"].split(","):
                raise ValueError("assigned car crossing link missing")
            terminal = make_terminal(direction, links.loc[link_id], nodes, border)
            if terminal["link_id"] in links.index or terminal["node_id"] in nodes.index:
                raise ValueError("Boundary endpoints already present; start from fresh prepared copies.")
            terminals[direction, link_id] = terminal
        except ValueError as error:
            errors[direction, link_id] = str(error)
    # MATSim's default router check removes one-way sources/sinks before raising
    # an error. Validate the real network first so disabling that check for the
    # intentional terminals cannot hide unrelated disconnected road components.
    # Check every configured network mode because the MATSim switch is global.
    if terminals:
        validation = validate_routing_core(network.links, network_modes)
        validation.to_csv(Path(population_path).parent / "border_endpoints_connectivity.csv", index=False)
    del network, links, nodes
    report, used = _patch_population(population_path, selected, terminals, errors, border_links)
    if used:
        _append_network(network_path, used)
        _append_facilities(facilities_path, used)
        if geometry_path is not None:
            # The prepared geometry is already a local copy; append matching
            # directed shapes for the new IDs without rewriting existing curves.
            with open(geometry_path, newline="") as stream:
                fields = next(csv.reader(stream))
            if fields != ["LinkId", "Geometry"]:
                raise ValueError("Unexpected detailed network geometry schema: " + str(fields))
            with open(geometry_path, "a", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows((row["link_id"], row["geometry"]) for row in used)
    report.to_csv(Path(population_path).parent / "border_endpoints_persons.csv", index=False)
    pd.DataFrame([{**row, "activity_types": ";".join(sorted(row["activity_types"]))}
                  for row in used]).to_csv(Path(population_path).parent / "border_endpoints_links.csv", index=False)
    logger.info("Boundary endpoints: %s; %d new terminal links.", report.status.value_counts().to_dict(), len(used))
    return len(used)


def validate_routing_core(links, modes):
    """Check directed reachability before adding intentional terminal links.

    This mirrors the node-connectivity condition of MATSim's default router
    check. Sequence-aware turn restrictions remain the upstream cleaner's job;
    no links, allowed modes or restrictions are changed here.
    """
    rows = []
    for mode in sorted(set(modes)):
        allowed = {value: mode in value.split(",") for value in links.modes.unique()}
        selected = links.loc[links.modes.map(allowed)]
        if selected.empty:
            raise ValueError("No road links for configured network-routing mode " + mode)
        # Factorizing just the used nodes excludes unrelated PT-only/isolated
        # nodes and bounds the memory needed for the sparse connectivity graph.
        codes, nodes = pd.factorize(pd.concat([selected.from_node, selected.to_node], ignore_index=True))
        sources, targets = codes[:len(selected)], codes[len(selected):]
        graph = csr_matrix((np.ones(len(selected), dtype=np.int8), (sources, targets)),
                           shape=(len(nodes), len(nodes)))
        count, labels = connected_components(graph, directed=True, connection="strong")
        if count != 1:
            largest = np.bincount(labels).argmax()
            outside = (labels[sources] != largest) | (labels[targets] != largest)
            raise ValueError(
                "Cannot enable boundary endpoints: the original %s network has %d links "
                "outside its largest connected component (examples: %s). Repair the road "
                "network before disabling MATSim's terminal-incompatible connectivity check."
                % (mode, int(outside.sum()), ", ".join(selected.loc[outside, "link_id"].head(10)))
            )
        rows.append(dict(mode=mode, links=len(selected), nodes=len(nodes), strong_components=count))
        logger.info("Validated original %s routing network: %d links in one directed component.",
                    mode, len(selected))
    return pd.DataFrame(rows)


def _append_facilities(path, terminals):
    """Keep endpoint activities consistent with eqasim's required facilities."""
    tmp = str(path) + ".boundary.tmp"
    written = False
    with gzip.open(path, "rb") as source, gzip.open(tmp, "wb") as target:
        for line in source:
            if line.strip() == b"</facilities>":
                for row in terminals:
                    element = ET.Element("facility", id=row["link_id"] + ":facility",
                                         x=str(row["x"]), y=str(row["y"]), linkId=row["link_id"])
                    for activity_type in sorted(row["activity_types"]):
                        ET.SubElement(element, "activity", type=activity_type)
                    target.write(ET.tostring(element) + b"\n")
                written = True
            target.write(line)
    if not written:
        os.unlink(tmp)
        raise ValueError("Expected MATSim facilities closing tag on a separate line.")
    os.replace(tmp, path)
