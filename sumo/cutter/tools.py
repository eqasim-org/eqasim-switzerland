"""Polygon clipping, boundary splitting and trip-cutting calculations."""

import copy
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import igraph as ig
from lxml import etree
from pyproj import CRS, Transformer
from shapely import from_wkb, union_all
from shapely.affinity import translate
from shapely.geometry import LineString, Point
from shapely.ops import substring, transform

from sumo.runtime import sumo
from sumo.scenario import modes
from sumo.scenario.network.tools import coordinates, lane_allows, position, read_network, shape_text
from sumo.scenario.tools import elements, write_xml
from sumo.scenario.population.tools import xml_stream


def geometry_token(filename):
    if filename is None:
        return []
    source = Path(filename).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    files = sorted(source.parent.glob(source.stem + ".*")) if source.suffix.lower() == ".shp" else [source]
    return [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in files if path.is_file()]


def load_region(filename, layer, location, region_crs=None, network_crs=None):
    frame = gpd.read_file(filename, **({"layer": layer} if layer is not None else {}))
    if frame.crs is None:
        if region_crs is None:
            raise ValueError("Region geometry has no CRS; set sumo_region_crs explicitly")
        frame = frame.set_crs(region_crs)
    polygons = frame.geometry[frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if polygons.empty or polygons.is_empty.any() or not polygons.is_valid.all():
        raise ValueError("Region geometry must contain valid, nonempty polygons")
    projection = location.get("projParameter")
    if projection in {None, "!", "-"}:
        projection = network_crs
    if projection is None:
        raise ValueError("Network has no usable CRS; set sumo_network_crs for its unshifted coordinates")
    transformer = Transformer.from_crs(frame.crs, CRS.from_user_input(projection), always_xy=True)
    geometry = transform(transformer.transform, union_all(polygons.to_numpy()))
    offset = [float(value) for value in location.get("netOffset", "0,0").split(",")]
    return translate(geometry, xoff=offset[0], yoff=offset[1])


def read_region(result):
    return from_wkb(Path(result["geometry"]).read_bytes())


def inside_intervals(line, region):
    """Return directed fractions, preserving every crossing and polygon hole."""
    if region.covers(line):
        return [(0.0, 1.0)]
    if not line.is_simple or line.is_ring:
        intervals, distance = [], 0.0
        for first, second in zip(line.coords, list(line.coords)[1:]):
            segment = LineString([first, second])
            if segment.length:
                intervals.extend(((distance + start * segment.length) / line.length,
                                  (distance + end * segment.length) / line.length)
                                 for start, end in inside_intervals(segment, region))
            distance += segment.length
        return _merge_intervals(intervals)
    intersection = line.intersection(region)
    parts = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
    intervals = []
    for part in parts:
        if part.geom_type != "LineString" or part.length < 1e-9:
            continue
        start = line.project(Point(part.coords[0])) / line.length
        end = line.project(Point(part.coords[-1])) / line.length
        if end < start and tuple(part.coords[-1])[:2] == tuple(line.coords[-1])[:2]:
            end = 1.0
        start, end = sorted((start, end))
        start = 0.0 if start < 1e-9 else start
        end = 1.0 if end > 1 - 1e-9 else end
        if end - start > 1e-10:
            intervals.append((start, end))
    return _merge_intervals(intervals)


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1e-10:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def build_network(context, source, region, keep_outside, components="largest", minimum_length=0.01):
    """Clip/split plain definitions and let native netconvert rebuild the network."""
    cache = Path(context.path()).resolve()
    prefix = cache / "source"
    sumo.run(context, ["--sumo-net-file", source, "--plain-output-prefix", prefix,
                       "--plain-output.lanes", "true", "--offset.disable-normalization", "true"], binary="netconvert")
    node_root = etree.parse(str(prefix) + ".nod.xml").getroot()
    edge_root = etree.parse(str(prefix) + ".edg.xml").getroot()
    connection_root = etree.parse(str(prefix) + ".con.xml").getroot()
    source_data = read_network(source)
    nodes = {element.get("id"): copy.deepcopy(element) for element in node_root.findall("node")}
    edge_ids = {element.get("id") for element in edge_root.findall("edge")}
    fragments, edges, mapping = {}, {}, {}
    for element in edge_root.findall("edge"):
        edge_id = element.get("id")
        if edge_id not in source_data["edges"]:
            continue
        start_node, end_node = nodes[element.get("from")], nodes[element.get("to")]
        shape = element.get("shape")
        line = LineString(coordinates(shape)) if shape else LineString([
            (float(start_node.get("x")), float(start_node.get("y"))),
            (float(end_node.get("x")), float(end_node.get("y")))])
        if line.length <= 1e-9:
            line = source_data["edges"][edge_id]["shape"]
            if line.length <= 1e-9:
                raise ValueError(f"Cannot cut zero-length edge geometry: {edge_id}")
        inside = inside_intervals(line, region)
        intervals = inside
        if keep_outside:
            boundaries = sorted({0.0, 1.0, *(value for pair in inside for value in pair)})
            intervals = list(zip(boundaries, boundaries[1:]))
        entries = []
        for index, (start, end) in enumerate(intervals):
            length = source_data["edges"][edge_id]["length"] * (end - start)
            if not keep_outside and length < minimum_length:
                continue
            new_id = edge_id if (keep_outside and index == 0) or (start == 0 and end == 1) else f"{edge_id}__roi_{index}"
            if new_id in edges or (new_id != edge_id and new_id in edge_ids):
                raise ValueError(f"Boundary split edge ID collision: {new_id}")
            piece = substring(line, start, end, normalized=True)
            attributes = copy.deepcopy(element)
            attributes.set("id", new_id)
            attributes.set("shape", shape_text(piece))
            attributes.set("length", format(length, ".12g"))
            for side, fraction, coordinate in (("from", start, piece.coords[0]), ("to", end, piece.coords[-1])):
                if fraction in {0, 1}:
                    node_id = element.get("from" if fraction == 0 else "to")
                else:
                    digest = hashlib.sha256(f"{edge_id}:{fraction:.12g}".encode()).hexdigest()[:24]
                    node_id = "sumo_boundary_" + digest
                    if node_id not in nodes:
                        nodes[node_id] = etree.Element("node", id=node_id, x=format(coordinate[0], ".12g"),
                            y=format(coordinate[1], ".12g"), type="unregulated", radius="0")
                attributes.set(side, node_id)
            for lane in attributes.findall("lane"):
                if lane.get("shape"):
                    lane_line = LineString(coordinates(lane.get("shape")))
                    lane.set("shape", shape_text(substring(lane_line, start, end, normalized=True)))
            entry = {"id": new_id, "source": edge_id, "start": start, "end": end, "length": length,
                     "inside": any(a - 1e-9 <= (start + end) / 2 <= b + 1e-9 for a, b in inside),
                     "from": attributes.get("from"), "to": attributes.get("to")}
            edges[new_id], fragments[new_id] = attributes, entry
            entries.append(entry)
        mapping[edge_id] = entries
    connections = []
    for element in connection_root.findall("connection"):
        before, after = mapping.get(element.get("from"), []), mapping.get(element.get("to"), [])
        if before and after and before[-1]["end"] == 1 and after[0]["start"] == 0:
            connection = copy.deepcopy(element)
            connection.set("from", before[-1]["id"])
            connection.set("to", after[0]["id"])
            connections.append(connection)
    if keep_outside:
        for pieces in mapping.values():
            for first, second in zip(pieces, pieces[1:]):
                count = int(edges[first["id"]].get("numLanes", "1"))
                for lane in range(count):
                    connections.append(etree.Element("connection", attrib={"from": first["id"], "to": second["id"],
                                        "fromLane": str(lane), "toLane": str(lane)}))
    if not edges:
        raise ValueError("The polygon does not retain any usable network edges")
    component_count, discarded = 1, 0
    if not keep_outside:
        kept, component_count = connected_edges(edges, connections, modes.selected(context.config("sumo_modes")), components)
        discarded = len(edges) - len(kept)
        edges = {key: value for key, value in edges.items() if key in kept}
        connections = [element for element in connections if element.get("from") in kept and element.get("to") in kept]
        mapping = {key: [piece for piece in pieces if piece["id"] in kept] for key, pieces in mapping.items()}
    used_nodes = {element.get(side) for element in edges.values() for side in ("from", "to")}
    location = node_root.find("location")
    node_elements = ([copy.deepcopy(location)] if location is not None else []) + [nodes[key] for key in sorted(used_nodes)]
    write_xml(cache / "clipped.nod.xml", "nodes", node_elements)
    write_xml(cache / "clipped.edg.xml", "edges", edges.values())
    write_xml(cache / "clipped.con.xml", "connections", connections)
    network = cache / ("boundary_split.net.xml.gz" if keep_outside else "region.net.xml.gz")
    arguments = ["--node-files", cache / "clipped.nod.xml", "--edge-files", cache / "clipped.edg.xml",
                 "--connection-files", cache / "clipped.con.xml", "--output-file", network,
                 "--offset.disable-normalization", "true", "--no-turnarounds", "true",
                 "--log", cache / "netconvert.log", "--precision", "8"]
    types = Path(str(prefix) + ".typ.xml")
    if types.is_file():
        arguments += ["--type-files", types]
    tls = Path(str(prefix) + ".tll.xml")
    if tls.is_file():
        root = etree.parse(str(tls)).getroot()
        tls_ids = {element.get("tl") for element in connections if element.get("tl")}
        logic = [copy.deepcopy(element) for element in root if element.tag == "tlLogic" and element.get("id") in tls_ids]
        if logic:
            write_xml(cache / "clipped.tll.xml", "tlLogics", logic)
            arguments += ["--tllogic-files", cache / "clipped.tll.xml"]
    sumo.run(context, arguments, binary="netconvert")
    desired = {(element.get("from"), element.get("to"), element.get("fromLane"), element.get("toLane")) for element in connections}
    actual = read_network(network)
    extra = [connection for connection in actual["connections"] if
             (connection["from"], connection["to"], connection["fromLane"], connection["toLane"]) not in desired]
    if extra:
        deletes = [etree.Element("delete", attrib={key: connection[key] for key in ("from", "to", "fromLane", "toLane")}) for connection in extra]
        write_xml(cache / "delete_connections.xml", "connections", deletes)
        cleaned = cache / "connections_checked.net.xml.gz"
        sumo.run(context, ["-s", network, "--connection-files", cache / "delete_connections.xml", "-o", cleaned,
                           "--offset.disable-normalization", "true"], binary="netconvert")
        cleaned.replace(network)
    final = read_network(network)
    if set(final["edges"]) != set(edges):
        raise ValueError("Native rebuilding unexpectedly changed the retained edge IDs")
    if any((connection["from"], connection["to"], connection["fromLane"], connection["toLane"]) not in desired
           for connection in final["connections"]):
        raise ValueError("Native rebuilding introduced a turn absent from the parent network")
    if not keep_outside and components != "all":
        connected_edges(edges, [etree.Element("connection", attrib=connection) for connection in final["connections"]],
                        modes.selected(context.config("sumo_modes")), "error")
    for pieces in mapping.values():
        for piece in pieces:
            piece["length"] = final["edges"][piece["id"]]["length"]
    mapping_path = cache / "edge_mapping.json"
    mapping_path.write_text(json.dumps(mapping, separators=(",", ":")))
    return {"network": str(network), "mapping": str(mapping_path), "source_network": str(source),
            "edge_count": len(edges), "components_before_filter": component_count, "discarded_edges": discarded,
            "connectivity": "weak graph of permitted turns"}


def connected_edges(edges, connections, selected, policy):
    if policy not in {"largest", "all", "error"}:
        raise ValueError("sumo_cutter_components must be 'largest', 'all', or 'error'")
    classes = modes.vehicle_classes(selected)
    ids = sorted(edges)
    index = {edge: position for position, edge in enumerate(ids)}
    pairs = []
    for element in connections:
        first, second = edges[element.get("from")], edges[element.get("to")]
        lanes1 = {int(lane.get("index")): lane.attrib for lane in first.findall("lane")}
        lanes2 = {int(lane.get("index")): lane.attrib for lane in second.findall("lane")}
        lane1 = lanes1.get(int(element.get("fromLane", "0")), first.attrib)
        lane2 = lanes2.get(int(element.get("toLane", "0")), second.attrib)
        if any(lane_allows(lane1, vehicle_class) and lane_allows(lane2, vehicle_class) for vehicle_class in classes):
            pairs.append((index[element.get("from")], index[element.get("to")]))
    components = ig.Graph(n=len(ids), edges=pairs, directed=True).connected_components(mode="weak")
    if policy == "error" and len(components) > 1:
        raise ValueError(f"Polygon cut has {len(components)} disconnected road components")
    if policy == "all":
        return set(ids), len(components)
    largest = max(components, key=lambda component: (len(component), -min(component)))
    return {ids[position] for position in largest}, len(components)


def remap_trip_endpoints(source, mapping_file, destination):
    mapping = json.loads(Path(mapping_file).read_text())
    with xml_stream(destination) as stream, etree.xmlfile(stream, encoding="utf-8") as writer:
        writer.write_declaration()
        with writer.element("routes"):
            for element in elements(source, ("vType", "trip")):
                element = copy.deepcopy(element)
                if element.tag == "trip":
                    for key, side in (("from", 0), ("to", -1)):
                        edge = element.get(key)
                        if edge in mapping:
                            pieces = mapping[edge]
                            position_key = "departPos" if key == "from" else "arrivalPos"
                            position = element.get(position_key)
                            if position not in {None, "base", "max", "last"}:
                                original_length = sum(piece["length"] for piece in pieces)
                                fraction = _position(position, original_length) / original_length
                                pieces = [piece for piece in pieces if piece["start"] - 1e-9 <= fraction <= piece["end"] + 1e-9]
                                if not pieces:
                                    raise ValueError(f"Cannot map {position_key} for trip {element.get('id')}")
                                chosen = pieces[0]
                                element.set(position_key, str(max(0, (fraction - chosen["start"]) / (chosen["end"] - chosen["start"]) * chosen["length"])))
                            else:
                                chosen = pieces[side]
                            element.set(key, chosen["id"])
                    if element.get("via"):
                        element.set("via", " ".join(piece["id"] for edge in element.get("via").split() for piece in mapping.get(edge, [{"id": edge}])))
                writer.write(element)


def cut_vehicle(vehicle, mapping, source_network, cut_network, transitions=None):
    """Keep every contiguous visit, retaining entry clocks and boundary positions."""
    attributes = vehicle["attributes"]
    original_depart = float(attributes["depart"])
    pairs = transitions if transitions is not None else {(connection["from"], connection["to"]) for connection in cut_network["connections"]}
    visits, active = [], []
    previous_index, previous_end = -2, None
    for index, (edge_id, exit_time) in enumerate(zip(vehicle["edges"], vehicle["exit_times"])):
        length = source_network["edges"][edge_id]["length"]
        lower = _position(attributes.get("departPos", "0"), length) / length if index == 0 else 0
        upper = _position(attributes.get("arrivalPos", "max"), length) / length if index == len(vehicle["edges"]) - 1 else 1
        entered = original_depart if index == 0 else vehicle["exit_times"][index - 1]
        if upper < lower:
            raise ValueError(f"Invalid endpoint positions for vehicle {attributes['id']}")
        for piece in mapping.get(edge_id, []):
            start, end = max(lower, piece["start"]), min(upper, piece["end"])
            if end - start < 1e-10:
                continue
            denominator = max(upper - lower, 1e-12)
            begin_time = entered + (exit_time - entered) * (start - lower) / denominator
            end_time = entered + (exit_time - entered) * (end - lower) / denominator
            chain = active and index == previous_index + 1 and previous_end >= 1 - 1e-9 and start <= 1e-9 and (active[-1]["id"], piece["id"]) in pairs
            if active and not chain:
                visits.append(active)
                active = []
            span = piece["end"] - piece["start"]
            active.append({"id": piece["id"], "begin": begin_time, "end": end_time,
                "new_entry": index > 0 or start > lower + 1e-9,
                "new_exit": index < len(vehicle["edges"]) - 1 or end < upper - 1e-9,
                "depart_pos": (start - piece["start"]) / span * piece["length"],
                "arrival_pos": (end - piece["start"]) / span * piece["length"],
                "distance": (end - start) / span * piece["length"],
                "interpolated": start > lower + 1e-9 or end < upper - 1e-9})
            previous_index, previous_end = index, end
    if active:
        visits.append(active)
    results = []
    for index, visit in enumerate(visits):
        attrs = dict(attributes)
        attrs["depart"] = format(visit[0]["begin"], ".12g")
        attrs["departPos"] = format(max(0, visit[0]["depart_pos"]), ".12g")
        attrs["arrivalPos"] = format(visit[-1]["arrival_pos"], ".12g")
        distance = format(sum(piece["distance"] for piece in visit), ".12g")
        if "routeLength" in attrs:
            attrs["routeLength"] = distance
        if visit[0]["new_entry"]:
            attrs["departLane"] = "best"
            attrs["departSpeed"] = "max"
        if visit[-1]["new_exit"]:
            attrs.pop("arrivalLane", None)
            attrs.pop("arrivalSpeed", None)
        if "arrival" in attrs:
            attrs["arrival"] = format(visit[-1]["end"], ".12g")
        if len(visits) > 1:
            attrs["id"] = f"{attributes['id']}__cut_{index}"
        parameters = {**vehicle["parameters"], "sumo_parent_vehicle_id": attributes["id"],
                      "sumo_cut_visit": str(index), "sumo_original_depart": str(original_depart),
                      "sumo_boundary_timing": "interpolated" if any(piece["interpolated"] for piece in visit) else "edge_exit_times"}
        results.append({"attributes": attrs, "edges": [piece["id"] for piece in visit],
            "exit_times": [piece["end"] for piece in visit], "parameters": parameters,
            "route_attributes": {"cost": format(visit[-1]["end"] - visit[0]["begin"], ".12g"),
                                 "routeLength": distance}})
    return results


_position = position
