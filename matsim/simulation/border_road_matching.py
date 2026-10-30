"""Follow a reference point's road to the border before weighting its capacity."""

import gzip
import heapq
from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import shapely
from lxml import etree

from matsim.scenario.network.utils.link_geometry import attach_link_geometries, GEOMETRY_COLUMN

# These are geometric safeguards, not traffic-count calibration parameters.
ANCHOR_RADIUS = 100.0
PARALLEL_ROAD_TOLERANCE = 25.0
MAX_CORRIDOR_DISTANCE = 2000.0


def attach_road_details(links, network_path, geometry_path):
    """Keep only border-near shapes/road classes; stream the large input files."""
    links = links.copy()
    nodes = pd.concat([
        links[['from_node', 'x_from', 'y_from']].rename(columns={'from_node': 'node_id', 'x_from': 'x', 'y_from': 'y'}),
        links[['to_node', 'x_to', 'y_to']].rename(columns={'to_node': 'node_id', 'x_to': 'x', 'y_to': 'y'}),
    ]).drop_duplicates('node_id')
    network = SimpleNamespace(nodes=nodes, links=links)
    attach_link_geometries(network, geometry_path)
    links['road_geometry'] = links.pop(GEOMETRY_COLUMN)
    bounds = shapely.bounds(links.road_geometry.to_numpy())
    for i, name in enumerate(['minx', 'miny', 'maxx', 'maxy']):
        links[name] = bounds[:, i]

    # Read only the road class needed for continuity, not every network attribute.
    wanted = set(links.link_id)
    classes = {}
    opener = gzip.open if str(network_path).endswith('.gz') else open
    with opener(network_path, 'rb') as stream:
        for _, element in etree.iterparse(stream, events=['end'], tag='link'):
            link_id = element.get('id')
            if link_id in wanted:
                classes[link_id] = element.findtext("attributes/attribute[@name='osm:way:highway']", '')
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
    links['road_class'] = links.link_id.map(classes).fillna('')
    return links


def match_road_corridor(point, direction, links):
    """Find a directed border link on the nearby anchor's connected road class.

    Directed continuity identifies the road corridor and its crossing. This is
    a reference match, not an agent route or a substitute for turn restrictions.
    Separate nearby carriageways can seed the
    search, but a different road class cannot steal a motorway reference point.
    """
    nearby = links[(links.minx <= point.x + ANCHOR_RADIUS) & (links.maxx >= point.x - ANCHOR_RADIUS)
                  & (links.miny <= point.y + ANCHOR_RADIUS) & (links.maxy >= point.y - ANCHOR_RADIUS)].copy()
    if nearby.empty:
        return None
    nearby['anchor_distance'] = shapely.distance(nearby.road_geometry.to_numpy(), point)
    nearby = nearby.sort_values(['anchor_distance', 'link_id'])
    first = nearby.iloc[0]
    if first.anchor_distance > ANCHOR_RADIUS or not first.road_class:
        return None
    anchors = nearby[(nearby.road_class == first.road_class)
                     & (nearby.anchor_distance <= min(ANCHOR_RADIUS, first.anchor_distance + PARALLEL_ROAD_TOLERANCE))]

    # Restrict to a small local corridor. Length bounds prevent following a road
    # to another distant border crossing when its intended crossing is absent.
    corridor = links[(links.road_class == first.road_class)
                     & (links.minx <= point.x + MAX_CORRIDOR_DISTANCE) & (links.maxx >= point.x - MAX_CORRIDOR_DISTANCE)
                     & (links.miny <= point.y + MAX_CORRIDOR_DISTANCE) & (links.maxy >= point.y - MAX_CORRIDOR_DISTANCE)]
    forward, backward = defaultdict(list), defaultdict(list)
    rows = {row.link_id: row for row in corridor.itertuples(index=False)}
    for row in rows.values():
        forward[row.from_node].append((row.to_node, row.link_id))
        backward[row.to_node].append((row.from_node, row.link_id))
    queue = []
    for anchor in anchors.itertuples(index=False):
        if _crosses_in_direction(anchor, direction):
            return _result(anchor, anchor, anchor.capacity, 0.0)
        # When the dot is already beyond the crossing in travel direction, trace
        # backwards. Otherwise follow the road forwards. This keeps capacities
        # on the correct carriageway even where the two directions differ.
        if anchor.from_inside_ch != anchor.to_inside_ch:
            continue
        reverse = bool(anchor.from_inside_ch) == (direction == 'entry')
        position = anchor.road_geometry.project(point)
        node = anchor.from_node if reverse else anchor.to_node
        distance = position if reverse else anchor.road_geometry.length - position
        heapq.heappush(queue, (distance + anchor.anchor_distance, node, reverse, anchor.link_id, anchor.capacity))
    visited = set()
    while queue:
        distance, node, reverse, anchor_id, capacity = heapq.heappop(queue)
        if distance > MAX_CORRIDOR_DISTANCE:
            break
        if (node, reverse) in visited:
            continue
        visited.add((node, reverse))
        graph = backward if reverse else forward
        for other, link_id in graph[node]:
            link = rows[link_id]
            bottleneck = min(capacity, link.capacity)
            if _crosses_in_direction(link, direction):
                return _result(link, rows[anchor_id], bottleneck, distance)
            if (other, reverse) not in visited:
                heapq.heappush(queue, (distance + link.length, other, reverse, anchor_id, bottleneck))
    return None


def _crosses_in_direction(link, direction):
    return ((not link.from_inside_ch and link.to_inside_ch) if direction == 'entry'
            else (link.from_inside_ch and not link.to_inside_ch))


def _result(link, anchor, capacity, distance):
    result = pd.Series(link._asdict())
    result['anchor_link_id'] = anchor.link_id
    # MATSim capacity already includes lanes. Use the minimum along the short
    # anchor-to-border corridor, avoiding inflated customs-plaza crossing widths.
    result['selection_capacity'] = capacity
    result['corridor_distance'] = distance
    return result
