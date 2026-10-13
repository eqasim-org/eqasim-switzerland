import os
import shapely.geometry as sgeo
import osmium
from multiprocessing import Pool
from shapely.prepared import prep
import logging
from collections import defaultdict, deque

from data.osm.utils import is_restriction, is_transport_relation, members_exist, relation_members, write_with_speed_correction

logger = logging.getLogger("synpp")

"""
This file contains functions that merge multiple osm files, and cut to the borders
"""


def collect_ids(args):
    osm_file, wkb_area, member_ways, member_nodes = args

    area = prep(sgeo.shape(wkb_area))
    way_ids = set()
    node_ids = set()

    processor = (
        osmium.FileProcessor(osm_file)
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.NODE | osmium.osm.WAY))
        .with_filter(osmium.filter.IdFilter(member_nodes).enable_for(osmium.osm.NODE))
    )

    for item in processor:
        if item.is_node():
            # Stops/platforms need not belong to any road or railway way.
            if item.location.valid() and area.intersects(
                    sgeo.Point(item.location.lon, item.location.lat)):
                node_ids.add(item.id)
            continue
        if not (item.id in member_ways or any(key in item.tags for key in
                ("highway", "railway", "public_transport", "aeroway"))
                or item.tags.get("route") == "ferry"):
            continue
        # GeoInterfaceFilter previously supplied this interface for selected ways.
        if not all(node.location.valid() for node in item.nodes) or len(item.nodes) < 2:
            continue
        geom = sgeo.LineString([(node.lon, node.lat) for node in item.nodes])

        if area.intersects(geom):
            # store way ID
            way_ids.add(item.id)

            # store node references
            if hasattr(item, "nodes"):
                node_ids.update(node.ref for node in item.nodes)

    return (way_ids, node_ids)


def collect_transport_relations(osm_files):
    """Snapshot relevant relations once, using first-file duplicate precedence.

    Streaming OSM objects cannot be retained after advancing the processor. Only
    member tuples and clipping policy are needed here; the writer copies all
    original tags and metadata directly from the input on its existing pass.
    """
    relations = {}
    roots = set()
    processors = [osmium.FileProcessor(path, entities=osmium.osm.RELATION) for path in osm_files]
    for items in osmium.zip_processors(*processors):
        relation = next(item for item in items if item is not None)
        atomic = is_restriction(relation) or relation.tags.get("type") in {
            "connectivity", "destination_sign", "multipolygon"}
        relations[relation.id] = (relation_members(relation), atomic)
        if is_transport_relation(relation):
            roots.add(relation.id)
    # Infrastructure may be represented by nested, otherwise untagged relations.
    # Traverse in memory so hierarchy depth does not require extra file passes.
    pending = deque(roots)
    while pending:
        for kind, ref, _ in relations[pending.popleft()][0]:
            if kind == "r" and ref in relations and ref not in roots:
                roots.add(ref)
                pending.append(ref)
    return {rid: relations[rid] for rid in roots}


def clipped_relation_members(relations, way_ids, node_ids):
    """Keep locally anchored relations and their parents, without dangling refs.

    Route/stop/site collections may be clipped. Rules and polygon rings are
    indivisible: dropping a member would change their meaning. Parent traversal
    handles route masters and stop-area groups regardless of ID ordering, while
    relation-only cycles without any retained node/way are discarded.
    """
    eligible = {rid: members for rid, (members, atomic) in relations.items()
                if not atomic or members_exist(members, way_ids, node_ids)}
    parents = defaultdict(list)
    retained = set()
    for rid, members in eligible.items():
        for kind, ref, _ in members:
            if kind == "r":
                parents[ref].append(rid)
            elif ref in (way_ids if kind == "w" else node_ids):
                retained.add(rid)
    pending = deque(retained)
    while pending:
        for parent in parents[pending.popleft()]:
            if parent not in retained:
                retained.add(parent)
                pending.append(parent)
    ids = {"w": way_ids, "n": node_ids, "r": retained}
    return {rid: [(kind, ref, role) for kind, ref, role in eligible[rid]
                  if ref in ids.get(kind, ())] for rid in retained}


def merge_using_pyosmium(context, osm_files, border, output_path, speed_corrections=None):
    # Geometry that defines the region of interest
    area = border["geometry"].iloc[0]
    wkb_area = area.__geo_interface__ # serializable geometry for multiprocessing

    logger.info("Scanning OSM transport relations and their dependencies ...")
    relations = collect_transport_relations(osm_files)
    member_ways, member_nodes = set(), set()
    for members, _ in relations.values():
        # Retain local platforms and other untagged members, not entire routes
        # outside the region. Every rule is still checked for completeness below.
        member_ways.update(ref for kind, ref, _ in members if kind == "w")
        member_nodes.update(ref for kind, ref, _ in members if kind == "n")

    # PHASE 1: PARALLEL SCAN
    logger.info("Starting Phase 1: Scanning OSM files to collect relevant IDs ...")
    processes = min(len(osm_files), os.cpu_count() or 1)
    with Pool(processes=processes) as pool:
        results = list(context.progress(
            pool.imap_unordered(
                collect_ids,
                [(f, wkb_area, member_ways, member_nodes) for f in osm_files]
            ),
            label=f"Scanning OSM files ({processes} parallel processes) ..."
        ))

    # Merge all ID sets from workers
    logger.info("Merging all ID sets from workers ...")
    all_way_ids = set().union(*(r[0] for r in results))
    all_node_ids = set().union(*(r[1] for r in results))

    kept_relations = clipped_relation_members(relations, all_way_ids, all_node_ids)
    all_relation_ids = set(kept_relations)
    logger.info("Keeping %d transport relations (including turn restrictions).", len(all_relation_ids))
    clipped_count = sum(len(members) != len(relations[rid][0]) for rid, members in kept_relations.items())
    logger.info("Clipped %d transport collections to surviving members.", clipped_count)

    # Build tracker for Phase 2
    tracker = osmium.IdTracker()
    for wid in all_way_ids:
        tracker.add_way(wid)
    for nid in all_node_ids:
        tracker.add_node(nid)
    for rid in all_relation_ids:
        tracker.add_relation(rid)

    # PHASE 2: SERIAL WRITE
    logger.info("Starting Phase 2: Writing merged OSM file ...")
    processors = [
        osmium.FileProcessor(osm_file)
        .with_filter(tracker.id_filter())
        for osm_file in osm_files
    ]

    total = len(all_way_ids) + len(all_node_ids) + len(all_relation_ids)
    with osmium.SimpleWriter(output_path) as writer:
        for items in context.progress(
            osmium.zip_processors(*processors),
            label="Writing merged OSM file ...",
            total=total
        ):
            for item in items:
                if item:
                    if item.is_relation():
                        writer.add_relation(osmium.osm.mutable.Relation(
                            item, members=kept_relations[item.id]))
                    else:
                        write_with_speed_correction(writer, item, speed_corrections)
                    break

    return output_path


def merge_files(context, osm_files, border, output_file, speed_corrections=None):
    new_file_path = output_file.replace(".osm.gz","-pyosmium.osm")
    return merge_using_pyosmium(context, osm_files, border, new_file_path, speed_corrections)
