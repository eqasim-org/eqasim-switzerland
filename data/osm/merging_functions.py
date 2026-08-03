import os
import shapely.geometry as sgeo
import osmium
from multiprocessing import Pool
from shapely.prepared import prep
import logging

logger = logging.getLogger("synpp")

"""
This file contains functions that merge multiple osm files, and cut to the borders
"""


def collect_ids(args):
    osm_file, wkb_area = args

    area = prep(sgeo.shape(wkb_area))
    way_ids = set()
    node_ids = set()

    processor = (
        osmium.FileProcessor(osm_file)
        .with_filter(osmium.filter.KeyFilter("highway", "railway"))
        .with_locations()
        .with_filter(osmium.filter.GeoInterfaceFilter())
    )

    for item in processor:
        geom = sgeo.shape(item.__geo_interface__["geometry"])

        if area.intersects(geom):
            # store way ID
            way_ids.add(item.id)

            # store node references
            if hasattr(item, "nodes"):
                node_ids.update(node.ref for node in item.nodes)

    return (way_ids, node_ids)

def merge_using_pyosmium(context, osm_files, border, output_path, speed_corrections=None):
    # Geometry that defines the region of interest
    area = border["geometry"].iloc[0]
    wkb_area = area.__geo_interface__ # to wkb format, for multiprocessing

    # PHASE 1: PARALLEL SCAN
    logger.info("Starting Phase 1: Scanning OSM files to collect relevant IDs ...")
    processes = min(len(osm_files), os.cpu_count() or 1)
    with Pool(processes=processes) as pool:
        results = list(context.progress(
            pool.imap_unordered(
                collect_ids,
                [(f, wkb_area) for f in osm_files]
            ),
            label=f"Scanning OSM files ({processes} parallel processes) ..."
        ))

    # Merge all ID sets from workers
    logger.info("Merging all ID sets from workers ...")
    all_way_ids = set().union(*(r[0] for r in results))
    all_node_ids = set().union(*(r[1] for r in results))

    # Build tracker for Phase 2
    tracker = osmium.IdTracker()
    for wid in all_way_ids:
        tracker.add_way(wid)
    for nid in all_node_ids:
        tracker.add_node(nid)

    # PHASE 2: SERIAL WRITE
    logger.info("Starting Phase 2: Writing merged OSM file ...")
    processors = [
        osmium.FileProcessor(osm_file)
        .with_filter(tracker.id_filter())
        .with_locations()
        for osm_file in osm_files
    ]

    total = len(tracker.way_ids()) + len(tracker.node_ids())
    write_item = _write_item if speed_corrections is None else _write_item_with_speed_corrections
    with osmium.SimpleWriter(output_path) as writer:
        for items in context.progress(
            osmium.zip_processors(*processors),
            label="Writing merged OSM file ...",
            total=total
        ):
            for item in items:
                if item:
                    write_item(writer, item, speed_corrections)
                    break

    return output_path


def _write_item(writer, item, speed_corrections):
    writer.add(item)

def _write_item_with_speed_corrections(writer, item, speed_corrections):
    if item.type_str() == "w":
        new_speed = speed_corrections.get(item.id)
        if new_speed is None:
            writer.add_way(item)
            return
        
        tags = dict(item.tags)
        tags["maxspeed"] = str(new_speed)

        writer.add_way(
            osmium.osm.mutable.Way(
                id=item.id,
                version=item.version,
                visible=item.visible,
                changeset=item.changeset,
                uid=item.uid,
                user=item.user,
                timestamp=item.timestamp,
                nodes=list(item.nodes),
                tags=tags,
            )
        )
    else:
        writer.add(item)
        

def merge_files(context, osm_files, border, output_file, speed_corrections=None):
    new_file_path = output_file.replace(".osm.gz","-pyosmium.osm")
    return merge_using_pyosmium(context, osm_files, border, new_file_path, speed_corrections)
        