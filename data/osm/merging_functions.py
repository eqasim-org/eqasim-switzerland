import os
import shapely.geometry as sgeo
import osmium
from multiprocessing import Pool
"""
This file contains functions that merge multiple osm files, and cut to the borders
"""


def collect_ids(args):
    osm_file, wkb = args
    area = sgeo.shape(wkb)
    tracker = osmium.IdTracker()

    processor = osmium.FileProcessor(osm_file).with_filter(
        osmium.filter.KeyFilter("highway", "railway")
    ).with_locations().with_filter(osmium.filter.GeoInterfaceFilter())

    for item in processor:
        geometry = sgeo.shape(item.__geo_interface__["geometry"])
        if area.intersects(geometry):
            tracker.add_way(item.id)
            tracker.add_references(item)

    # return way IDs + node IDs
    return (set(tracker.way_ids), set(tracker.node_ids))

def merge_using_pyosmium(context, osm_files, border, output_path):
    area = border["geometry"].iloc[0]
    wkb = area.__geo_interface__

    # ---- PHASE 1: PARALLEL SCAN ----
    with Pool() as pool:
        all_results = list(context.progress(
            pool.imap(collect_ids, [(f, wkb) for f in osm_files]),
            label="Scanning..."
        ))

    # merge ID sets
    way_ids = set().union(*(r[0] for r in all_results))
    node_ids = set().union(*(r[1] for r in all_results))

    tracker = osmium.IdTracker()
    for wid in way_ids:
        tracker.add_way(wid)
    for nid in node_ids:
        tracker.add_node(nid)

    # ---- PHASE 2: SERIAL WRITE ----
    processors = [
        osmium.FileProcessor(f).with_filter(tracker.id_filter()).with_locations()
        for f in osm_files
    ]

    with osmium.SimpleWriter(output_path) as writer:
        for items in context.progress(
            osmium.zip_processors(*processors),
            label="Writing..."
        ):
            for item in items:
                if item:
                    writer.add(item)
                    break
    
    return output_path


def merge_files(context, osm_files, border, output_file):
    new_file_path = output_file.replace(".osm.gz","-pyosmium.osm")
    return merge_using_pyosmium(context, osm_files, border, new_file_path)
        