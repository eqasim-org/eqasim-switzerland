import osmium
import pandas as pd
import geopandas as gpd
import logging
from data.osm.clean import get_region
from multiprocessing import get_context
import time

logger = logging.getLogger("synpp")


def configure(context):
    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")    
    context.config("include_tolls", True) 
    context.config("only_french_tolls", True)


def execute(context):
    if not context.config("include_tolls"):
        logger.info("Skipping toll processing as per configuration.")
        return set()
    
    logger.info("Processing tolls...")
    # If osm_file is not a list, convert it to a list (generalize for single file or multiple files)
    osm_files = context.config("osm_file")
    if not isinstance(osm_files,list):
        osm_files = [osm_files]
    
    # keep only french network for now
    if context.config("only_french_tolls"):
        logger.info("\t Filtering OSM files to include only French network regions...")
        files_to_keep = []
        for file in osm_files:
            french_network = ["alsace", "franche-comte", "rhone-alpes"]
            if any(region in file.lower() for region in french_network):
                files_to_keep.append(file)
        osm_files = files_to_keep
        logger.info(f"\t Remaining OSM files after filtering: {osm_files}")

    if not len(osm_files):
        return set()
    
    # Ensure all osm files are correctly formatted with the data path
    osm_files = ['%s/osm/%s' % (context.config("data_path"), i) for i in osm_files]
    
    # Process each osm file to find toll way IDs
    processes = min(len(osm_files), 3)  # Limit to 3 processes to avoid overloading the system

    mp_ctx = get_context("spawn")
    with mp_ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        # lower overhead than apply_async list
        results = pool.imap_unordered(find_toll_way_ids, osm_files, chunksize=1)
        df = [res for res in context.progress(results, total=len(osm_files),
                                              label=f"Processing OSM files for toll ways ({processes} parallel processes) ...")]

    logger.info(f"\t -> Total number of toll ways is: {sum(len(dfi) for dfi in df)}")

    # Merge all toll way IDs into one set
    tolls_links = set()
    for dfi in df:
        tolls_links.update(dfi)

    return tolls_links





################### helper functions ####################
def find_toll_way_ids(pbf_path: str) -> list[int]:
    """
    Scan an OSM .pbf (or .osm) file and return the IDs of all ways
    that carry a toll-related tag (e.g. toll=yes, toll:hgv=yes, toll:motorcar=yes),
    excluding ways explicitly tagged toll=no.
    """
    logger.info(f"\t Scanning {pbf_path} for toll ways...")
    toll_way_ids: set[int] = set()

    class TollHandler(osmium.SimpleHandler):
        def way(self, w):
            for tag in w.tags:
                key, val = tag.k, tag.v
                if key == "toll" or key.startswith("toll:"):
                    if val.lower() != "no":
                        toll_way_ids.add(w.id)
                        break
    
    to = time.time()
    TollHandler().apply_file(pbf_path)
    tf = time.time()
    logger.info(f"\t Scanning {pbf_path} completed in {tf - to:.2f} seconds.")
    return list(toll_way_ids)
