import osmium
import pandas as pd
import geopandas as gpd
import logging
from data.osm.clean import get_region
from multiprocessing import get_context

logger = logging.getLogger(__name__)


def configure(context):
    context.stage("data.spatial.swiss_border")

    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000) 
    context.config("add_traffic_lights", True) 
    context.config("cross_border_exclude_shapefiles", default=None)


CAR_ROAD_TAGS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary','unclassified', 'residential', 'motorway_link', 
                 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link', 'living_street'}


class TrafficLightsHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.signal_nodes = {}      # node_id -> (lon, lat, direction)
        self.matched_signal_ids = set()

    def node(self, n):
        if n.tags.get("highway") == "traffic_signals":
            self.signal_nodes[n.id] = (
                n.location.lon,
                n.location.lat,
                n.tags.get("traffic_signals:direction"),
            )

    def way(self, w):
        if w.tags.get("highway") in CAR_ROAD_TAGS:
            for nd in w.nodes:
                if nd.ref in self.signal_nodes:
                    self.matched_signal_ids.add(nd.ref)


def read_file(osm_file):
    logger.info(f"  Finding traffic lights is in progress for {osm_file}...")
    handler = TrafficLightsHandler()
    handler.apply_file(osm_file)

    rows = [
        {"node_id": nid, "x": lon, "y": lat, "direction": direction}
        for nid, (lon, lat, direction) in handler.signal_nodes.items()
        if nid in handler.matched_signal_ids
    ]

    if len(rows) == 0:
        logger.warning(f"    No traffic lights found in {osm_file}. This may indicate an issue with the OSM data or the extraction process.")
        return pd.DataFrame(columns=["node_id", "x", "y", "direction"])

    logger.info(f"    Found {len(rows)} traffic lights that belong to car roads in {osm_file}.")
    return pd.DataFrame(rows, columns=["node_id", "x", "y", "direction"])


def execute(context):
    # If not requested, do not proces traffic lights
    if not context.config("add_traffic_lights"):
        logger.info("Traffic lights not added, skipping.")
        return
    
    logger.info("Processing traffic lights...")
    # If osm_file is not a list, convert it to a list (generalize for single file or multiple files)
    osm_files = context.config("osm_file")
    if not isinstance(osm_files,list):
        osm_files = [osm_files]
    
    # Ensure all osm files are correctly formatted with the data path
    osm_files = ['%s/osm/%s' % (context.config("data_path"), i) for i in osm_files]
    
    # Process each osm file to add traffic lights
    processes = min(len(osm_files), 2)  # Limit to 2 processes to avoid overloading the system
    
    # define the output path
    output_path = "%s/traffic_lights.shp" % context.path()

    mp_ctx = get_context("spawn")
    with mp_ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        # lower overhead than apply_async list
        results = pool.imap_unordered(read_file, osm_files, chunksize=1)
        df = [res for res in context.progress(results, total=len(osm_files),
                                              label=f"Processing OSM files for traffic lights ({processes} parallel processes) ...")]

    logger.info(f"  -> Total number of traffic lights is: {sum(len(dfi) for dfi in df)}")

    # Merge all traffic lights into a single GeoDataFrame
    if df:
        df = pd.concat(df, ignore_index=True)
        df["node_id"] = df["node_id"].astype(str)  # Ensure node_id is string for consistency
        df = df.drop_duplicates(subset=["node_id", "x", "y"])
        df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs='EPSG:4326')
        df = df.to_crs(epsg=2056)
        logger.info(f"  -> Number of unique traffic lights after merging is: {len(df)}.")

        if len(osm_files)>1:
            # only keep the traffic lights located in the interest area
            region = get_region(context)["geometry"].iloc[0]
            minx, miny, maxx, maxy = region.bounds
            df = df.cx[minx:maxx, miny:maxy]
            df = df[df.geometry.within(region)].reset_index(drop=True)
            logger.info(f"  -> Number of traffic lights within the region of interest is: {len(df)}.")

        # Save the traffic lights to a file
        df.to_file(output_path)
        
        logger.info(f"    Traffic lights saved to {output_path}.")

    return output_path