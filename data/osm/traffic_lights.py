
import osmium
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd
import logging
logger = logging.getLogger(__name__)


def configure(context):
    context.stage("data.spatial.swiss_border")

    context.config("data_path")
    context.config("osm_path", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000) 
    context.config("add_trafic_lights", False) 


# This handler processes nodes in the OSM data to find traffic lights
class NodesHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.traffic_lights = []            

    def node(self, n):
        if n.tags.get('highway') == 'traffic_signals':
            lon, lat = n.location.lon, n.location.lat
            direction = n.tags.get("traffic_signals:direction")
            self.traffic_lights.append({'node_id': n.id,'x': lon, 'y': lat,'direction': direction, 'geometry': Point(lon, lat)})

CAR_ROAD_TAGS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary','unclassified', 'residential', 'motorway_link', 
                 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link', 'living_street'}
# This handler processes ways in the OSM data to find traffic lights that belong to car roads
class WaysHandler(osmium.SimpleHandler):
    def __init__(self, traffic_light_nodes):
        super().__init__()
        self.traffic_light_node_ids = set(traffic_light_nodes)
        self.belong_to_car_road = []

    def way(self, w):
        if w.tags.get("highway") not in CAR_ROAD_TAGS:
            return

        matching_node_ids = [n.ref for n in w.nodes if n.ref in self.traffic_light_node_ids]
        if matching_node_ids:
            self.belong_to_car_road.extend(matching_node_ids)


def get_region(context):
    # Bounding Area
    border = context.stage("data.spatial.swiss_border")
    border = border.reset_index()[["geometry"]].to_crs(epsg=2056) 
    
    buffer = context.config("border_offset") 
    if buffer>0:
        border["geometry"] = border.geometry.buffer(buffer)
        border["geometry"] = border.geometry.simplify(min(buffer, 3000), preserve_topology=True) #Simplify: Faster
           
    return border["geometry"].iloc[0]


def execute(context):
    # If not requested, do not proces traffic lights
    if not context.config("add_trafic_lights"):
        logger.info("Traffic lights not added, skipping.")
        return
    
    logger.info("Processing traffic lights...")
    # If osm_path is not a list, convert it to a list (generalize for single file or multiple files)
    osm_files = context.config("osm_path")
    if not isinstance(osm_files,list):
        osm_files = [osm_files]
    
    # Ensure all osm files are correctly formatted with the data path
    osm_files = ['%s/osm/%s' % (context.config("data_path"), i) for i in osm_files]
    
    # Process each osm file to add traffic lights
    df = []
    for file in osm_files:
        handler = NodesHandler()
        handler.apply_file(file)
        
        ways_handler = WaysHandler([n['node_id'] for n in handler.traffic_lights])
        ways_handler.apply_file(file)
        belong_to_car_road = ways_handler.belong_to_car_road
        
        # Filter traffic lights that belong to car roads
        logger.info(f"    Found {len(handler.traffic_lights)} traffic lights in {file}.")
        handler.traffic_lights = [tl for tl in handler.traffic_lights if tl['node_id'] in belong_to_car_road]
        logger.info(f"    Found {len(handler.traffic_lights)} traffic lights that belong to car roads in {file}.")
        
        # If traffic lights are found, append them to df
        if handler.traffic_lights:
            dfi = pd.DataFrame(handler.traffic_lights)
            df.append(dfi)
    
    logger.info(f"  -> Total number of traffic lights is: {sum(len(dfi) for dfi in df)}")

    # Merge all traffic lights into a single GeoDataFrame
    if df:
        df = pd.concat(df, ignore_index=True)
        df["node_id"] = df["node_id"].astype(str)  # Ensure node_id is string for consistency
        df = df.drop_duplicates(subset=["node_id", "x", "y"])
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        df = df.to_crs(epsg=2056)
        
        if len(osm_files)>1:
            # only keep the traffic light located i the interest area
            region = get_region(context)
            df = df[df.geometry.within(region)].reset_index(drop=True)
            logger.info(f"  -> Number of traffic lights within the region of interest is: {len(df)}.")

        # Save the traffic lights to a file
        output_path = "%s/traffic_lights.shp" % context.path()
        df.to_file(output_path)
        
        logger.info(f"    Traffic lights saved to {output_path}.")

    return output_path