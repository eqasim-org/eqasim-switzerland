
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


class NodesHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.traffic_lights = []            

    def node(self, n):
        if n.tags.get('highway') == 'traffic_signals':
            lon, lat = n.location.lon, n.location.lat
            direction = n.tags.get("traffic_signals:direction")
            self.traffic_lights.append({'id': n.id,'x': lon, 'y': lat,'direction': direction, 'geometry': Point(lon, lat)})

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
        
        # Log the number of traffic lights found
        logger.info(f"    Found {len(handler.traffic_lights)} traffic lights in {file}.")
        
        # If traffic lights are found, return them
        if handler.traffic_lights:
            dfi = pd.DataFrame(handler.traffic_lights)
            df.append(dfi)
    
    logger.info(f"    Total number of traffic lights is: {sum(len(dfi) for dfi in df)}")

    # Merge all traffic lights into a single GeoDataFrame
    if df:
        df = pd.concat(df, ignore_index=True)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        df = df.to_crs(epsg=2056)
        
        # only keep the traffic light located i the interest area
        region = get_region(context)
        df = df[df.geometry.within(region)].reset_index(drop=True)
        logger.info(f"    Number of traffic lights within the region of interest is: {len(df)}.")

        # Save the traffic lights to a file
        output_path = "%s/traffic_lights.shp" % context.path()
        df.to_file(output_path)
        
        logger.info(f"    Traffic lights saved to {output_path}.")

    return df if not df.empty else None