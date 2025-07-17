import data.osm.conversion_functions as cf
import data.osm.merging_functions as mf
import logging
logger = logging.getLogger(__name__)

def configure(context):
    context.stage("matsim.runtime.osmosis")
    context.stage("data.spatial.swiss_border")
    
    context.config("data_path")
    context.config("osm_path", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000) 


def execute(context):
    if not isinstance(context.config("osm_path"),list):
        osm_file = '%s/osm/%s' % (context.config("data_path"), context.config("osm_path"))
        return cf.from_pbf_to_osm_gz(context, osm_file)
    else:
        osm_files = ['%s/osm/%s' % (context.config("data_path"), i) for i in context.config("osm_path")]
        
        # Bounding Area
        border = context.stage("data.spatial.swiss_border")
        border = border.reset_index()[["geometry"]].to_crs(epsg=2056) 
        
        buffer = context.config("border_offset") 
        if buffer>0:
            border["geometry"] = border.geometry.buffer(buffer)
            border["geometry"] = border.geometry.simplify(min(buffer, 3000), preserve_topology=True) #Simplify: Faster
        
        # project to WGS84 (to be compatible with osm data)
        border = border.to_crs("EPSG:4326")        

        # Merge and cut to the area
        return mf.merge_files(context, osm_files, border)



 
