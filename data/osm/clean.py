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
    # if the path is not a list, treat it as a single file, else treat it as a list of files, merge them, and keep only
    # the network within 'border_offset' distance to the border
    output_file = '%s/osm_network.osm.gz' % context.path()
    osm_file = context.config("osm_path")

    if not isinstance(osm_file,list):
        osm_file = '%s/osm/%s' % (context.config("data_path"), osm_file)
        return cf.from_pbf_to_osm_gz(context, osm_file, output_file)
    else:
        osm_files = ['%s/osm/%s' % (context.config("data_path"), f) for f in osm_file]

        # Bounding Area
        border = context.stage("data.spatial.swiss_border")
        border = border.reset_index()[["geometry"]].to_crs(epsg=2056) 
        
        buffer = context.config("border_offset") 
        if buffer>0:
            border["geometry"] = border.geometry.buffer(buffer)
            border["geometry"] = border.geometry.simplify(min(buffer, 3000), preserve_topology=True) #Simplify: Faster
        
        border = border.to_crs("EPSG:4326") # because osm in in wgs84       
        # Merge and cut to the area
        return mf.merge_files(context, osm_files, border, output_file)



 
