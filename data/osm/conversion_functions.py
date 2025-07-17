import os
import matsim.runtime.osmosis as osmosis
import sys
import logging
logger = logging.getLogger(__name__)


"""
This file contains functions that convert .pbf file into .osm file. It uses osmosis, but if osmosis is not installed, it uses osmium.
"""

def convert_pbf_to_osm_pyosmium(input_file, output_file): 
    import osmium        
    class OSMHandler(osmium.SimpleHandler):
        def __init__(self, writer):
            super(OSMHandler, self).__init__()
            self.writer = writer

        def node(self, n):
            self.writer.add_node(n)

        def way(self, w):
            self.writer.add_way(w)

        def relation(self, r):
            self.writer.add_relation(r)

    logger.info("using pyosmium to convert .pbf data")
    if os.path.exists(output_file):
        logger.info("The file: %s already exists. It will be overridden." % output_file)
        os.remove(output_file)

    writer = osmium.SimpleWriter(output_file)
    handler = OSMHandler(writer)

    try:
        logger.info(f"Processing {input_file} → {output_file} ...")
        handler.apply_file(input_file)
        logger.info(f"Conversion successful: {output_file}")
    except Exception as e:
        logger.info(f"Error during network conversion using pyosmium: {e}")
        sys.exit(1)
    finally:
        writer.close()

def convert_pbf_to_osm_osmosis(context, input_file, output_file):

    logger.info("using osmosis to convert .pbf data")
    if os.path.exists(output_file):
        logger.info("The file: %s already exists. It will be overridden." % output_file)
        os.remove(output_file)
    
    try:
        osmosis.run(context, [
                "--read-pbf", input_file,            
                "--tag-filter", "accept-ways", "highway=*", "railway=*",
                #"--tag-filter","reject-ways","highway=service",
                "completeWays=yes",     
                "--used-node", 
                "--write-xml", "compressionMethod=gzip", output_file
            ])
        
    except Exception as e:
        logger.info(f"Error during network conversion using osmosis: {e}")
        sys.exit(1)

def from_pbf_to_osm_gz(context, osm_file):
    # Change format from .pbf format to osm.gz
    filename = os.path.basename(osm_file)

    if filename.endswith('.pbf'):  
        new_file_name = "%s/%s" % (context.path(),filename.split('.')[0])
        # If osmosis is installed, use it, else, use pyosmium
        if osmosis.is_osmosis_installed(context):
            new_file_name = new_file_name+"-osmosis.osm.gz"
            convert_pbf_to_osm_osmosis(context, osm_file, new_file_name)
        else:
            new_file_name = new_file_name+"-pyosmium.osm"
            convert_pbf_to_osm_pyosmium(osm_file, new_file_name)
        osm_file = new_file_name

    return osm_file
