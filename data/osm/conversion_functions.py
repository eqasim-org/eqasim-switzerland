import os
import sys
import logging
import osmium
logger = logging.getLogger(__name__)

"""
This file contains functions that convert .pbf file into .osm file. It uses osmosis, but if osmosis is not installed, it uses osmium.
"""

def convert_pbf_to_osm_pyosmium(input_file, output_file): 
    # import it here, because doesn't need to be installed in the environment if osmosis is installed
      
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

def from_pbf_to_osm_gz(context, osm_file, output_file):
    # Change format from .pbf format to osm.gz
    filename = os.path.basename(osm_file)

    # if it is .pbf format, we convert the file
    if filename.endswith('.pbf'):          
        new_file_name = output_file.replace(".osm.gz","-pyosmium.osm")
        convert_pbf_to_osm_pyosmium(osm_file, new_file_name)        
    else:
        # else, the original file stays unchanged, and is used in next stages (not creating a copy)
        new_file_name = osm_file

    return new_file_name
