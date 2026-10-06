import os
import sys
import osmium
import logging

from data.osm.utils import write_with_speed_correction
logger = logging.getLogger(__name__)

"""
This file contains functions that convert .pbf file into .osm file. 
"""
class OSMHandler(osmium.SimpleHandler):
    def __init__(self, writer, speed_corrections=None):
        super().__init__()
        self.writer = writer
        self.speed_corrections = speed_corrections

    def node(self, n):
        self.writer.add_node(n)

    def relation(self, r):
        self.writer.add_relation(r)

    def way(self, way):
        write_with_speed_correction(self.writer, way, self.speed_corrections)


def convert_pbf_to_osm_pyosmium(input_file, output_file, speed_corrections=None):      
    logger.info("using pyosmium to process OSM data")
    if os.path.exists(output_file):
        logger.info("The file: %s already exists. It will be overridden." % output_file)
        os.remove(output_file)

    writer = osmium.SimpleWriter(output_file)
    handler = OSMHandler(writer, speed_corrections)

    try:
        logger.info(f"Processing {input_file} → {output_file} ...")
        handler.apply_file(input_file)
        logger.info(f"Conversion successful: {output_file}")
    except Exception as e:
        logger.info(f"Error during network conversion using pyosmium: {e}")
        sys.exit(1)
    finally:
        writer.close()

def from_pbf_to_osm_gz(context, osm_file, output_file, speed_corrections=None):
    # Change format from .pbf format to osm.gz
    filename = os.path.basename(osm_file)

    # if it is .pbf format, we convert the file
    if filename.endswith('.pbf') or speed_corrections is not None:          
        new_file_name = output_file.replace(".osm.gz","-pyosmium.osm")
        convert_pbf_to_osm_pyosmium(osm_file, new_file_name, speed_corrections)        
    else:
        # else, the original file stays unchanged, and is used in next stages (not creating a copy)
        new_file_name = osm_file

    return new_file_name
