import os
import shapely.geometry as sgeo
import osmium
from shapely.geometry import Polygon, MultiPolygon

"""
This file contains functions that merge multiple osm files, and cut to the borders
"""


def merge_using_pyosmium(context, osm_files, border, output_path):
      
    area = border["geometry"].iloc[0]
    # Read identifiers that are relevant
    tracker = osmium.IdTracker()

    for osm_file in osm_files:
        processor = osmium.FileProcessor(osm_file).with_filter(
            osmium.filter.KeyFilter("highway", "railway")).with_locations().with_filter(
            osmium.filter.GeoInterfaceFilter())
        
        for item in context.progress(processor, label = "Reading {} ...".format(osm_file.split("/")[-1])):
            geometry = sgeo.shape(item.__geo_interface__["geometry"])

            if area.intersects(geometry):
                tracker.add_way(item.id) # add the way itself
                tracker.add_references(item) # add the referenced nodes
    
    # Read all files again in parallel and write out the relevant items
    processors = [
        osmium.FileProcessor(osm_file).with_filter(tracker.id_filter()).with_locations()
        for osm_file in osm_files
    ]
        
    with osmium.SimpleWriter(output_path) as writer:
        for items in context.progress(osmium.zip_processors(*processors), label = "Writing ..."):
            for item_index, item in enumerate(items):
                if item:
                    writer.add(item)
                    break # already written, skip duplicate
    
    return output_path


def save_as_osmosis_poly(gdf, path, name="boundary"):
    """
    Save a GeoDataFrame (single-row, Polygon or MultiPolygon) as an Osmosis .poly file.
    
    Parameters:
        gdf (GeoDataFrame): The GeoDataFrame to convert (should have one row).
        path (str): Path to save the .poly file.
        name (str): Name to use as the polygon name.
    """
    
    geom = gdf.iloc[0].geometry

    with open(path, 'w') as f:
        f.write(f"{name}\n")
        
        if isinstance(geom, Polygon):
            rings = [geom.exterior] + list(geom.interiors)
        elif isinstance(geom, MultiPolygon):
            rings = []
            for poly in geom.geoms:
                rings.append(poly.exterior)
                rings.extend(poly.interiors)
        else:
            raise ValueError("Geometry must be Polygon or MultiPolygon")
        
        for i, ring in enumerate(rings):
            f.write(f"Area{i}\n")
            for x, y in ring.coords:
                f.write(f" {x} {y}\n")
            f.write("END\n")
        
        f.write("END\n")

def merge_files(context, osm_files, border, output_file):
   
    new_file_path = output_file.replace(".osm.gz","-pyosmium.osm")
    return merge_using_pyosmium(context, osm_files, border, new_file_path)
        