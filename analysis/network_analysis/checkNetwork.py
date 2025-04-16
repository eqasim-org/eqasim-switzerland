#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:47:29 2025

@author: dabdelkader
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from matsim.readers import read_network
from matsim.writers import NetworkWriter

# path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output0p1last/outputtest-networkCleaner/switzerland_network.xml.gz")
# path_to_detailed_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output0p1last/outputtest-networkCleaner/switzerland_detailed_network.csv")


path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output0p1last/outputtest-networkCleaner")
network_file = "switzerland_network.xml.gz"
path_to_network = os.path.join(path_to_network, network_file)



print("Reading the network")
net = read_network(path_to_network)

print("There are %d links in the current network." %len(net.links))

sel = net.links.link_id.apply(lambda x: "pt" not in x)
df = net.links[sel]
df_rest = net.links[~sel]

print("There are %d links when removing pt links" %len(df))

# Checking for loops
sel = (df['from_node'] == df['to_node'])
print("There are %d loops in the network." % sel.sum())

# Checking replicated links
num_duplicates = df.duplicated(subset=['length', 'modes', 'from_node', 'to_node', 'capacity']).sum()
print("There are %d link duplicates in the network." % num_duplicates.sum())


# Checking 



# # Import detailed network
# from shapely import wkt
# from shapely.errors import WKTReadingError
# import geopandas as gpd

# detailed_network = pd.read_csv(path_to_detailed_network)

# def safe_wkt_load(wkt_str):
#     try:
#         geom = wkt.loads(wkt_str)
#         if geom.geom_type == "LineString" and len(geom.coords) > 1:
#             return geom
#         else:
#             return None
#     except (WKTReadingError, AttributeError):
#         return None
    
# detailed_network['Geometry'] = detailed_network['Geometry'].apply(safe_wkt_load)
# detailed_network = detailed_network.dropna(subset=['Geometry'])
# detailed_network = gpd.GeoDataFrame(detailed_network, geometry='Geometry', crs='EPSG:2056') 

# import geopandas as gpd
# from shapely.geometry import MultiLineString
# from shapely.ops import unary_union

# # Step 1: Extract the selected links
# selected_link_ids = df2.iloc[77867].link_id.split('_')
# selected_links = net_geo[net_geo['link_id'].isin(selected_link_ids)]

# # Combine the geometries of the selected links
# selected_geometries = selected_links['geometry']
# combined_geometry = unary_union(selected_geometries)  # Merge all geometries into one

# # Step 2: Define a buffer around the selected links
# buffer_distance = 200  # Adjust this value based on your CRS units (e.g., meters)
# buffered_area = combined_geometry.buffer(buffer_distance)

# # Step 3: Find nearby links
# nearby_links = net_geo[net_geo.geometry.intersects(buffered_area)]

# # Step 4: Plot the results
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(10, 10))

# # Plot all nearby links in light gray
# nearby_links.plot(ax=ax, color='lightgray', linewidth=0.5, label='Nearby Links')

# # Plot the selected links in red
# selected_links.plot(ax=ax, color='red', linewidth=2, label='Selected Links')

# # Add legend and title
# ax.legend()
# plt.title("Selected Links and Nearby Links")
# plt.show()
