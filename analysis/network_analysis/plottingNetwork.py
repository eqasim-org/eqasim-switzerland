#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:55:04 2025

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
import folium
import geopandas as gpd
import pydeck as pdk
import Functions as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matsim.readers import read_network
from matsim.writers import NetworkWriter



# cantons_path = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/cache/data.spatial.cantons__e8aca7c721ee335dd2deb6b68d6fad0b.p")
cantons_path = os.path.abspath("Y:/ch-zh-synpop/cache/data.spatial.cantons__e8aca7c721ee335dd2deb6b68d6fad0b.p")

# path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/cache/matsim.scenario.network.convert_osm__81c8c18b26c9e068338ed6435a9935a0.cache") #old
# path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/cache0p1last/matsim.scenario.network.convert_osm__72cba64235fffe043f04679662e35451.cache") #new
# path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/cache0p1last/matsim.scenario.network.convert_osm__822d61a3432669723daa2327952b3435.cache") #last
path_to_network = os.path.join("Z:\switzerland_data\osm","switzerland-osm2matsim.xml.gz")

# network_file = "converted_network.xml.gz"
# path_to_network = os.path.join(path_to_network, network_file)

crs = 'EPSG:2056'
columns_to_show = ["length","freespeed","permlanes", "modes"]

print("Reading the network")
net = read_network(path_to_network)

net_geo = net.as_geo(crs)
net_geo = net_geo[["link_id","geometry", *columns_to_show]]

net_geo = net_geo.to_crs(epsg=4326)

print("Reading cantons")
cantons = pd.read_pickle(cantons_path)
cantons = cantons.to_crs(epsg=4326)

print("Selecting only zurich")
zurich = cantons[cantons.canton_name=="Zürich"]
net_zurich = gpd.clip(net_geo, zurich).reset_index(drop=True)

print("Plotting the map for zurich")
F.create_map(net_zurich, 
             data_to_show = columns_to_show,
             path_to_save = "osm2network_zurich_dynamic_map.html")


# print("Plotting one link")
# target_link_id = "742785"
# target_link = net_zurich[net_zurich['link_id'] == target_link_id]
# target_geometry = target_link.geometry.iloc[0]

# buffer_region = target_geometry.buffer(0.0001) 

# links_within_buffer = net_zurich[net_zurich.intersects(buffer_region)]


# fig, ax = plt.subplots(figsize=(10, 10))
# links_within_buffer.plot(ax=ax, color='lightgray', linewidth=2, label='Nearby Links')
# target_link.plot(ax=ax, color='red', linewidth=3, alpha=0.5, label='Selected Link')

# ax.legend()
# plt.title("Selected Links and Nearby Links")
# ax.axis('off')
# plt.show()



