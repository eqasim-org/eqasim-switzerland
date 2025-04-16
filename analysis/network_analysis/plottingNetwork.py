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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matsim.readers import read_network
from matsim.writers import NetworkWriter



path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/cache/matsim.scenario.network.convert_osm__81c8c18b26c9e068338ed6435a9935a0.cache")
network_file = "converted_network.xml.gz"
path_to_network = os.path.join(path_to_network, network_file)


print("Reading the network")
net = read_network(path_to_network)
net_geo = net.as_geo('EPSG:2056')


m = folium.Map(location=[net_geo.geometry.centroid.y.mean(), 
                         net_geo.geometry.centroid.x.mean()], 
               zoom_start=12, 
               tiles='OpenStreetMap')

folium.GeoJson(net_geo).add_to(m)

m.save('road_network_map.html')
