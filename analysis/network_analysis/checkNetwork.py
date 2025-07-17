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

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# from matsim.readers import read_network
# from matsim.writers import NetworkWriter
from matsim import read_network

# path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output0p1last/outputtest-networkCleaner/switzerland_network.xml.gz")
# path_to_detailed_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output0p1last/outputtest-networkCleaner/switzerland_detailed_network.csv")


path_to_network = os.path.abspath("/home/dabdelkader/Euler/ch-zh-synpop/cache10p100_2/matsim.scenario.network.convert_osm__933ace3ee3304fe22930de9612bbf728.cache")
network_file = "converted_network.xml.gz"
path_to_network = os.path.join(path_to_network, network_file)


print("Reading the network")
net = read_network(path_to_network)

# link_attrs = self.link_attrs.groupby('link_id').apply(lambda x: dict(zip(x['name'], x['value']))).reset_index(name='attributes')
# self.links = self.links.merge(link_attrs, on="link_id",how="left")
# self.links.loc[self.links["attributes"].isna(), "attributes"] = None

