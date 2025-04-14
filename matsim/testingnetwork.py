# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 11:26:32 2025

@author: abdel
"""

from readers import read_network
from writers import NetworkWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output10p100/vdf/switzerland_network.xml.gz")

print("Reading the network")
net = read_network(path_to_network)



sel = net.links.link_id.apply(lambda x: "pt" not in x)
df = net.links[sel]
df_rest = net.links[~sel]


def merge_link_chains_directed(df):
    df = df.copy()

    # Step 1: Build directed graph
    print("Converting network to networkx")
    G = nx.DiGraph()
    for idx, row in tqdm(df.iterrows(), total=len(G)):
        G.add_edge(row['from_node'], row['to_node'], idx=idx, link_id=row['link_id'])

    visited_links = set()
    merged_links = []
    attribute_consistency = True
    new_start_node = None
    
    # Step 2: Identify nodes that are NOT degree-2 (start/end points)
    def is_degree2(node):
        return G.in_degree(node) == 1 and G.out_degree(node) == 1
    
    
    print("Searching for nodes to remove")
    node_iterator = iter(G.nodes)
    iteration = 0
    while iteration < len(G):   
        if attribute_consistency:
            node = next(node_iterator)
            iteration+=1
        else:
            node = new_start_node
        
        if (G.out_degree(node) > 1) or (not attribute_consistency):
            # Potential chain start point
            for succ in G.successors(node):
                edge_data = G.get_edge_data(node, succ)
                if not edge_data:
                    continue
                idx = edge_data['idx']
                if idx in visited_links:
                    continue

                # Start building the chain
                current_chain = [idx]
                current_node = succ
                current_attrs = df.loc[idx, ['freespeed', 'capacity', 'permlanes',
                                             'oneway', 'modes']].to_dict()

                while is_degree2(current_node):
                    next_nodes = list(G.successors(current_node))
                    if len(next_nodes) != 1:
                        break

                    next_node = next_nodes[0]
                    edge_data = G.get_edge_data(current_node, next_node)
                    if not edge_data:
                        break

                    next_idx = edge_data['idx']
                    next_row = df.loc[next_idx]

                    # Check attribute consistency
                    if not ((next_row["modes"] == current_attrs["modes"])&
                            (next_row["capacity"] == current_attrs["capacity"])&
                            (abs(next_row["freespeed"]-current_attrs["freespeed"])<1)):   
                        attribute_consistency = False
                        new_start_node = current_node
                        break
                    else:
                       attribute_consistency = True 

                    current_chain.append(next_idx)
                    current_node = next_node

                if len(current_chain) > 1:
                    # Merge the chain
                    chain_rows = df.loc[current_chain]
                    merged_links.append({
                        'from_node': df.loc[current_chain[0], 'from_node'],
                        'to_node': df.loc[current_chain[-1], 'to_node'],
                        'link_id': "_".join(df.loc[current_chain, 'link_id'].tolist()),
                        'length': chain_rows['length'].sum(),
                        **current_attrs
                    })
                    visited_links.update(current_chain)

    # Step 3: Build final DataFrame
    links_to_remove = visited_links
    df_cleaned = df[~df.index.isin(links_to_remove)]
    df_merged = pd.DataFrame(merged_links)
    final_df = pd.concat([df_cleaned, df_merged], ignore_index=True)

    return final_df



df2 = pd.concat([merge_link_chains_directed(df),
                 df_rest], ignore_index=True)
net.links = df2


link_attrs = net.link_attrs.groupby('link_id').apply(lambda x: dict(zip(x['name'], x['value']))).reset_index(name='attributes')
net.links = net.links.merge(link_attrs, on="link_id",how="left")
net.links.loc[net.links["attributes"].isna(), "attributes"] = None

net.save("new_network.xml.gz")







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
