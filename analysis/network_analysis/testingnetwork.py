# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 11:26:32 2025

@author: abdel
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
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matsim.readers import read_network
from matsim.writers import NetworkWriter


path_to_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output10p100/vdf/switzerland_network.xml.gz")
path_to_detailed_network = os.path.abspath("/home/dabdelkader/Scratch/ch-zh-synpop/output10p100/vdf/switzerland_detailed_network.csv")

print("Reading the network")
net = read_network(path_to_network)


stats = {
    "number_of_nodes":0,
    "number_of_links":0,
    "attributes_change":0,
    "skiped_loop":0,
    "successful_merge":0,
    "degree_is_2":0,  
    "one_in_one_out":0,
    "already_visited":0,        
    }


sel = net.links.link_id.apply(lambda x: "pt" not in x)
df = net.links[sel]
df_rest = net.links[~sel]

# Removing loops
sel = (df['from_node'] == df['to_node'])
print("There are %d loops in the network that are removed." % sel.sum())
df = df[~sel]

# Remove replicated links
len_df = len(df)
df = df.drop_duplicates(subset=['length','modes','from_node', 'to_node', 'capacity'],
                        ignore_index=True)
print("There are %d link duplicates in the network that are removed." % (len_df-len(df)))

# Remove unnecessary nodes
def merge_link_chains_directed(df):
    df = df.copy()
    # Step 1: Build directed graph
    print("Converting network to networkx")
    G = nx.MultiDiGraph()    
    G.add_edges_from(zip(df['from_node'], 
                         df['to_node'], 
                        ({'idx': idx, 'link_id': lid} for idx, lid in zip(df.index, df['link_id']))))
        
    stats['number_of_links'] = G.number_of_edges()
    stats['number_of_nodes'] = G.number_of_nodes()
    
    visited_links = set()
    merged_links = []
    attribute_consistency = True
    new_start_node = None
    
    nodes_to_remove = []
    # Step 2: Identify nodes that are NOT degree-2 (start/end points)
    def is_degree2(node):
        return G.in_degree(node) == 1 and G.out_degree(node) == 1
    
    
    print("Searching for nodes to remove")
    node_iterator = G.__iter__()
    iteration = 0
    progress_bar = tqdm(total=len(G), desc="Finding nodes to remove ")
    
    while iteration < len(G):   
        if attribute_consistency:
            node = next(node_iterator)
            iteration+=1            
            progress_bar.update(1) # Updtae the bar only here (follow the iterator)
            
            stats['degree_is_2']+=int(G.degree(node)==2)
            stats['one_in_one_out']+=int(is_degree2(node))
        else:
            node = new_start_node            
        
        # # Simple way to capture nodes to be removed in case of two way link without intersection:
        # if G.in_degree(node)==G.out_degree(node)==2:
        #     if set(G.predecessors(node))==set(G.successors(node)):
        #         neibors = list(G.predecessors(node))
        #         edge1 = G.get_edge_data(neibors[0], node)
        #         edge1c = G.get_edge_data(node, neibors[1])
                
        #         edge2 = G.get_edge_data(neibors[1], node)
        #         edge2c = G.get_edge_data(node, neibors[0])
        #         if ((edge1["modes"] == edge1c["modes"])&
        #             (edge1["capacity"] == edge1c["capacity"])&
        #             (abs(edge1["freespeed"]-edge1c["freespeed"])<1)&
        #             (edge2["modes"] == edge2c["modes"])&
        #             (edge2["capacity"] == edge2c["capacity"])&
        #             (abs(edge2["freespeed"]-edge2c["freespeed"])<1)): 
                    
        #             nodes_to_remove.append(node)
        #             # I can do that, or build tuples of links to merge and merge them later
        #             # However, it is not a good thing to do, because if they share the same 
        #             # node, it means vehicles can turn at that node. If the two links are merged
        #             # The vehicle cannot turn.
        
        
        #Skip if not potential chain start            
        if (G.degree(node) <= 2 or G.out_degree(node) == 0) and attribute_consistency:        
            continue
        
        attribute_consistency = True
        # Potential chain start point
        for succ in G.successors(node):
            edge_data = G.get_edge_data(node, succ)[0]
            idx = edge_data['idx']
            
            if idx in visited_links:
                stats["already_visited"]+=1
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
                edge_data = G.get_edge_data(current_node, next_node)[0]
     

                next_idx = edge_data['idx']
                next_row = df.loc[next_idx]

                # Check attribute consistency
                if not ((next_row["modes"] == current_attrs["modes"])&
                        (next_row["capacity"] == current_attrs["capacity"])&
                        (abs(next_row["freespeed"]-current_attrs["freespeed"])<1)):   
                    attribute_consistency = False
                    new_start_node = current_node
                    stats['attributes_change']+=1
                    break
                else:
                   attribute_consistency = True 

                current_chain.append(next_idx)
                current_node = next_node

            if len(current_chain) > 1:
                # Merge the chain
                chain_rows = df.loc[current_chain]
                first_node = df.loc[current_chain[0], 'from_node']
                last_node  = df.loc[current_chain[-1], 'to_node']
                if first_node!=last_node:
                    # Otherwise, it would just create a loop               
                    merged_links.append({
                        'from_node': first_node,
                        'to_node': last_node,
                        'link_id': "_".join(df.loc[current_chain, 'link_id'].tolist()),
                        'length': chain_rows['length'].sum(),
                        **current_attrs
                    })
                    visited_links.update(current_chain)
                    stats["successful_merge"]+=1
                else:
                    stats["skiped_loop"]+=1

    # Step 3: Build final DataFrame
    links_to_remove = visited_links
    df_cleaned = df[~df.index.isin(links_to_remove)]
    df_merged = pd.DataFrame(merged_links)
    final_df = pd.concat([df_cleaned, df_merged], ignore_index=True)
    
    progress_bar.close()
    return final_df


cleaned_df = merge_link_chains_directed(df)


net.links = pd.concat([cleaned_df,df_rest], ignore_index=True)


unique_nodes = pd.concat([df.from_node, df.to_node]).unique()
net.nodes = net.nodes[net.nodes.node_id.isin(unique_nodes)].reset_index(drop=True)

link_attrs = net.link_attrs.groupby('link_id').apply(lambda x: dict(zip(x['name'], x['value']))).reset_index(name='attributes')
net.links = net.links.merge(link_attrs, on="link_id",how="left")
net.links.loc[net.links["attributes"].isna(), "attributes"] = None

with open("cleaning_stats.json", "w") as f:
    json.dump(stats, f, indent=4) 
    
# print("Saving network")
# net.save("new_network.xml.gz")



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
