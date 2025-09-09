import warnings
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)

class networkCleaner():
    def __init__(self, network):
        self.links = network.links.copy()
        self.nodes = network.nodes.copy()        
        self.network = network

    def run(self):
        stats = self.clean_network()
        self.network.nodes = self.nodes
        self.network.links = self.links
        return self.network, stats

    def clean_network(self):
        stats = dict()

        """All network should be considered, because intersections could be with (pt) and (pt,car)"""
        df = self.links

        # Removing loops
        sel = (df['from_node'] == df['to_node'])
        stats["removed_loops"] = int(sel.sum())
        logger.info("There are %d loops in the network that are removed." % stats["removed_loops"] )
        df = df[~sel].reset_index(drop=True)

        # Remove replicated links
        len_df = len(df)
        df = df.drop_duplicates(subset=['length','modes','from_node', 'to_node', 'capacity'],
                                ignore_index=True)
        stats["removed_link_duplicates"] = len_df-len(df)
        logger.info("There are %d link duplicates in the network that are removed." % stats["removed_link_duplicates"] )                
        
        # Removing nodes with no intersection
        logger.info("Removing nodes that do not represent an intersection.")
        df, stats2 = self.merge_link_chains(df)
        
        # Removed links that are not connected to the whole graph
        logger.info("Remove unconnected links")
        cond = "disallowedNextLinks" in self.network.link_attrs["name"].unique()
        func = self.remove_unconnected_links_with_turns_restrictions if cond else self.remove_unconnected_links
        df, num_removed = func(df)        
        stats["removed_unconnected_links"] = num_removed

        self.links = df
        stats.update(stats2)
        
        # Limit the speed limit in the network
        logger.info("Change the infinit freespeed to 85.")
        # This is also done in : https://github.com/eqasim-org/eqasim-java/blob/develop/core/src/main/java/org/eqasim/core/scenario/preparation/AdjustLinkLength.java#L9
        self.links.loc[self.links.freespeed.apply(np.isinf),"freespeed"] = 85
        self.links.loc[self.links.freespeed<20/3.6,"freespeed"] = 20/3.6

        unique_nodes = set(df.from_node.tolist() + df.to_node.tolist())
        self.nodes = self.nodes[self.nodes.node_id.isin(unique_nodes)].reset_index(drop=True)
        
        if "attributes" not in self.links:
            warnings.warn("Attributes are not present in the links dataframe.", category=UserWarning)
        
        return stats
        
    def remove_unconnected_links(self, df):        
        """
        Removes links that are not part of the largest connected component of the road network.
        
        Parameters:
            df (pd.DataFrame): DataFrame with road network links. Required columns:
                               ['from_node', 'to_node', 'link_id']
        
        Returns:
            pd.DataFrame: Filtered DataFrame with only the connected links.
        """        
        
        df = df.copy()
        sel = df["modes"].apply(lambda x: "car" in x)
        df_other = df[~sel].copy()
        df       = df[sel].reset_index(drop=True) #don't include pt links because maybe disconencted from the network (rail for example)
        
        logger.info("    Converting network to networkx ...")
        G = nx.Graph()    
        G.add_edges_from(zip(df['from_node'], 
                             df['to_node'], 
                            ({'idx': idx, 'link_id': lid} for idx, lid in zip(df.index, df['link_id']))))
    
        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
    
        # Keep only edges where both nodes are in the largest component
        connected_links = df[df['from_node'].isin(largest_cc) | df['to_node'].isin(largest_cc)]
        
        df = pd.concat([connected_links, df_other], ignore_index=True)
        return df, len(sel)-len(df)
        
    def remove_unconnected_links_with_turns_restrictions(self, df):
        """
        Removes links not part of the largest connected component for cars, considering turn restrictions.
        In this case, the links are the nodes of the created graph.
        Parameters:
            df (pd.DataFrame): Road network with columns ['from_node', 'to_node', 'link_id', 'modes', 'disallowedNextLinks']
        
        Returns:
            pd.DataFrame: Filtered DataFrame with connected car links only.
        """        
        df = df.copy()
        
        # Filter to car-accessible links
        sel = df["modes"].apply(lambda x: "car" in x)
        df_other = df[~sel].copy()
        df = df[sel].reset_index(drop=True)

        logger.info("    Building turn-aware connectivity graph...")

        # Create a mapping from to_node to list of incoming links
        links_from_node = df.groupby('from_node')['link_id'].apply(list).to_dict()
        link_info = df.set_index('link_id')

        # Build a graph where nodes are link_ids, and edges represent allowed turns
        G = nx.DiGraph()
        for link_id, row in link_info.iterrows():
            to_node = row['to_node']
            disallowed = set()          
            disallowed_links = eval(row.attributes.get("disallowedNextLinks", "{}"))
            if "car" in disallowed_links:                
                disallowed.update([j for i in disallowed_links["car"] for j in i])

            # Get all links starting from this to_node (i.e., possible successors)
            next_links = links_from_node.get(to_node, [])
            for next_link_id in next_links:
                if next_link_id not in disallowed:
                    G.add_edge(link_id, next_link_id)

        # Get largest strongly connected component
        strongly_connected_components = list(nx.strongly_connected_components(G))
        largest_scc = max(strongly_connected_components, key=len)

        # Filter to only links part of the largest component
        connected_links = df[df['link_id'].isin(largest_scc)]
        
        # print number of unconnected links
        disconnected_components = [c for c in strongly_connected_components if len(c) < len(largest_scc)]
        logger.info(f"{len(disconnected_components)} small disconnected components found. These are their sizes: {[len(c) for c in disconnected_components]}")   
        
        # Add back non-car links
        df_final = pd.concat([connected_links, df_other], ignore_index=True)

        removed_count = len(df) - len(connected_links)
        return df_final, removed_count

    def merge_link_chains(self, df):        
        stats = {"number_of_nodes":0,
                 "number_of_links":0,
                 "attributes_change":0,
                 "skiped_loop":0,
                 "successful_merge":0,
                 "degree_is_2":0,  
                 "one_in_one_out":0,                        
                 "ignored_no_car":0,
                 "break_no_car":0,
                 "already_visited":0, 
                    }
                        
        # Step 1: Build directed graph
        logger.info("    Converting network to networkx ...")
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
        
        # Step 2: Identify nodes that are NOT degree-2 (start/end points)
        def is_degree2(node):
            return G.in_degree(node) == 1 and G.out_degree(node) == 1
        
        
        logger.info("    Searching for nodes to remove")
        node_iterator = G.__iter__()
        iteration = 0
        progress_bar = tqdm(total=len(G), desc="Finding nodes to remove ", 
                            disable= not os.isatty(1) )
        
        while iteration < len(G):   
            if attribute_consistency:
                node = next(node_iterator)
                iteration+=1            
                progress_bar.update(1) # Updtae the bar only here (follow the iterator)
                
                stats['degree_is_2']+=int(G.degree(node)==2)
                stats['one_in_one_out']+=int(is_degree2(node))
            else:
                node = new_start_node            
            
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
                                
                if not "car" in df.loc[idx,"modes"]:
                    stats["ignored_no_car"]+=1
                    continue
                
                # Start building the chain
                current_chain = [idx]
                current_node = succ
                current_attrs = df.loc[idx, ['link_id', 'freespeed', 'capacity', 'permlanes',
                                             'oneway', 'modes']].to_dict()

                while is_degree2(current_node):
                    next_nodes = list(G.successors(current_node))
                    if len(next_nodes) != 1:
                        break

                    next_node = next_nodes[0]
                    edge_data = G.get_edge_data(current_node, next_node)[0]                    

                    next_idx = edge_data['idx']
                    next_row = df.loc[next_idx]

                    if not "car" in next_row['modes']:
                        stats["break_no_car"]+=1
                        break

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
                        new_link = {
                            'from_node': first_node,
                            'to_node': last_node,                            
                            'length': chain_rows['length'].sum(),
                            **current_attrs #we use the link_id of the first link
                        }
                        new_link["attributes"] = df.loc[current_chain[-1], 'attributes'] #should keep the attributes of last link, in case of turn restrictions
                        new_link["attributes"]["old_link_id"] = "_".join(chain_rows['link_id'].tolist())
                        merged_links.append(new_link)
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
        return final_df, stats
    