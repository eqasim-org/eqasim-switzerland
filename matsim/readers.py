# -*- coding: utf-8 -*-

import xopen
import xml.etree.ElementTree as ET
import pandas as pd
from .writers import NetworkWriter
import gzip
from tqdm import tqdm
import os
import numpy as np

"""
This script is designed to read MATSim network files. It also include a function that simplifies the network, 
by removing nodes that do not represent intersection and where links attributes do not change.

It is partially based on the implementation from:
https://github.com/matsim-vsp/matsim-python-tools/blob/master/matsim/Network.py

Additional network readers may be integrated in the future as needed.
"""

class Network:

    _crsTag = 'coordinateReferenceSystem'

    def __init__(self, nodes, links, node_attrs, link_attrs, net_attrs=None):
        self.nodes = nodes
        self.links = links
        self.link_attrs = link_attrs
        self.node_attrs = node_attrs

        self.network_attrs = {}
        if net_attrs: self.network_attrs = net_attrs

    def __str__(self):
        return 'Network: {nodes} nodes, {links} links, {crs}'.format(
            nodes=len(self.nodes),
            links=len(self.links),
            crs=Network._crsTag in self.network_attrs and self.network_attrs[Network._crsTag] or 'No CRS')
    
    def as_geo(self, projection=None):
        import geopandas as gpd
        import shapely.geometry as shp
    
        """Return a GeoPandas GeoDataFrame containing link geometries suitable for plotting."""
    
        # Project the coords, if CRS is specified somehow
        if projection:
            crs = {'init': projection}
        elif Network._crsTag in self.network_attrs:
            crs = self.network_attrs[Network._crsTag]
        else:
            crs = None
    
        # attach xy to links
        full_net = (self.links
        .merge(self.nodes,
                left_on='from_node',
                right_on='node_id')
        .merge(self.nodes,
                left_on='to_node',
                right_on='node_id',
                suffixes=('_from_node', '_to_node'))
        )
    
        # create the geometry column from coordinates
        geometry = [shp.LineString([(ox,oy), (dx,dy)]) for ox, oy, dx, dy in zip(full_net.x_from_node, full_net.y_from_node, full_net.x_to_node, full_net.y_to_node)]
    
        # build the geopandas geodataframe
        geo_net = (gpd.GeoDataFrame(full_net,
            geometry=geometry,
            crs = crs)
            .drop(columns=['x_from_node','y_from_node','node_id_from_node','node_id_to_node','x_to_node','y_to_node'])
            )
    
        return geo_net
    
    
    def clean_network(self):
        stats = dict()

        # Only treat road network obtained from osm, don't touch pt links
        sel = self.links.link_id.apply(lambda x: "pt" not in x)
        sel &= (self.links.modes.apply(lambda x: "car" in x))
        df = self.links[sel]
        df_rest = self.links[~sel]

        # Removing loops
        sel = (df['from_node'] == df['to_node'])
        stats["removed_loops"] = int(sel.sum())
        print("There are %d loops in the network that are removed." % stats["removed_loops"] )
        df = df[~sel].reset_index(drop=True)

        # Remove replicated links
        len_df = len(df)
        df = df.drop_duplicates(subset=['length','modes','from_node', 'to_node', 'capacity'],
                                ignore_index=True)
        stats["removed_link_duplicates"] = len_df-len(df)
        print("There are %d link duplicates in the network that are removed." % stats["removed_link_duplicates"] )
        
        print("Remove unconnected links")
        df, num_removed = self.remove_unconnected_links(df)
        stats["removed_unconnected_links"] = num_removed
        
        # Removing nodes with no intersection
        print("Removing nodes that do not represent an intersection.")
        df, stats2 = self.merge_link_chains(df)
        df = pd.concat([df,df_rest], ignore_index=True)
        self.links = df
        stats.update(stats2)
        
        print("Change the infinit freespeed to 85.")
        # This is also done in : https://github.com/eqasim-org/eqasim-java/blob/develop/core/src/main/java/org/eqasim/core/scenario/preparation/AdjustLinkLength.java#L9
        self.links.loc[self.links.freespeed.apply(np.isinf),"freespeed"] = 85
        self.links.loc[self.links.freespeed<20/3.6,"freespeed"] = 20/3.6
        
        unique_nodes = pd.concat([df.from_node, df.to_node]).unique()
        self.nodes = self.nodes[self.nodes.node_id.isin(unique_nodes)].reset_index(drop=True)

        link_attrs = self.link_attrs.groupby('link_id').apply(lambda x: dict(zip(x['name'], x['value']))).reset_index(name='attributes')
        self.links = self.links.merge(link_attrs, on="link_id",how="left")
        self.links.loc[self.links["attributes"].isna(), "attributes"] = None
        
        return stats
    
    @classmethod
    def remove_unconnected_links(self, df):        
        """
        Removes links that are not part of the largest connected component of the road network.
        
        Parameters:
            df (pd.DataFrame): DataFrame with road network links. Required columns:
                               ['from_node', 'to_node', 'link_id']
        
        Returns:
            pd.DataFrame: Filtered DataFrame with only the connected links.
        """
        
        import networkx as nx #only if we use this function
        
        print("    Converting network to networkx ...")
        G = nx.Graph()    
        G.add_edges_from(zip(df['from_node'], 
                             df['to_node'], 
                            ({'idx': idx, 'link_id': lid} for idx, lid in zip(df.index, df['link_id']))))
    
        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
    
        # Keep only edges where both nodes are in the largest component
        connected_links = df[df['from_node'].isin(largest_cc) & df['to_node'].isin(largest_cc)]
    
        return connected_links.reset_index(drop=True), len(df)-len(connected_links)
    

    @classmethod
    def merge_link_chains(self, df):
        import networkx as nx #only if we use this function
        stats = {"number_of_nodes":0,
                 "number_of_links":0,
                 "attributes_change":0,
                 "skiped_loop":0,
                 "successful_merge":0,
                 "degree_is_2":0,  
                 "one_in_one_out":0,
                 "already_visited":0,        
                    }
                        
        # Step 1: Build directed graph
        print("    Converting network to networkx ...")
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
        
        
        print("    Searching for nodes to remove")
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
                            'link_id': "_".join(chain_rows['link_id'].tolist()),
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
        return final_df, stats
    
    def save(self,file_path,write_attrbs=True):        
        # set the types
        self.nodes = self.nodes.astype({'node_id':str,'x':float,'y':float})
        self.links = self.links.astype({'link_id':str,
                                        'from_node':str, 
                                        'to_node':str, 
                                        'length':float,     
                                        'freespeed':float, 
                                        'capacity':int, 
                                        'permlanes':int,      
                                        'oneway':int,
                                        'modes':str,
                                        })
        
        # save the network in xml file
        file_open = gzip.open if file_path.endswith('.gz') else open
        with file_open(file_path, 'wb+') as f_write:
            writer = NetworkWriter(f_write)
            writer.start_network()
            writer.add_nodes(self.nodes['node_id'],self.nodes.x, self.nodes.y)
            writer.add_links(self.links['link_id'],
                             self.links['from_node'],
                             self.links['to_node'],
                             self.links['length'],
                             self.links['freespeed'],
                             self.links['capacity'],
                             self.links['permlanes'],
                             self.links['oneway'],
                             self.links['modes'],
                             self.links['attributes'] if "attributes" in self.links else None,
                             write_attrbs = write_attrbs)
            writer.end_network()

    
    def _correct_capacity(self, row, sampling_rate, minimum_speed):
        current_capacity = row["capacity"]
        length = row["length"]
        return np.maximum(current_capacity, 3600*minimum_speed/(length*sampling_rate))
    
    
    def correct_capacity(self, sampling_rate, minimum_speed=2/3.6):
        """        
        This function here correct the capacity of a link based on its length and 
        the sampling rate of teh population. It will only impact small links.
        """
        car_links = self.links.modes.apply(lambda x: "car" in x)
        
        self.links.loc[car_links, "capacity"] = \
            self.links.loc[car_links, ["capacity", "length"]].apply(
            lambda x: self._correct_capacity(x, sampling_rate, minimum_speed),
            axis=1)
    
    def correct_speed(self, current_speed, length, time_step = 1):
        steps = np.maximum(1,np.round(length/(current_speed*time_step)))
        return length/(steps*time_step)
        
        

def read_network(filename, skip_attributes=False):
    """Read a MATSim network.xml.gz file. Returns a Network object with dataframes
    for nodes, links, node_attributes, and link_attributes. If the network has a CRS
    projection set, it will be available in network_attrs."""
    tree = ET.iterparse(xopen.xopen(filename, 'r'), events=['start', 'end'])
    nodes = []
    links = []
    node_attrs = []
    link_attrs = []

    network_attrs = {}

    attributes = node_attrs
    attr_label = 'node_id'
    current_id = None

    for xml_event, elem in tree:
        # the nodes element CLOSES at the end of the nodes, followed by links:
        if elem.tag == 'links' and xml_event == 'start':
            attributes = link_attrs
            attr_label = 'link_id'

        elif elem.tag == 'node' and xml_event == 'start':
            atts = elem.attrib
            current_id = atts['id']

            atts['node_id'] = atts.pop('id')
            atts['x'] = float(atts['x'])
            atts['y'] = float(atts['y'])
            if 'z' in atts: atts['z'] = float(atts['z'])

            nodes.append(atts)

        elif elem.tag == 'link' and xml_event == 'start':
            atts = elem.attrib
            current_id = atts['id']

            atts['link_id'] = atts.pop('id')
            atts['from_node'] = atts.pop('from')
            atts['to_node'] = atts.pop('to')

            atts['length'] = float(atts['length'])
            atts['freespeed'] = float(atts['freespeed'])
            atts['capacity'] = float(atts['capacity'])
            atts['permlanes'] = float(atts['permlanes'])

            if 'volume' in atts: atts['volume'] = float(atts['volume'])

            links.append(atts)


        elif elem.tag == 'attribute' and xml_event == 'end':
            if elem.attrib['name'] == Network._crsTag:
                network_attrs[Network._crsTag] = elem.text

            elif not skip_attributes:
                atts = {}
                atts[attr_label] = current_id
                atts['name'] = elem.attrib['name']
                atts['value'] = elem.text

                # TODO: pandas will make the value column "object" since we're mixing types
                if 'class' in elem.attrib:
                    if elem.attrib['class'] == 'java.lang.Long':
                        atts['value'] = int(elem.text)
                    if elem.attrib['class'] == 'java.lang.Double':
                        atts['value'] = float(elem.text)
                    if elem.attrib['class'] == 'java.lang.Integer':
                        atts['value'] = int(elem.text)

                attributes.append(atts)

        # clear the element when we're done, to keep memory usage low
        if elem.tag in ['node', 'link'] and xml_event == 'end':
            elem.clear()

    nodes = pd.DataFrame.from_records(nodes)
    links = pd.DataFrame.from_records(links)
    node_attrs = pd.DataFrame.from_records(node_attrs)
    link_attrs = pd.DataFrame.from_records(link_attrs)

    return Network(nodes, links, node_attrs, link_attrs, network_attrs)