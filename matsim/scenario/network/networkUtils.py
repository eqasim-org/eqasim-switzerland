import pandas as pd
import geopandas as gpd
import networkx as nx 
from shapely import wkt
from shapely.errors import WKTReadingError
from shapely.geometry import Point
import numpy as np


class TrafficLightsMatcher:
    """
    Class to match traffic lights to the network links.
    
    Parameters:
        links (pd.DataFrame): DataFrame containing the network links.
    """
    
    def __init__(self, network):
        self.links = network.links
        self.nodes = network.nodes        

    def read_detailed_network(self, detailed_network_path):
        """
        Read the detailed network geometry.
        
        Parameters:
            detailed_network_path (str): Path to the detailed network geometry file.
        
        Returns:
            pd.DataFrame: DataFrame containing the detailed network geometry.
        """
        detailed_geo = pd.read_csv(detailed_network_path)
        detailed_geo = detailed_geo.rename(columns={"Geometry": "geometry", "LinkId": "link_id"})
        # convert ids to str for consistency
        detailed_geo["link_id"] = detailed_geo["link_id"].astype(str)
        # Remove links that are not car links
        detailed_geo = detailed_geo.merge(self.links[["link_id","from_node","to_node", "modes"]])
        detailed_geo = detailed_geo[(detailed_geo["modes"].str.contains("car"))&(~detailed_geo["link_id"].str.contains("pt"))]
        
        # convert the geometry from string to shapely geometries
        def safe_load_wkt(wkt_str):
            try:
                return wkt.loads(wkt_str)
            except (WKTReadingError, ValueError, AttributeError, TypeError):
                return None
        detailed_geo["geometry"] = detailed_geo["geometry"].map(safe_load_wkt)
        detailed_geo = gpd.GeoDataFrame(detailed_geo, geometry = "geometry", crs = "EPSG:2056")
        return detailed_geo

    def read_traffic_lights(self, traffic_lights_path):
        """
        Read the traffic lights data.
        
        Parameters:
            traffic_lights_path (str): Path to the traffic lights data file.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the traffic lights data.
        """
        traffic_lights = gpd.read_file(traffic_lights_path)
        # Ensure the CRS is set correctly
        traffic_lights = traffic_lights.to_crs("EPSG:2056")
        return traffic_lights

    def as_nx(self):
        """ Convert the road network to a NetworkX graph.
        Returns:
            nx.Graph: NetworkX graph representation of the network.
        """
        G = nx.Graph()  
        sel = (self.links["modes"].str.contains("car"))&(~self.links["link_id"].str.contains("pt"))
        # Add nodes with coordinates        
        G.add_nodes_from(
            zip(self.nodes['node_id'], 
                ({'x': x, 'y': y} for x, y in zip(self.nodes['x'], self.nodes['y'])))
        )

        # Add edges with attributes
        G.add_edges_from(zip(self.links.loc[sel,'from_node'], self.links.loc[sel,'to_node']))
        return G

    def nodes_as_geo(self):
        """ Convert the nodes of the network to a GeoDataFrame.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the nodes of the network.
        """
        return gpd.GeoDataFrame(
            self.nodes, 
            geometry=gpd.points_from_xy(self.nodes['x'], self.nodes['y']),
            crs="EPSG:2056")

    def preprocess_matched_traffic_light_nodes(self, G, traffic_light_nodes, threshold=30):
        sel = (self.links["modes"].str.contains("car"))&(~self.links["link_id"].str.contains("pt"))
        sel|= (self.links.from_node.isin(traffic_light_nodes)|self.links.to_node.isin(traffic_light_nodes))
        
        links = self.links[sel].copy()
        all_nodes = set([*links['from_node'], *links['to_node']])
        degrees = dict(G.degree(all_nodes))
        two_degree_nodes = [k for k,v in degrees.items() if v==2]
        
        # Create a mapping from node → (to_node, length)
        target_links = links[(links['from_node'].isin(two_degree_nodes))&(links['length'] < threshold)]
        target_links["downstream_degree"] = target_links["to_node"].apply(lambda x: G.degree(x) )
        target_links["class_distance"] = target_links["length"] - 5*(target_links["downstream_degree"]-2) #5 meters by additional degree
        
        target_links = target_links.sort_values('class_distance').drop_duplicates('from_node', keep='first')
        
        node_transfer_map = target_links.set_index('from_node')['to_node'].to_dict()
        distance_map = target_links.set_index('from_node')['length'].to_dict()

        # move the node down to the intersection, or a max distance of 40 meters    
        def transfer_node(node, distance = 0):
            if distance>40 or pd.isna(node): 
                return node            
            new_node = node_transfer_map.get(node, node)
            if new_node==node:
                return new_node        
            transfered_distance = distance + distance_map.get(node)
            return transfer_node(new_node, transfered_distance)        
            
        # Use the map to transfer nodes efficiently
        traffic_light_nodes = np.unique([transfer_node(node) for node in traffic_light_nodes])
        return traffic_light_nodes
        
    def run(self, traffic_lights_path, detailed_network_path):
        """
        Add traffic lights to the network.
        
        Parameters:
            traffic_lights_path (str): Path to the traffic lights data.
            detailed_network_path (str): Path to save the detailed network with traffic lights.
        """        
        # Read the traffic lights data
        traffic_lights = self.read_traffic_lights(traffic_lights_path)
        
        #Read the geometry of the detailed network
        detailed_geo = self.read_detailed_network(detailed_network_path)
        
        # Get the nodes geometry
        nodes_geo = self.nodes_as_geo()

        # Get the graph representation of the network (for fast degrees access)
        G = self.as_nx()

        # Make the geospatial join to find the traffic lights on the links
        assert traffic_lights.crs==detailed_geo.crs, "Traffic lights and detailed network must have the same CRS."
        ### FIRST: MERGE THE NODES
        matched_nodes = traffic_lights.sjoin_nearest(nodes_geo[["node_id","geometry"]], how="left",lsuffix = "", rsuffix="matsim", 
                                                     max_distance=2)["node_id_matsim"].unique()
        ### SECOND: MERGE THE LINKS
        matched_links = traffic_lights.sjoin_nearest(detailed_geo, how="left", lsuffix="_tl", rsuffix="_link", max_distance=5)        
        matched_links = matched_links[matched_links.link_id.notna()]
        matched_links = matched_links.merge(detailed_geo[["link_id","geometry"]].rename(columns={"geometry":"link_geometry"}), on="link_id", how="left")

        ### FROM THE MATCHED LINKS, GET THE INTERSECTION NODES
        def get_intersection_node(g):
            entry, exit = map(Point, [g.link_geometry.coords[0], g.link_geometry.coords[-1]])
            entry_distance, exit_distance = entry.distance(g.geometry), exit.distance(g.geometry)   
            entry_degree, exit_degree = G.degree(g.from_node), G.degree(g.to_node)
            #traffic lights in the middle (43%->58% of the link length) will be attributed to the node with higher degree
            equivalent_length = 0.2*g.link_geometry.length
            entry_distance   -= equivalent_length if entry_degree>exit_degree else 0
            exit_distance    -= equivalent_length if exit_degree>entry_degree else 0
            return g["from_node"] if entry_distance<exit_distance else g["to_node"]

        matched_links["matsim_node"] = matched_links.apply(get_intersection_node, axis=1).astype(str)

        ### MERGE THE NODES OBTAINED FROM MERGING NODES AND LINKS
        traffic_light_nodes = set(matched_links["matsim_node"].unique()).union(matched_nodes)

        ### PROCESS THESE NODES, SOME OF WHICH HAVE A DEGREE OF 2 (TO BE REMOVED). AND SOME OF WHICH ARE IN THE WRONG INTERSECTION
        traffic_light_nodes = self.preprocess_matched_traffic_light_nodes(G, traffic_light_nodes)        
        traffic_light_nodes = [node for node in traffic_light_nodes if G.degree(node)>2]
        
        # Now all the links that are going to the node where traffic lights are assigned, are taged in their attributes
        links_having_traffic_lights = self.links.to_node.isin(traffic_light_nodes)
        self.links.loc[links_having_traffic_lights, "attributes"] = \
            self.links.loc[links_having_traffic_lights, "attributes"].apply(lambda x: {**x, "traffic_light":True})
        
        return self.links



class CapacityCorrector:
    """
    Class to correct the capacity of the links in the network.
    
    Parameters:
        network (Network): The network object containing the links.
    """
    
    def __init__(self, network):
        self.links = network.links

    def _correct_capacity(self, row, sampling_rate, minimum_speed):
        current_capacity = row["capacity"]
        length = row["length"]
        return np.maximum(current_capacity, 3600*minimum_speed/(length*sampling_rate))
    
    
    def run(self, sampling_rate, minimum_speed=2/3.6):
        """        
        This function here correct the capacity of a link based on its length and 
        the sampling rate of teh population. It will only impact small links.
        """
        car_links = self.links.modes.apply(lambda x: "car" in x)
        
        self.links.loc[car_links, "capacity"] = \
            self.links.loc[car_links, ["capacity", "length"]].apply(
            lambda x: self._correct_capacity(x, sampling_rate, minimum_speed),
            axis=1)
        return self.links










