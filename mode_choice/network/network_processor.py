"""
Network processing utilities for MATSim network data.
Handles network loading, graph construction, and congestion integration.
"""

import os
import pandas as pd
import numpy as np
from .road_network import read_network, Network
from typing import Dict, Tuple, Optional
import logging
import pickle

logger = logging.getLogger(__name__)


class NetworkProcessor:
    """
    Processes MATSim network data and creates a routing-ready graph with congestion information.
    """
    
    def __init__(self, network_file: str = None, network: Network = None, 
                 congestion_file: Optional[str] = None, graph_type: str = "igraph"):
        """
        Initialize the network processor.
        
        Args:
            network_file: Path to the MATSim network XML file
            network: Network object (optional)
            congestion_file: Path to the congestion data file (optional)
            graph_type: Type of graph to create ("igraph", "networkx", "pandana")
        """
        assert graph_type in ["igraph", "networkx", "pandana"], "Unsupported graph type"
        assert (network_file is not None) or (network is not None), "Either network_file or network must be provided"
        
        self.network_file = network_file
        self.congestion_file = congestion_file
        self.graph_type = graph_type        
        self.network = network
        self.graph = None
        self.congestion_data = None
        self.congestion_aware = False

    def load_network(self):
        """
        Load the MATSim network file.
        
        Returns:
            Dictionary containing links and nodes data
        """
        if self.network is not None:
            return 

        logger.info(f"\t Reading network from {self.network_file}")
        net = read_network(self.network_file)
        
        # make sure id is str
        net.links['link_id'] = net.links['link_id'].astype(str)
        net.links['from_node'] = net.links['from_node'].astype(str)
        net.links['to_node'] = net.links['to_node'].astype(str)
        net.nodes['node_id'] = net.nodes['node_id'].astype(str)

        # filter out non-car links
        car_links = net.links["modes"].str.split(',').map(lambda x: "car" in x)
        net.links = net.links[car_links].reset_index(drop=True)
        net.nodes = net.nodes[net.nodes['node_id'].isin(
            pd.unique(net.links['from_node'].tolist() + net.links['to_node'].tolist())
        )].reset_index(drop=True)        

        # del unecessary data
        del net.link_attrs
        net.links = net.links.drop(columns=['capacity', 'permlanes', 'oneway'])
        
        # save
        self.network = net
        logger.info(f"\t Loaded {len(self.network.links)} links and {len(self.network.nodes)} nodes")

    def load_congestion_data(self):
        """
        Load and process congestion data.
        
        Returns:
            DataFrame with processed congestion data
        """
        if not self.congestion_file:
            logger.warning("No congestion file provided")
            return None
        
        if not os.path.exists(self.congestion_file):
            logger.warning("Congestion file does not exist")
            return None
        
        logger.info(f"\t Reading congestion data from {self.congestion_file}")
        
        # Create renaming and dtypes map for congestion columns
        renaming_map = {"LINK": "link_id"}
        dtype_map = {"LINK": str}
        for i in range(24):
            old = f"TRAVELTIME{i}-{i+1}avg"
            new = f"travel_time_{i}_{i+1}_avg"
            renaming_map[old] = new
            dtype_map[old] = "float32"

        congestion_data = pd.read_csv(
            self.congestion_file,
            sep="\t",
            usecols=renaming_map.keys(),
            dtype=dtype_map
        ).rename(columns=renaming_map)

        # Ensure types after renaming
        congestion_data["link_id"] = congestion_data["link_id"].astype(str)
        for i in range(24):
            col = f"travel_time_{i}_{i+1}_avg"
            if col in congestion_data.columns:
                congestion_data[col] = congestion_data[col].round(1).astype("float32")
        
        # only keep links present in the network
        if self.network is None:
            self.load_network()

        congestion_data = congestion_data[
            congestion_data['link_id'].isin(self.network.links['link_id'])
        ].reset_index(drop=True)
        
        # store congestion data
        self.congestion_data = congestion_data
        self.congestion_aware = True
        logger.info(f"\t Loaded congestion data for {len(congestion_data)} links")        
    
    def created_congestion_aware_network(self):
        """
        Create a congestion-aware network by integrating congestion data.
        """
        if self.network is None:
            self.load_network()
        
        network = self.network.links
        network["travel_time"] = (network["length"] / network["freespeed"]
                                  ).round(1).astype("float32")

        if self.congestion_data is None and self.congestion_file is not None:
            self.load_congestion_data()

        if self.congestion_data is None:
            logger.info("No congestion file provided; using free-flow travel times")
            return network
        
        network = network.merge(self.congestion_data, how='left', on='link_id')
        min_speed_travel_time = (network["length"] / (1 / 3.6)).round(1).astype("float32")  # speed = 1 km/h
        for i in range(24):
            col = f"travel_time_{i}_{i+1}_avg"            
            if col in self.congestion_data.columns:
                # fill the missing values with free-flow travel time
                network[col] = network[col].fillna(network["travel_time"])
                # ensure no travel time is less than free-flow travel time
                network[col] = np.maximum(network[col], network["travel_time"])
                # ensure no travel time is higher than travel time corresponding to speed 1 km/h                
                network[col] = np.minimum(network[col], min_speed_travel_time)
                # ensure it is float32
                network[col] = network[col].astype("float32")

        return network

    def build(self):
        """
        Create a graph representation from the MATSim network.
        """
        if self.network is None:
            self.load_network()

        if self.congestion_data is None and self.congestion_file is not None:
            self.load_congestion_data()

        if self.graph is None:
            if self.graph_type == "igraph":
                self.create_igraph()
            elif self.graph_type == "networkx":
                self.create_nx_graph()
            elif self.graph_type == "pandana":
                self.create_pandana_graph()
            else:
                raise ValueError(f"Unsupported graph type: {self.graph_type}")
    
    def create_igraph(self):
        """
        Create an igraph directed graph from the MATSim network.
        
        Returns:
            igraph Graph with link attributes
        """
        from igraph import Graph
        if self.network is None:
            self.load_network()
            
        logger.info("\t Creating igraph graph from MATSim network")

        graph = Graph(directed=True)
        
        # Add vertices using the node_id as name
        graph.add_vertices(self.network.nodes["node_id"].tolist())

        # Add edges using from_node → to_node
        links = self.created_congestion_aware_network()
        graph.add_edges(zip(links["from_node"], links["to_node"]))
        
        # Set all travel times as edge attributes
        graph.es["length"] = links["length"].values
        for c in links.columns:
            if "travel_time" in c:
                graph.es[c] = links[c].values

        self.graph = graph
        logger.info(f"Created graph with {self.graph.vcount()} nodes and {self.graph.ecount()} edges")

    def create_nx_graph(self):
        """
        Create a NetworkX directed graph from the MATSim network.
        
        Returns:
            NetworkX DiGraph with link attributes
        """
        import networkx as nx
        if self.network is None:
            self.load_network()
            
        logger.info("\t Creating NetworkX graph from MATSim network")

        graph = nx.DiGraph()
        
        # Add nodes
        graph.add_nodes_from(self.network.nodes["node_id"].tolist())

        # Add edges with attributes
        links = self.created_congestion_aware_network()
        travel_time_cols = ["length"] + [col for col in links.columns if "travel_time" in col]
        graph.add_edges_from(
            zip(links['from_node'], links['to_node'],
                (l.to_dict() for _, l in links[travel_time_cols].iterrows())
            )
        )

        self.graph = graph
        logger.info(f"Created NetworkX graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    def create_pandana_graph(self):
        """
        Create a Pandana graph from the MATSim network.
        
        Returns:
            Pandana Network with link attributes
        """
        import pandana as pdna
        if self.network is None:
            self.load_network()
            
        logger.info("\t Creating Pandana graph from MATSim network")

        nodes = self.network.nodes                
        links = self.created_congestion_aware_network()
        
        # get indices        
        node_id_map = dict(zip(nodes["node_id"], nodes.index))
        from_nodes = links["from_node"].map(lambda x: node_id_map.get(x)).astype('int32')
        to_nodes = links["to_node"].map(lambda x: node_id_map.get(x)).astype('int32')

        # columns
        travel_time_cols = ["length"] + [col for col in links.columns if "travel_time" in col]
        # Create Pandana network
        graph = pdna.Network(
            nodes['x'],
            nodes['y'],
            from_nodes,
            to_nodes,
            edge_weights = links[travel_time_cols],
            twoway=False
        )
        
        self.graph = graph
        logger.info(f"\t Created Pandana graph with {len(nodes)} nodes and {len(links)} edges")

    def save(self, path: str):
        """
        Save the NetworkProcessor object to a file.
        
        Args:
        path (str): The file path where the NetworkProcessor object will be saved
        """
        logger.info(f"\t Saving NetworkProcessor to {path}")

        data_path = os.path.join(path, "network_processor.pkl")
        data = dict(
            network_file = self.network_file,
            congestion_file = self.congestion_file,
            graph_type = self.graph_type,
            network = self.network,
            congestion_data = self.congestion_data,
            congestion_aware = self.congestion_aware
        )
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"\t Saved NetworkProcessor data to {data_path}")
        # if graph type is pandana
        if self.graph_type == "pandana" and self.graph is not None:
            graph_path = os.path.join(path, "pandana_graph.h5")
            self.graph.save_hdf5(graph_path)
        # if it is igraph
        elif self.graph_type == "igraph" and self.graph is not None:
            graph_path = os.path.join(path, "igraph_graph.graphml")
            self.graph.write_graphml(graph_path)
        # if it is networkx
        elif self.graph_type == "networkx" and self.graph is not None:
            import networkx as nx
            graph_path = os.path.join(path, "networkx_graph.gpickle")
            nx.write_gpickle(self.graph, graph_path)

        logger.info(f"\t Saved {self.graph_type} graph to {graph_path}")
    
    @staticmethod
    def load(path: str):
        """
        Load the NetworkProcessor object from a file.
        
        Args:
        path (str): The file path from where the NetworkProcessor object will be loaded
        """
        logger.info(f"\t Loading NetworkProcessor from {path}")

        data_path = os.path.join(path, "network_processor.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        net_processor = NetworkProcessor(
            network_file = data['network_file'],
            congestion_file = data['congestion_file'],
            graph_type = data['graph_type'],
            network = data['network']
        )
        
        net_processor.congestion_data = data['congestion_data']
        net_processor.congestion_aware = data['congestion_aware']

        logger.info(f"\t Loaded NetworkProcessor data from {data_path}")
        # if graph type is pandana, load the graph too
        if net_processor.graph_type == "pandana":
            graph_path = os.path.join(path, "pandana_graph.h5")
            if os.path.exists(graph_path):
                import pandana as pdna
                net_processor.graph = pdna.Network.from_hdf5(graph_path)
        # if it is igraph
        elif net_processor.graph_type == "igraph":
            graph_path = os.path.join(path, "igraph_graph.graphml")
            if os.path.exists(graph_path):
                import igraph as ig
                net_processor.graph = ig.read(graph_path)
        # if it is networkx
        elif net_processor.graph_type == "networkx":
            graph_path = os.path.join(path, "networkx_graph.gpickle")
            if os.path.exists(graph_path):
                import networkx as nx
                net_processor.graph = nx.read_gpickle(graph_path)
        logger.info(f"\t Loaded {net_processor.graph_type} graph from {graph_path}")
        return net_processor
            


def configure(context):
    context.config("dmc_graph_type", default="pandana")
    context.config("dmc_network_file", 
                default=os.path.join(context.config("data_path"), "dmc", "switzerland_network.xml.gz"))
    context.config("dmc_congestion_file", 
                   default=os.path.join(context.config("data_path"), "dmc", "linkstats.txt.gz"))  
    context.stage("mode_choice.network.road_network")

def execute(context):
    # get network and congestion file paths
    network_file = context.config("dmc_network_file")
    congestion_file = context.config("dmc_congestion_file")
    
    # get the road network
    road_network = context.stage("mode_choice.network.road_network")

    # prepare the network processor
    network_processor = NetworkProcessor(
            network_file=network_file,
            network=road_network,
            congestion_file=congestion_file,
            graph_type=context.config("dmc_graph_type")
        )

    network_processor.build()

    # path to save the processor object
    path_to_save = context.path()
    network_processor.save(path_to_save)
    
    return NetworkProcessor, path_to_save




