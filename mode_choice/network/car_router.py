"""
Trip routing engine using Dijkstra's algorithm with time-dependent travel times.
Handles individual trip routing and batch processing.
"""

import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from .network_processor import NetworkProcessor

logger = logging.getLogger(__name__)


class CarTripRouter:
    """
    Routes trips through a network considering time-dependent travel times.
    """
    
    def __init__(self, network_processor: NetworkProcessor):
        """
        Initialize the trip router.
        
        Args:
            network_processor: Instance of NetworkProcessor with loaded network
        """
        self.network_processor = network_processor
        self.graph = None
        self.congestion_aware = None
        self.graph_type = network_processor.graph_type
        self.route_node_to_node = None
        self.route_batch_trips = None        
        
    def build(self):
        """
        Build the routing graph and prepare for routing.
        """
        logger.info("Building routing graph ...")
        if self.network_processor.graph is None:
            self.network_processor.build()
        self.graph = self.network_processor.graph

        # build spatial index for nodes
        logger.info("Building spatial index for nodes ...")
        net = self.network_processor.network
        node_coords = np.column_stack((net.nodes['x'], net.nodes['y']))
        self._node_kdtree = cKDTree(node_coords)
        self._node_id_lookup = np.array(net.nodes['node_id'])

        # congestion aware
        self.congestion_aware = self.network_processor.congestion_aware

        # design routing function based on graph type
        if self.graph_type == 'igraph':
            self.route_node_to_node = self.route_node_to_node_igraph
            self.route_batch_trips = self.route_batch_trips_nx_or_igraph
        elif self.graph_type == 'networkx':
            self.route_node_to_node = self.route_node_to_node_networkx
            self.route_batch_trips = self.route_batch_trips_nx_or_igraph
        elif self.graph_type == 'pandana':
            self.route_node_to_node = self.route_node_to_node_pandana 
            self.route_batch_trips = self.route_batch_trips_pandana 
        else:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")
        
        logger.info("CarTripRouter is ready for routing.")

    def nearest_node(self, x: Union[List[float],float], y: Union[List[float],float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the nearest network node to the given coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Distance to the nearest node, index of the nearest node, and node IDs
        """
        query_point = np.array([x, y]).T
        distances, indices = self._node_kdtree.query(query_point, k=1)
        return distances, indices, self._node_id_lookup[indices]

    def route_node_to_node_igraph(self, 
                                origin_node_id:str, 
                                destination_node_id:str, 
                                origin_node_idx: Optional[int] = None, 
                                destination_node_idx: Optional[int] = None, 
                                weight: str = None) -> Optional[float]:
        
        total_travel_time = self.graph.shortest_paths_dijkstra(
                                                    source=origin_node_id,
                                                    target=destination_node_id,
                                                    weights=self.graph.es[weight]
                                                    )[0][0]            
        return total_travel_time
        
    def route_node_to_node_networkx(self, 
                                    origin_node_id:str, 
                                    destination_node_id:str, 
                                    origin_node_idx: Optional[int] = None, 
                                    destination_node_idx: Optional[int] = None, 
                                    weight: str = None) -> Optional[float]:
            total_travel_time = nx.shortest_path_length(
                                    self.graph, 
                                    source=origin_node_id, 
                                    target=destination_node_id, 
                                    weight=weight
                                    )
            return total_travel_time
        
    def route_node_to_node_pandana(self, 
                           origin_node_id:str, 
                           destination_node_id:str, 
                           origin_node_idx: Optional[int] = None, 
                           destination_node_idx: Optional[int] = None, 
                           weight: str = None) -> Optional[float]:
            total_travel_time = self.graph.shortest_path_length(origin_node_idx, destination_node_idx, imp_name=weight)
            return total_travel_time

    def route_one_trip(self, origin_x:float, origin_y:float, destination_x:float, destination_y:float, departure_time:int, congestion:bool=True)-> Optional[Dict]:
        """
        Route a single trip from origin to destination.
        
        Args:
            origin_x: Origin x coordinate
            origin_y: Origin y coordinate
            destination_x: Destination x coordinate
            destination_y: Destination y coordinate
            departure_time: Departure time as seconds from midnight
            congestion: Whether to consider congestion data

        Returns:
            Dictionary with walking access distance, walking egress distance, and total travel time
        """
        if self.graph is None:
            self.prepare_routing()
        
        # Find nearest origin and destination nodes
        access_euc_distance , origin_node_idx, origin_node_id = self.nearest_node(origin_x, origin_y)
        egress_euc_distance , destination_node_idx, destination_node_id = self.nearest_node(destination_x, destination_y)

        # departure hour
        departure_hour = int(departure_time // 3600) % 24

        # travel time attribute
        travel_time_attr = (f"travel_time_{departure_hour}_{departure_hour+1}_avg" 
                            if (self.congestion_aware and congestion) else "travel_time")

        # find the shortest path
        total_travel_time = self.route_node_to_node(origin_node_id, destination_node_id, 
                                                    origin_node_idx, destination_node_idx,
                                                    travel_time_attr)
        
        # return the total travel time
        return dict(access_euc_distance=access_euc_distance, 
                    egress_euc_distance=egress_euc_distance,
                    total_travel_time=total_travel_time)

    def route_batch_trips_nx_or_igraph(self, trips: pd.DataFrame, departure_hour: int, congestion: bool=True) -> pd.DataFrame:
        """
        Route a batch of trips. 
        Args:
            trips: DataFrame with columns ['origin_x', 'origin_y', 'destination_x', 'destination_y']
            departure_hour: Departure hour as integer (0-23)
            congestion: Whether to consider congestion data
        """
        if self.graph is None:
            self.prepare_routing()
        
        # Precompute nearest nodes for all trips        
        access_euc_distances, origin_idx, origin_nodes = self.nearest_node(trips['origin_x'], trips['origin_y'])
        egress_euc_distances, dest_idx, dest_nodes = self.nearest_node(trips['destination_x'], trips['destination_y'])

        travel_time_attr = (f"travel_time_{departure_hour}_{departure_hour+1}_avg" 
                            if (self.congestion_aware and congestion) else "travel_time")

        # Batch compute shortest paths                  
        results = [
            self.route_node_to_node(origin_node_id, destination_node_id, 
                                    origin_node_idx, destination_node_idx,
                                    travel_time_attr)
            for (origin_node_id, destination_node_id, origin_node_idx, destination_node_idx)
            in zip(origin_nodes, dest_nodes, origin_idx, dest_idx)
        ]
                    
        # Compile results into DataFrame
        df = pd.DataFrame({
            'access_euc_distance': np.atleast_1d(access_euc_distances),
            'egress_euc_distance': np.atleast_1d(egress_euc_distances),
            'total_travel_time': pd.Series(results, index=trips.index)
        }, index=trips.index)
        return df

    def route_batch_trips_pandana(self, trips: pd.DataFrame, departure_hour: int, congestion: bool=True) -> pd.DataFrame:
        """
        Route a batch of trips efficiently. 
        Args:
            trips: DataFrame with columns ['origin_x', 'origin_y', 'destination_x', 'destination_y']
            departure_hour: Departure hour as integer (0-23)
            congestion: Whether to consider congestion data
        """
        if self.graph is None:
            self.prepare_routing()
        
        # Precompute nearest nodes for all trips        
        access_euc_distances, origin_idx, origin_nodes = self.nearest_node(trips['origin_x'], trips['origin_y'])
        egress_euc_distances, dest_idx, dest_nodes = self.nearest_node(trips['destination_x'], trips['destination_y'])

        travel_time_attr = (f"travel_time_{departure_hour}_{departure_hour+1}_avg" 
                            if (self.congestion_aware and congestion) else "travel_time")        
        # Batch compute shortest paths                  
        tt = self.graph.shortest_path_lengths(
            origin_idx,
            dest_idx,
            imp_name=travel_time_attr
        )
                    
        # Compile results into DataFrame
        df = pd.DataFrame({
            'access_euc_distance': np.atleast_1d(access_euc_distances),
            'egress_euc_distance': np.atleast_1d(egress_euc_distances),
            'total_travel_time': pd.Series(tt, index=trips.index)
        }, index=trips.index)
        return df

    def router_trips_dataframe(self, df, congestion: bool=True, batch_size: int=512, departure_hour: int=None) -> pd.DataFrame:
        """
        Route all trips in the provided DataFrame.

        Args:
            df: DataFrame with trip data, including origin and destination coordinates.
            congestion: Whether to consider congestion data (default: True).
            batch_size: Number of trips to route in one batch (default: 512).
            departure_hour: Departure hour as integer (0-23). If None, departure time should be in the DataFrame in seconds after midnight.

        Returns:
            DataFrame with routing results, including access distances and total travel times.
        """
        if self.graph is None:
            logger.warning("Router graph is not built yet. Building now...")
            self.build()
        
        df = df[['person_id', 'trip_index', 
                 'origin_x', 'origin_y', 
                 'destination_x', 'destination_y', 
                 'departure_time']].reset_index(drop=True).copy()
        # Extract departure hours
        if departure_hour is None:
            departure_hours = (df['departure_time'] // 3600).astype(int) % 24
        else:
            departure_hours = pd.Series(departure_hour, index=df.index)

        logger.info("Starting car routing process ...")
        logger.info(f"\t Batch size: {batch_size}")
        logger.info(f"\t Considering congestion: {congestion}")
        logger.info(f"\t The graph type used: {self.graph_type}")        
        
        # Start routing
        self.results = []
        total_trips = len(df)
        routed_count = 0    
        iteration = 0    
        start_time = time.time()
        for hour in range(24):
            hour_trips = df[departure_hours == hour]
            if not hour_trips.empty:
                for start_idx in range(0, len(hour_trips), batch_size):
                    end_idx = start_idx + batch_size
                    batch_trips = hour_trips.iloc[start_idx:end_idx]
                    batch_results = self.route_batch_trips(batch_trips, departure_hour=hour, congestion=congestion)
                    batch_results["person_id"] = batch_trips["person_id"].values
                    batch_results["trip_index"] = batch_trips["trip_index"].values
                    batch_results["departure_time"] = batch_trips["departure_time"].values
                    self.results.append(batch_results)
                    routed_count += len(batch_trips) 
                    iteration += 1
                    if iteration % 10 == 0:                   
                        logger.info(f"\t Routing progress: {routed_count}/{total_trips} trips routed")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\t Completed routing of {routed_count} trips in {elapsed_time:.2f} seconds.")
        # Append results to trips DataFrame
        results_df = pd.concat(self.results, ignore_index=True)   
        del self.results
        return results_df

def configure(context):
    pass

def execute(context):
    return CarTripRouter



