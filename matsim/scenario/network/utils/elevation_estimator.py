import requests
from matsim.readers import Network
import logging
import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


class ElevationEstimator:
    """
    Estimates elevation for network nodes using cached CSV data or swisstopo API.
    If elevations exist in the directory, loads them and assigns to nodes by nearest location.
    If the nearest cached elevation is >2 meters away, fetches from swisstopo.
    If no elevation file exists, fetches all elevations from swisstopo.
    """
    logger = logging.getLogger("ElevationEstimator")

    def __init__(self, network: Network, data_path: str):
        self.network = network
        self.data_path = data_path
        self.elevation_file = f"{data_path}/elevations.csv"

    def run(self):        
        if os.path.exists(self.elevation_file):
            self.logger.info("Elevation file found. Loading elevations.")
            self.elevations_df = pd.read_csv(self.elevation_file)
            self.assign_elevations_from_file()
        else:
            self.logger.info("Elevation file not found. Fetching elevations from swisstopo.")
            self.assign_elevations_from_swisstopo()
        
        self.save_elevations_to_csv()        
        return self.network
    
    def assign_elevations_from_file(self):
        """
        Assigns elevation to each node from the cached file, or fetches from swisstopo if >2m away.
        Uses KDTree for fast nearest neighbor search.        
        """

        self.network.nodes["z"] = np.nan
        coords = self.elevations_df[["x", "y"]].values
        elevs = self.elevations_df["z"].values
        tree = KDTree(coords)

        num_missing_nodes = 0
        node_coords = self.network.nodes[["x", "y"]].values
        dists, idxs = tree.query(node_coords)

        for i, (dist, idx) in enumerate(zip(dists, idxs)):
            if dist <= 2:
                elevation = elevs[idx]
            else:
                node = self.network.nodes.iloc[i]
                elevation = self.get_swisstopo_elevation(node.x, node.y)
                num_missing_nodes += 1
            self.network.nodes.at[i, "z"] = elevation

        self.elevations_df = pd.concat([self.elevations_df, self.network.nodes[["x", "y", "z"]]]).drop_duplicates(subset=["x", "y"], keep="last")
        self.logger.info(f"Assigned elevations to {len(self.network.nodes)} nodes. "
                        f"Missing nodes fetched from swisstopo: {num_missing_nodes}")

    def assign_elevations_from_swisstopo(self):
        """
        Fetches elevation for all nodes from swisstopo.
        """
        self.network.nodes["z"] = np.nan
        for i, node in self.network.nodes.iterrows():
            try:
                elevation = self.get_swisstopo_elevation(node.x, node.y)
                self.network.nodes.at[i, "z"] = elevation
            except Exception as e:
                self.logger.warning(f"Error fetching elevation for node {node.id}: {e}")
        
        self.elevations_df = self.network.nodes[["x", "y", "z"]]

    def save_elevations_to_csv(self):
        """
        Saves node elevations to CSV, including coordinates for future matching.
        """
        output_file = self.elevation_file        
        self.elevations_df.to_csv(output_file, index=False)
        self.logger.info(f"Elevations saved to {output_file}")

    @staticmethod
    def get_swisstopo_elevation(x, y):
        """
        Returns elevation at given LV95 coordinates using swisstopo API.
        """
        url = "https://api3.geo.admin.ch/rest/services/height"
        params = {
            "easting": f"{x:.2f}",   # The easting coordinate in LV03 (EPSG:21781) or LV95 (EPSG:2056)
            "northing": f"{y:.2f}"   # The northing coordinate in LV03 (EPSG:21781) or LV95 (EPSG:2056)
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("height")

