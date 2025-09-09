import json
import requests
import logging
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from io import StringIO
from shapely import vectorized
import time
from typing import List, Tuple

os.chdir(os.path.join("..","..","..",".."))
from matsim.readers import Network, read_network

# Assuming 'Network' is a custom class with a 'nodes' attribute (pandas.DataFrame)
# from your_project import Network 

class ElevationEstimator:
    """
    Estimates elevation for network nodes using cached CSV data or the swisstopo API.
    
    If elevations exist in the directory, loads them and assigns to nodes by nearest location.
    If the nearest cached elevation is >2 meters away, fetches from swisstopo.
    If no elevation file exists, fetches all elevations from swisstopo.

    """
    logger = logging.getLogger("ElevationEstimator")

    # --- Constants ---
    DISTANCE_THRESHOLD = 3    # Max distance in meters to use cached elevation
    API_BATCH_SIZE  = 64      # Number of coordinates per API batch request
    API_MAX_RETRIES = 3       # Max retries for one API request  
    API_RETRY_DELAY = 0.2     # Seconds to wait between retries
    API_DELAY       = 0.1     # Seconds to wait between API requests
    SWISSTOPO_HEIGHT_URL  = "https://api3.geo.admin.ch/rest/services/height"
    SWISSTOPO_PROFILE_URL = "https://api3.geo.admin.ch/rest/services/profile.csv"

    def __init__(self, network=None, data_path: str = None, coordinates:List[Tuple[float, float]] = None, polygone = None):
        # you can provide either the network, or a list of coordinates
        # when polygone is provided, only the nodes within the polygone will be processed
        self.network = network
        self.data_path = data_path
        self.coordinates = coordinates
        self.polygone = polygone

        self._build_nodes()

        self.elevation_file = os.path.join(data_path, "osm", "elevations.csv")
        self.elevations_df = None        
        self.session = requests.Session()

    def _build_nodes(self):
        if self.network is None and self.coordinates is None:
            raise ValueError("Either network or coordinates must be provided.")
        if self.network is not None and self.coordinates is not None:
            raise ValueError("Provide either network or coordinates, not both.")

        if self.network is not None:
            self.nodes = self.network.nodes
        if self.coordinates is not None:
            self.nodes = pd.DataFrame(self.coordinates, columns=["x", "y"])
        
        self.nodes["within_polygone"] = True
        if self.polygone is not None:
            self.nodes["within_polygone"] = vectorized.contains(self.polygone, self.nodes["x"].values, self.nodes["y"].values)

    def run(self):
        nodes = self.nodes
        nodes["z"] = np.nan  # Initialize elevation column
        
        nodes_within_polygone = nodes[nodes["within_polygone"]].copy() # copy to avoid the warnings later
        if not nodes_within_polygone.empty:        
            nodes_within_polygone = self.run_on_nodes(nodes_within_polygone)
            nodes.update(nodes_within_polygone)
        
        nodes["z"] = nodes["z"].fillna(0) # fill all nans by 0, these nans are coming from points outside the polygone (or switzerland for the api)
        self.nodes = nodes.copy()

        nodes = nodes.drop(columns=["within_polygone"])
        if self.network is not None:
            self.network.nodes = nodes
            return self.network
        if self.coordinates is not None:
            return nodes
        
    def run_on_nodes(self, nodes):
        if not isinstance(nodes, pd.DataFrame) or nodes.empty or "x" not in nodes or "y" not in nodes:
            raise ValueError("nodes must be a non-empty DataFrame with 'x' and 'y' columns.")
                
        """Main method to run elevation estimation."""
        if os.path.exists(self.elevation_file):
            self.logger.info("Elevation file found. Loading and assigning elevations.")
            nodes = self._assign_from_cache_and_fetch_missing(nodes)
        else:
            self.logger.info("Elevation file not found. Fetching all elevations from swisstopo.")
            nodes = self._fetch_all_elevations(nodes)

        # whenever I request elevations, I store the new dataframe, because maybe it containes more locations, hence enriching the database with more usage
        if self.elevations_df is not None and not self.elevations_df.empty:
            self._save_elevations_to_csv()
        
        nodes = self.fill_nans(nodes)  # Fill any remaining NaNs after initial assignment
        return nodes

    def _assign_from_cache_and_fetch_missing(self, nodes):
        """
        Loads elevations from cache, assigns them, and fetches any that are missing or too far away.        
        """
        self.elevations_df = pd.read_csv(self.elevation_file)        
        
        if self.elevations_df.empty:
            return self._fetch_all_elevations(nodes)            

        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(self.elevations_df[["x", "y"]].values)
        dists, idxs = tree.query(nodes[["x", "y"]].values, k=1)

        # Vectorized assignment for nodes within the distance threshold        
        mask_close = dists <= self.DISTANCE_THRESHOLD
        nodes.loc[mask_close, "z"] = self.elevations_df["z"].values[idxs[mask_close]]

        # Identify nodes that need fetching from the API
        mask_fetch = nodes["z"].isna()
        nodes_to_fetch = nodes[mask_fetch]
        
        if not nodes_to_fetch.empty:
            self.logger.info(f"Found {len(nodes_to_fetch)} nodes missing or too far from cache. Fetching from swisstopo.")
            coords_to_fetch = list(nodes_to_fetch[["x", "y"]].itertuples(index=False, name=None))
            
            # Fetch elevations in batches
            fetched_elevations = self._fetch_elevations_in_batches(coords_to_fetch)
            
            # Update the main nodes DataFrame and the cache DataFrame
            nodes.loc[mask_fetch, "z"] = fetched_elevations

            new_data = nodes_to_fetch[["x", "y"]].copy()
            new_data['z'] = fetched_elevations
            new_data.dropna(inplace=True)
            
            self.update_elevations(new_data, False, False)            
        
        self.logger.info(f"Finished assigning elevations. Total nodes with elevation: {nodes['z'].notna().sum()}/{len(nodes)}")
        return nodes

    def _fetch_all_elevations(self, nodes):
        """Fetches elevations for all network nodes when no cache file exists."""        
        coords = list(nodes[["x", "y"]].itertuples(index=False, name=None))
        
        self.logger.info(f"Fetching elevations for all {len(nodes)} nodes.")
        elevations = self._fetch_elevations_in_batches(coords)
        nodes["z"] = elevations

        # Create the elevations DataFrame from scratch or update it
        self.update_elevations(nodes, from_csv=True)

        self.logger.info(f"Successfully fetched {self.elevations_df.shape[0]} elevations.")
        
        return nodes

    def _fetch_elevations_in_batches(self, coords: list[tuple[float, float]]) -> list[float]:
        """Fetch a list of coordinates in batches with retries."""
        all_elevations = []
        total_batches = -(-len(coords) // self.API_BATCH_SIZE)
        for i in range(0, len(coords), self.API_BATCH_SIZE):
            batch_coords = coords[i:i + self.API_BATCH_SIZE]
            
            for attempt in range(self.API_MAX_RETRIES):
                try:
                    batch_elevs = self._get_swisstopo_profile_elevation(batch_coords)
                    all_elevations.extend(batch_elevs)
                    
                    batch_num = i//self.API_BATCH_SIZE + 1
                    if batch_num % 10 == 0 or batch_num == total_batches:                        
                        self.logger.info(f"Fetched batch {batch_num}/{total_batches} successfully.")
                    break  # Success, exit retry loop

                except (requests.RequestException, ValueError, AssertionError) as e:
                    self.logger.warning(f"Error fetching batch (attempt {attempt + 1}/{self.API_MAX_RETRIES})")
                    if attempt + 1 == self.API_MAX_RETRIES:
                        self.logger.error("Max retries reached. Filling batch with NaNs.")
                        all_elevations.extend([np.nan] * len(batch_coords))
                    else:
                        time.sleep(self.API_RETRY_DELAY)
            time.sleep(self.API_DELAY) 
        
        assert len(all_elevations) == len(coords), "Mismatch in number of elevations fetched."
        return all_elevations
    
    def get_swisstopo_point_elevation(self, x, y):
        """
        Returns elevation at given LV95 coordinates using swisstopo API.
        """
        if pd.isna(x) or pd.isna(y):
            raise ValueError("Coordinates cannot be NaN")
        
        params = {
            "easting": f"{x:.2f}",   # The easting coordinate in LV03 (EPSG:21781) or LV95 (EPSG:2056)
            "northing": f"{y:.2f}",   # The northing coordinate in LV03 (EPSG:21781) or LV95 (EPSG:2056)
            "sr": 2056 # Spatial Reference for LV95
        }
        try:
            response = self.session.get(self.SWISSTOPO_HEIGHT_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data.get("height"))
        except:
            return np.nan
    
    def _get_swisstopo_profile_elevation(self, coords: list[tuple[float, float]]) -> list[float]:
        """
        Fetches elevations for a batch of coordinates using the swisstopo profile API.
        """
        if not coords:
            return []

        geom = {"type": "LineString", "coordinates": coords}
        params = {
            'geom': json.dumps(geom),
            'sr': 2056,
            'nb_points':2,
            'distinct_points': True,
        }

        response = self.session.get(self.SWISSTOPO_PROFILE_URL, params=params)
        response.raise_for_status()
        response_data = pd.read_csv(StringIO(response.text), delimiter=';')

        # Handle cases where the API returns a different number of points
        if len(response_data) != len(coords):
            self.logger.warning(f"API returned {len(response_data)} points for {len(coords)} coordinates. Re-matching...")
            if response_data.empty:
                return [np.nan] * len(coords)

            # Efficiently match returned points to original coordinates using a KDTree
            returned_coords = [(p['Easting'], p['Northing']) for _,p in response_data.iterrows()]
            returned_elevs = response_data["Altitude"].tolist()           
            tree = cKDTree(returned_coords)
            # Find the nearest neighbor in the response for each original point within a small tolerance
            dist, idxs = tree.query(coords, k=1, distance_upper_bound=self.DISTANCE_THRESHOLD) 

            final_elevs = []
            for i,(d, idx) in enumerate(zip(dist, idxs)):
                if np.isfinite(d):  # If a match was found within the tolerance
                    final_elevs.append(returned_elevs[idx])
                else:
                    estimated_point_elevation = self.get_swisstopo_point_elevation(*coords[i])
                    final_elevs.append(estimated_point_elevation)
            return final_elevs
        
        return response_data["Altitude"].tolist()

    def fill_nans(self, nodes):
        """Fills NaN elevations by fetching from swisstopo for each node."""        
        is_nans = nodes["z"].isna()
        if not is_nans.any():            
            return nodes
                
        missing_elevations = nodes[is_nans]
        correct_elevations = nodes[~is_nans]

        tree = cKDTree(correct_elevations[["x", "y"]].values)
        dists, idxs = tree.query(missing_elevations[["x", "y"]].values, k=1)
        
        nodes.loc[is_nans, "z"] = correct_elevations["z"].values[idxs]
        return nodes
                    
    def update_elevations(self, nodes, from_csv, save_to_csv = False):
        if os.path.exists(self.elevation_file) and from_csv:
            self.elevations_df = pd.read_csv(self.elevation_file)
        else:
            if self.elevations_df is None:
                self.elevations_df = pd.DataFrame(columns=["x", "y", "z"])

        self.elevations_df = pd.concat([self.elevations_df, nodes[["x", "y", "z"]].dropna()]).drop_duplicates(subset=["x", "y"], keep="last")

        if save_to_csv:
            self._save_elevations_to_csv()

    def _save_elevations_to_csv(self):
        """Saves the consolidated node elevations to a CSV file."""
        self.elevations_df.to_csv(self.elevation_file, index=False)
        self.logger.info(f"Elevations saved to {self.elevation_file}")




# if __name__ == "__main__":
#     network_path = "Z:\ch-zh-synpop\output0p1\queue_lastVersionEqasim\switzerland_network.xml.gz"    
#     network = read_network(network_path)
#     estimator = ElevationEstimator(network, "")
#     updated_network = estimator.run()
#     #print(updated_network.nodes.head())
    
    
    
    
    
    
    
    
    
    
    
    
    