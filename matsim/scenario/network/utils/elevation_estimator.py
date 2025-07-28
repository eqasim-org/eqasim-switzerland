import json
import requests
import logging
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from io import StringIO
import time

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

    # --- Constants for better readability and maintenance ---
    DISTANCE_THRESHOLD = 2  # Max distance in meters to use cached elevation
    API_BATCH_SIZE  = 64     # Number of coordinates per API batch request
    API_MAX_RETRIES = 3     # Max retries for API requests  
    API_RETRY_DELAY = 0.2     # Seconds to wait between retries
    API_DELAY       = 0.1     # Seconds to wait between API requests
    SWISSTOPO_HEIGHT_URL = "https://api3.geo.admin.ch/rest/services/height"
    SWISSTOPO_PROFILE_URL = "https://api3.geo.admin.ch/rest/services/profile.csv"

    def __init__(self, network, data_path: str):
        self.network = network
        self.data_path = data_path
        self.elevation_file = os.path.join(data_path, "osm", "elevations.csv")
        self.elevations_df = None
        # Use a requests.Session for connection pooling, improving performance
        self.session = requests.Session()

    def run(self):
        """Main method to run elevation estimation."""
        if os.path.exists(self.elevation_file):
            self.logger.info("Elevation file found. Loading and assigning elevations.")
            self._assign_from_cache_and_fetch_missing()
        else:
            self.logger.info("Elevation file not found. Fetching all elevations from swisstopo.")
            self._fetch_all_elevations()

        if self.elevations_df is not None and not self.elevations_df.empty:
            self._save_elevations_to_csv()
        
        self.fill_nans()  # Fill any remaining NaNs after initial assignment
        return self.network

    def _assign_from_cache_and_fetch_missing(self):
        """
        Loads elevations from cache, assigns them, and fetches any that are missing or too far away.        
        """
        self.elevations_df = pd.read_csv(self.elevation_file)
        nodes = self.network.nodes
        
        if self.elevations_df.empty:
            self._fetch_all_elevations()
            return

        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(self.elevations_df[["x", "y"]].values)
        dists, idxs = tree.query(nodes[["x", "y"]].values, k=1)

        # Vectorized assignment for nodes within the distance threshold
        nodes["z"] = np.nan
        mask_close = dists <= self.DISTANCE_THRESHOLD
        nodes.loc[mask_close, "z"] = self.elevations_df["z"].values[idxs[mask_close]]

        # Identify nodes that need fetching from the API
        mask_fetch = ~mask_close
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
            
            self.elevations_df = pd.concat([self.elevations_df, new_data]).drop_duplicates(subset=["x", "y"], keep="last")

        self.network.nodes = nodes
        self.logger.info(f"Finished assigning elevations. Total nodes with elevation: {nodes['z'].notna().sum()}/{len(nodes)}")

    def _fetch_all_elevations(self):
        """Fetches elevations for all network nodes when no cache file exists."""
        nodes = self.network.nodes
        coords = list(nodes[["x", "y"]].itertuples(index=False, name=None))
        
        self.logger.info(f"Fetching elevations for all {len(nodes)} nodes.")
        elevations = self._fetch_elevations_in_batches(coords)
        nodes["z"] = elevations

        # Create the elevations DataFrame from scratch or update it
        if os.path.exists(self.elevation_file):
            self.elevations_df = pd.read_csv(self.elevation_file)
            self.elevations_df = pd.concat([self.elevations_df, nodes[["x", "y", "z"]].dropna()]).drop_duplicates(subset=["x", "y"], keep="last")
        else:
            self.elevations_df = nodes[["x", "y", "z"]].dropna().drop_duplicates(subset=["x", "y"], keep="last")

        self.logger.info(f"Successfully fetched {self.elevations_df.shape[0]} elevations.")

    def _fetch_elevations_in_batches(self, coords: list[tuple[float, float]]) -> list[float]:
        """Fetch a list of coordinates in batches with retries."""
        all_elevations = []
        total_batches = -(-len(coords) // self.API_BATCH_SIZE)  # Ceiling division for total batches
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

    def fill_nans(self):
        """Fills NaN elevations by fetching from swisstopo for each node."""
        nodes = self.network.nodes
        is_nans = nodes["z"].isna()
        if not is_nans.any():            
            return
                
        missing_elevations = nodes[is_nans]
        correct_elevations = nodes[~is_nans]

        tree = cKDTree(correct_elevations[["x", "y"]].values)
        dists, idxs = tree.query(missing_elevations[["x", "y"]].values, k=1)
        
        nodes.loc[is_nans, "z"] = correct_elevations["z"].values[idxs]
        self.network.nodes = nodes
                    
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
    
    
    
    
    
    
    
    
    
    
    
    
    