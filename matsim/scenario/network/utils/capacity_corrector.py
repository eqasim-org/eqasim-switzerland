import numpy as np
from matsim.readers import Network
import shapely.vectorized
import geopandas as gpd

class CapacityCorrector:
    """
    Class to correct the capacity of the links in the network.
    
    Parameters:
        network (Network): The network object containing the links.
    """
    
    def __init__(self, context = None, network:Network = None):
        self.context = context
        self.links = network.links
        self.nodes = network.nodes

    def _correct_capacity(self, row, sampling_rate, minimum_speed):
        current_capacity = row["capacity"]
        length = row["length"]
        return np.maximum(current_capacity, 3600*minimum_speed/(length*sampling_rate))
    
    
    def run(self):
        """        
        This function here correct the capacity of a link based on its length and 
        the sampling rate of teh population. It will only impact small links.
        """
        sampling_rate = self.context.config("input_downsampling")
        minimum_speed = self.context.config("minimum_speed")/3.6
        
        car_links = self.links.modes.str.contains(r"\bcar\b")
        
        self.links.loc[car_links, "capacity"] = \
            self.links.loc[car_links, ["capacity", "length"]].apply(
            lambda x: self._correct_capacity(x, sampling_rate, minimum_speed),
            axis=1)
        return self.links

    def get_centroids(self):
        """
        This method calculates the centroids of all links in the network.
        It merges the link data with node coordinates to compute the midpoints.
        """
        links_centers = (self.links[["link_id", "from_node", "to_node"]]
                        .merge(self.nodes, left_on='from_node', right_on='node_id', how="left")
                        .merge(self.nodes, left_on='to_node', right_on='node_id', suffixes=('_from_node', '_to_node'), how="left")
                        )
        centroids_x = (links_centers.x_from_node + links_centers.x_to_node) / 2
        centroids_y = (links_centers.y_from_node + links_centers.y_to_node) / 2
        geometry    = gpd.points_from_xy(centroids_x, centroids_y, crs="EPSG:2056")
        links_centers = gpd.GeoDataFrame(links_centers[["link_id"]], geometry=geometry, crs="EPSG:2056")
        return links_centers
    
    def get_swiss_border(self):
        """
        This method retrieves the Swiss border geometry from the context.
        """
        border = self.context.stage("data.spatial.swiss_border")
        border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
        return border.geometry.iloc[0]

    def reduce_capacity_outside_border(self):
        # links outside of switzerland
        centroids = self.get_centroids()
        border = self.get_swiss_border()
        outside_mask = ~shapely.vectorized.contains(border, centroids.geometry.x.tolist(), centroids.geometry.y.tolist())
        links_outside_border = set(centroids.loc[outside_mask, "link_id"].unique())
        
        # mask
        car_links = self.links.modes.str.contains(r"\bcar\b")        
        mask = car_links & self.links['link_id'].isin(links_outside_border)

        # apply correction
        factor = self.context.config("capacity_factor_outside_border")
        self.links.loc[mask, "capacity"] *= factor
        
        return self.links


