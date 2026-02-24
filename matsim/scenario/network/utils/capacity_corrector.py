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
        self.links = network.links.copy()
        self.nodes = network.nodes

    def _correct_capacity(self, row, sampling_rate, minimum_speed):
        current_capacity = row["capacity"]
        length = row["length"]
        min_speed = minimum_speed
        if row["freespeed"] > 70/3.6:
            min_speed = minimum_speed * 1.5
            
        return np.maximum(current_capacity, 3600*min_speed/(length*sampling_rate))
    
    
    def correct_capacity_on_motorways(self):
        
        car_links = self.links.modes.str.split(',').map(lambda x: 'car' in x)
        road_type = self.links['attributes'].apply(lambda x: x.get('osm:way:highway') if x is not None else False)
        motorway_links = car_links & road_type.isin(['motorway'])
        trunk_links = car_links & road_type.isin(['trunk'])
        primary_links = car_links & road_type.isin(['primary'])
        
        self.links["permlanes"] = self.links["permlanes"].clip(lower=1) # make sure there is at least one lane

        self.links.loc[motorway_links, "capacity"] = np.clip(self.links.loc[motorway_links, "capacity"], 
                                                             a_min = self.links.loc[motorway_links, "permlanes"] * 1800.0, 
                                                             a_max = self.links.loc[motorway_links, "permlanes"] * 2400.0)

        self.links.loc[trunk_links, "capacity"] = np.clip(self.links.loc[trunk_links, "capacity"], 
                                                        a_min = self.links.loc[trunk_links, "permlanes"] * 1600.0, 
                                                        a_max = self.links.loc[trunk_links, "permlanes"] * 2200.0)

        self.links.loc[primary_links, "capacity"] = np.clip(self.links.loc[primary_links, "capacity"], 
                                                        a_min = self.links.loc[primary_links, "permlanes"] * 1300.0, 
                                                        a_max = self.links.loc[primary_links, "permlanes"] * 1900.0)


    def run(self):
        """        
        This function here correct the capacity of a link based on its length and 
        the sampling rate of the population. It will only impact small links.
        After regorous checking, this function has been modified to correct capacities of important links
        """
        self.correct_capacity_on_motorways()
        
        sampling_rate = self.context.config("input_downsampling")
        minimum_speed = self.context.config("minimum_speed")/3.6
        
        car_links = self.links.modes.str.split(',').map(lambda x: 'car' in x)
        
        self.links.loc[car_links, "capacity"] = \
            self.links.loc[car_links, ["capacity", "length", "freespeed"]].apply(
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


