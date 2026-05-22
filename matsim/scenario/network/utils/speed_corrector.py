import numpy as np
from matsim.readers import Network
import shapely.vectorized
import geopandas as gpd
import shapely.geometry as shp
import logging
from matsim.scenario.network.utils.network_attribute_assigner import NetworkAttributeAssigner

logger = logging.getLogger("synpp")

class SpeedCorrector:
    """
    Class to correct the speed of the links in the network.
    
    Parameters:
        network (Network): The network object containing the links.
    """
    
    def __init__(self, context, network:Network):
        self.context = context
        self.network = network        

    def assign_slopes(self):
        """
        This method calculates the elevation gain for each link in the network based on the elevations of its from_node and to_node.
        It adds a new column 'elevation_gain' to the links DataFrame.
        """
        nodes_elevation = self.network.nodes[['node_id', 'z']]
        links = (self.network.links[["link_id", "length", "from_node", "to_node"]]
                    .merge(nodes_elevation, left_on='from_node', right_on='node_id', how="left")
                    .merge(nodes_elevation, left_on='to_node', right_on='node_id', suffixes=('_from_node', '_to_node'), how="left")
                        )
        no_elevation = ((links['z_from_node'].isna()) | (links['z_to_node'].isna()) | 
                        (links['z_from_node']==0) | (links['z_to_node']==0) | (links['length']==0))
        
        links['slope'] = 0.0
        links.loc[~no_elevation, "slope"] = ((links.loc[~no_elevation,'z_to_node'] - links.loc[~no_elevation,'z_from_node']) 
                                             / links.loc[~no_elevation,'length']) * 100
        return links[['link_id', 'slope']]
    
    def run_uphill_based_correction(self):
        """
        This method corrects the free speed of links in the network based on their gradient.
        """
        # Merge the slopes with the network links
        links = self.network.links.copy()
        slopes = self.assign_slopes()
        links = links.merge(slopes, on="link_id", how="left")

        # Apply speed correction to uphill car links
        car_links = self.network.links.modes.str.split(',').map(lambda x: 'car' in x)
        max_gradient_threshold = self.context.config("max_gradient_threshold")
        speed_factor_uphill = self.context.config("speed_factor_uphill")

        def correct_speed(row):
            gradient = abs(row['slope'] / 100) # convert percentage to decimal
            if gradient > max_gradient_threshold:
                if row['slope'] > 0:
                    return row['freespeed'] * speed_factor_uphill
                else:
                    return row['freespeed'] * ((1+2*speed_factor_uphill)/3) # less reduction for downhill
            else:
                return row['freespeed']

        links.loc[car_links, 'freespeed'] = links[car_links].apply(correct_speed, axis=1)
        # drop the slope column after correction
        links = links.drop(columns=["slope"])
        return links
    
    def run(self, correction_type="municipality_type"):
        """
        This method runs the speed correction process based on the specified correction type.
        Currently, it supports 'municipality_type', 'uphill', and 'straightness' correction types.
        
        Parameters:
            correction_type (str): The type of correction to apply. Default is 'municipality_type'.
        """
        assert correction_type in ["municipality_type","uphill","outside_border","motorway","straightness"], f"Unsupported correction type: {correction_type}"

        if correction_type == "municipality_type":
            raise ValueError("The municipality_type based speed correction has been deprecated. Please use the SpeedFactorProvider to assign speed factors based on municipality types.")
        elif correction_type == "uphill":
            return self.run_uphill_based_correction()
        elif correction_type == "outside_border":           
            raise ValueError("The outside_border based speed correction has been deprecated. Please use the SpeedFactorProvider to assign speed factors based on border proximity.")
        elif correction_type=="motorway":
            raise ValueError("The motorway based speed correction has been deprecated. Please use the SpeedFactorProvider to assign speed factors based on road types.")
        elif correction_type=="straightness":
            return self.run_straightness_based_correction()
        else:
            raise ValueError(f"Unsupported correction type: {correction_type}")

    def run_straightness_based_correction(self):
        """
        This method corrects the free speed of links in the network based on their straightness (length ratio).
        Links that are less straight (i.e., have a higher length ratio) will have their free speed reduced more.
        """
        
        # Merge the length ratios with the network links
        links = self.network.links.copy()
        length_ratios = self.length_ratio()
        links = links.merge(length_ratios, on="link_id", how="left")
        
        # speed correction based on length ratio
        ration_1 = (links['length_ratio'] < 0.75) & (links["length"] < 500) # only correct links longer than 100m to avoid correcting short links that are not necessarily straight
        ration_2 = (links['length_ratio'] < 0.60) & (links["length"] < 500)
        ration_3 = (links['length_ratio'] < 0.40) & (links["length"] < 500)
        ration_4 = (links['length_ratio'] < 0.20) & (links["length"] < 500)

        # TODO: these thresholds and speed limits should be configurable, and better chosen based on real data
        links.loc[ration_1, 'freespeed'] = np.minimum(links.loc[ration_1, 'freespeed'], round(100/3.6,2))
        links.loc[ration_2, 'freespeed'] = np.minimum(links.loc[ration_2, 'freespeed'], round(80/3.6,2))
        links.loc[ration_3, 'freespeed'] = np.minimum(links.loc[ration_3, 'freespeed'], round(50/3.6,2))
        links.loc[ration_4, 'freespeed'] = np.minimum(links.loc[ration_4, 'freespeed'], round(35/3.6,2))

        # drop the length_ratio column after correction
        links = links.drop(columns=["length_ratio"])
        return links
    
    def length_ratio(self):
        # This is a ratio of the length of the link over the euclidean distance between the from_node and to_node. It can be used to correct the speed of the link by multiplying it with the speed factor.
        nodes_coordinates = self.network.nodes[['node_id', 'x', 'y']]
        links = (self.network.links[["link_id", "length", "from_node", "to_node"]]
                    .merge(nodes_coordinates, left_on='from_node', right_on='node_id', how="left")
                    .merge(nodes_coordinates, left_on='to_node', right_on='node_id', suffixes=('_from_node', '_to_node'), how="left")
                        )
        links['euclidean_distance'] = np.sqrt((links['x_to_node'] - links['x_from_node'])**2 + (links['y_to_node'] - links['y_from_node'])**2)
        links['length_ratio'] = links['euclidean_distance']/links['length']
        return links[['link_id', 'length_ratio']]
