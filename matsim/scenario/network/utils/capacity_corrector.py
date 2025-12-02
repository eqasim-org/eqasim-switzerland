import numpy as np
from matsim.readers import Network
import shapely.vectorized

class CapacityCorrector:
    """
    Class to correct the capacity of the links in the network.
    
    Parameters:
        network (Network): The network object containing the links.
    """
    
    def __init__(self, network:Network):
        self.links = network.links
        self.nodes = network.nodes

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


    def reduce_capacity(self, border, factor):
        # nodes within switzerland
        within_ch = shapely.vectorized.contains(border, self.nodes.x.tolist(), self.nodes.y.tolist())        
        nodes_within_ch = set(self.nodes[within_ch].node_id)
        
        # links withing switzerland
        links_within_ch = (self.links.from_node.isin(nodes_within_ch) &
                           self.links.to_node.isin(nodes_within_ch))
        
        car_links = self.links["modes"].str.contains(r'\bcar\b')

        #is_in = links_within_ch&car_links
        is_out = (~links_within_ch)&car_links

        self.links.loc[is_out, "capacity"] *= factor
        
        return self.links


