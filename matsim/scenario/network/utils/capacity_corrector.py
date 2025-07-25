import numpy as np
from matsim.readers import Network


class CapacityCorrector:
    """
    Class to correct the capacity of the links in the network.
    
    Parameters:
        network (Network): The network object containing the links.
    """
    
    def __init__(self, network:Network):
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





