import numpy as np
from matsim.readers import Network
import pyproj
import logging
from calibration.network_freespeed.mountains_links_speeds import SPEED_SEGMENTS, ROUTING_PENALTY_M_PER_S, SPEED_TOLERANCE_KMH, MIN_FREESPEED_M_S

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
        Currently, it supports 'municipality_type', 'uphill', 'outside_border', 'motorway', 'straightness', and 'mountain_links' correction types.
        
        Parameters:
            correction_type (str): The type of correction to apply. Default is 'municipality_type'.
        """
        assert correction_type in ["municipality_type","uphill","outside_border","motorway","straightness","mountain_links"], f"Unsupported correction type: {correction_type}"

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
        elif correction_type=="mountain_links":
            return self.run_mountain_links_correction()
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


    ###################################### Mountain links correction ######################################
    
    def run_mountain_links_correction(self):
        if not hasattr(self, "_G"):
            self._G = self.network.as_igraph(only_car_links=True)
            self._precompute_igraph_data()
        if not hasattr(self, "_coord_transformer"):
            self._coord_transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    
        links = self.network.links.copy()
        car_links_selector = links["modes"].str.split(",").map(lambda x: "car" in x)
        links_car = links[car_links_selector].reset_index(drop=True)
        freespeed = links_car["freespeed"].values.copy()
        links_changed = set()
    
        for segment in SPEED_SEGMENTS:
            freespeed, links_changed = self.adjust_speeds_of_segment(links_car, freespeed, links_changed, segment)
    
        links.loc[car_links_selector, "freespeed"] = freespeed
        return links
    
    
    def _precompute_igraph_data(self):
        """
        Precompute and cache all edge attributes and routing weights from the igraph
        graph so that shortest-path calls don't recompute anything per-call.
        """
        g = self._G
    
        # Cache node coordinates for nearest-node lookup
        self._node_ids = np.array(g.vs["node_id"])
        self._nodes_coords = np.column_stack([g.vs["x"], g.vs["y"]])
    
        # Cache edge attributes as arrays (index = igraph edge index)
        travel_times = np.array(g.es["travel_time"], dtype=float)
        lengths = np.array(g.es["length"], dtype=float)
        speed_factors = np.array(g.es["speed_factor"], dtype=float)
        self._edge_travel_times = travel_times
        self._edge_lengths = lengths
        self._edge_speed_factors = speed_factors
        self._edge_link_ids = np.array(g.es["link_id"])
    
        # Precompute combined routing weight once
        self._routing_weights = travel_times + ROUTING_PENALTY_M_PER_S * lengths
    
        # Map link_id -> edge index for O(1) lookups during path extraction
        self._link_id_to_edge_idx = {lid: i for i, lid in enumerate(self._edge_link_ids)}
    
    
    def adjust_speeds_of_segment(self, links, freespeed, links_changed, segment):
        route_info = self.route_segment(segment)
    
        for link_ids, total_travel_time, total_distance in route_info:
            if len(link_ids) == 0 or total_travel_time == 0 or total_distance == 0:
                continue
    
            computed_average_speed = (total_distance / total_travel_time) * 3.6
            target_speed = segment.speed
    
            if computed_average_speed > target_speed + SPEED_TOLERANCE_KMH:
                speed_factor = target_speed / computed_average_speed
                logger.info(
                    f"\tAdjusting speeds of links in segment "
                    f"({segment.origin_x}, {segment.origin_y}) -> "
                    f"({segment.destination_x}, {segment.destination_y}) "
                    f"from {computed_average_speed:.2f} km/h to {target_speed} km/h "
                    f"with a speed factor of {speed_factor:.2f}"
                )
                not_yet_changed = np.array([lid for lid in link_ids if lid not in links_changed])
                if len(not_yet_changed) == 0:
                    continue
                sel = links["link_id"].isin(not_yet_changed).to_numpy()
                freespeed[sel] = np.maximum(freespeed[sel] * speed_factor, MIN_FREESPEED_M_S)
                links_changed.update(link_ids)
    
        return freespeed, links_changed
    
    
    def route_segment(self, segment):        
        origin = self._coord_transformer.transform(segment.origin_y, segment.origin_x)
        destination = self._coord_transformer.transform(segment.destination_y, segment.destination_x)
    
        origin_node_id = self.get_nearest_node(origin)
        destination_node_id = self.get_nearest_node(destination)
    
        # igraph uses integer vertex indices, not node_id labels
        origin_idx = self._G.vs.find(node_id=origin_node_id).index
        destination_idx = self._G.vs.find(node_id=destination_node_id).index
    
        out = []
        for src, tgt in [(origin_idx, destination_idx), (destination_idx, origin_idx)]:
            # get_shortest_paths returns a list of vertex-index lists (one per target)
            paths = self._G.get_shortest_paths(
                src, to=tgt,
                weights=self._routing_weights,
                output="epath"  # return edge indices directly — no need to convert vertex pairs
            )
            edge_indices = paths[0] if paths else []
    
            link_ids = []
            total_travel_time = 0.0
            total_distance = 0.0
            for eidx in edge_indices:
                link_ids.append(self._edge_link_ids[eidx])
                total_travel_time += (self._edge_travel_times[eidx] / self._edge_speed_factors[eidx]) # Important, use speedfactors so that we do not double penalize them
                total_distance += self._edge_lengths[eidx]
    
            out.append((link_ids, total_travel_time, total_distance))
    
        return out
    
    
    def get_nearest_node(self, coord):
        """Vectorized nearest-node lookup. Cache is populated in _precompute_igraph_data."""
        dists = np.sqrt(
            (self._nodes_coords[:, 0] - coord[0]) ** 2 +
            (self._nodes_coords[:, 1] - coord[1]) ** 2
        )
        return self._node_ids[np.argmin(dists)]