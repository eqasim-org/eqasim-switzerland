import numpy as np
import logging
import unicodedata
from functools import lru_cache
from calibration.network_capacities import special_places_capacities
from matsim.readers import Network
from shapely import contains_xy
import geopandas as gpd
from matsim.scenario.network.utils.routing_penalty import RoutingPenaltyProvider
from data.osm.clean import read_outside_region
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def normalize_street_name(name):
    """Compare names independently of case, accents, spacing and punctuation."""
    name = name.casefold().translate(str.maketrans({"œ": "oe", "æ": "ae", "ø": "o", "ł": "l"}))
    return "".join(char for char in unicodedata.normalize("NFKD", name) if char.isalnum())


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
                                                             a_min = self.links.loc[motorway_links, "permlanes"] * 1600.0, 
                                                             a_max = self.links.loc[motorway_links, "permlanes"] * 2400.0)

        self.links.loc[trunk_links, "capacity"] = np.clip(self.links.loc[trunk_links, "capacity"], 
                                                        a_min = self.links.loc[trunk_links, "permlanes"] * 1300.0, 
                                                        a_max = self.links.loc[trunk_links, "permlanes"] * 2200.0)

        self.links.loc[primary_links, "capacity"] = np.clip(self.links.loc[primary_links, "capacity"], 
                                                        a_min = self.links.loc[primary_links, "permlanes"] * 1000.0, 
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
        return self.correct_from_manual_list()

    def correct_from_manual_list(self):
        """Apply street-specific upper bounds to all matching links.

        Capacity entries are veh/hour/lane; MATSim capacity is veh/hour/link.
        Speeds are converted from km/hour to m/second. Existing lower values
        remain lower. Multiple matching entries use the most restrictive cap.
        """
        if not self.context.config("correct_capacities_from_manual_list"):
            return self.links

        limits = {}
        display_names = {}
        for segment in special_places_capacities.CAPACITY_SEGMENTS:
            if not segment.street_name:
                continue
            key = normalize_street_name(segment.street_name)
            if not key:
                continue
            cap, speed = limits.get(key, (np.inf, np.inf))
            for value in (segment.maximum_capacity_veh_h_lane, segment.maximum_speed_km_h):
                if value is not None and (not np.isfinite(value) or value < 0):
                    raise ValueError(f"Invalid manual capacity/speed limit for {segment.street_name!r}: {value}")
            if segment.maximum_capacity_veh_h_lane is not None:
                cap = min(cap, segment.maximum_capacity_veh_h_lane)
            if segment.maximum_speed_km_h is not None:
                speed = min(speed, segment.maximum_speed_km_h / 3.6)
            limits[key] = (cap, speed)
            display_names[key] = segment.street_name

        matched = set()

        @lru_cache(maxsize=None)
        def lookup(names):
            cap, speed = np.inf, np.inf
            for name in names.split(";"):
                key = normalize_street_name(name)
                if key in limits:
                    matched.add(key)
                    cap = min(cap, limits[key][0])
                    speed = min(speed, limits[key][1])
            return cap, speed

        def link_limits(attrs):
            if not isinstance(attrs, dict):
                return np.inf, np.inf
            names = ";".join(value for key in ("osm:way:name", "old_street_names")
                             if isinstance(value := attrs.get(key), str))
            return lookup(names)

        values = self.links.attributes.map(link_limits)
        capacity_per_lane = values.map(lambda pair: pair[0])
        speed_limits = values.map(lambda pair: pair[1])
        capacity_mask = np.isfinite(capacity_per_lane)
        speed_mask = np.isfinite(speed_limits)
        self.links.loc[capacity_mask, "capacity"] = np.minimum(
            self.links.loc[capacity_mask, "capacity"],
            capacity_per_lane[capacity_mask] * self.links.loc[capacity_mask, "permlanes"],
        )
        self.links.loc[speed_mask, "freespeed"] = np.minimum(
            self.links.loc[speed_mask, "freespeed"], speed_limits[speed_mask],
        )
        logger.info("Manual street limits: %d capacity matches, %d speed matches across %d streets.",
                    int(capacity_mask.sum()), int(speed_mask.sum()), len(matched))
        unmatched = [display_names[key] for key in limits.keys() - matched]
        if unmatched:
            logger.info("Manual streets not found in this network: %s", ", ".join(sorted(unmatched)))
        return self.links
    
    def get_swiss_border(self):
        """
        This method retrieves the Swiss border geometry from the context.
        """
        border = self.context.stage("data.spatial.swiss_border")
        border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
        return border.geometry.iloc[0]

    @staticmethod
    def get_region_border(context, buffer=100.0):
        """
        This method retrieves the geometry of the area that is simulated
        """
        # Bounding Area
        border = context.stage("data.spatial.swiss_border")
        border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
        
        # Collect all geometries to combine
        geometries = [unary_union(border.geometry)]
        
        # outside CH region
        out_region_file = context.config("cross_border_exclude_shapefiles")
        include_external_population = context.config("include_external_population")
        if out_region_file is not None and include_external_population:
            out_region = read_outside_region(out_region_file)
            geometries.append(unary_union(out_region.geometry))
        
        # Combine border + outside regions into one geometry
        combined = unary_union(geometries)
        
        # Apply buffer just in case geometries are not perfect (I don't want to have holes in the geometry)
        combined = combined.buffer(buffer)
        return combined

    def reduce_capacity_outside_border(self):
        # links outside of switzerland
        centroids = RoutingPenaltyProvider.get_centroids(self.links, self.nodes)
        border = self.get_region_border(self.context)
        outside_mask = ~contains_xy(border, centroids.geometry.x.to_numpy(), centroids.geometry.y.to_numpy())
        links_outside_border = set(centroids.loc[outside_mask, "link_id"].unique())
        
        # mask
        car_links = self.links.modes.str.contains(r"\bcar\b")        
        mask = car_links & self.links['link_id'].isin(links_outside_border)

        # apply correction
        factor = self.context.config("capacity_factor_outside_border")
        self.links.loc[mask, "capacity"] *= factor
        
        return self.links

