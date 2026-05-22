import json
import os
import shutil

from matsim.readers import read_network
from matsim.scenario.network.utils.network_attribute_assigner import NetworkAttributeAssigner
from matsim.scenario.network.utils.capacity_corrector import CapacityCorrector
from matsim.scenario.network.utils.elevation_estimator import ElevationEstimator
from matsim.scenario.network.utils.network_cleaner import networkCleaner
from matsim.scenario.network.utils.speed_corrector import SpeedCorrector
from matsim.scenario.network.utils.traffic_light_matcher import TrafficLightsMatcher

import logging
logger = logging.getLogger("synpp:\t\t NetworkHandler")

class NetworkHandler:
    def __init__(self, context, network_path: str):
        self.network_path = network_path
        self.net = read_network(network_path)
        self.context = context

    def process_network(self):
        self._assign_network_attributes()
        self._assign_elevations_if_requested()
        self._add_traffic_lights_if_requested()
        self._simplify_network_if_requested()
        self._correct_link_capacity_if_requested()
        self._adjust_capacity_outside_border_if_requested()
        self._adjust_uphill_speed_if_requested()
        self._adjust_straightness_speed_if_requested()
        self._route_bike_if_requested()
        self._final_cleaning()
        return self._save_processed_network()

    def _assign_network_attributes(self):
        self.net.links = NetworkAttributeAssigner(self.context, self.net).assign_attributes()

    def _assign_elevations_if_requested(self):
        if not self.context.config("assign_elevations"):
            return
        
        logger.info("Assigning Elevations...")
        df_switzerland = self.context.stage("data.spatial.swiss_border")
        ch_polygon = df_switzerland.buffer(0).iloc[0]
        self.net = ElevationEstimator(
            network=self.net,
            data_path=self.context.config("data_path"),
            polygone=ch_polygon,
        ).run()

    def _add_traffic_lights_if_requested(self):
        if not self.context.config("add_traffic_lights"):
            return

        logger.info("Adding Traffic Lights...")
        traffic_lights_path = self.context.stage("data.osm.traffic_lights")
        detailed_network_path = "%s/detailed_network.csv" % self.context.path()
        self.net.links = TrafficLightsMatcher(self.net).run(traffic_lights_path, detailed_network_path)

    def _simplify_network_if_requested(self):
        if not self.context.config("simplify_network_in_eqasim"):
            return

        logger.info("Simplifying Network...")
        self.net, stats = networkCleaner(self.net).run(
            remove_network_loops=self.context.config("remove_network_loops"),
            remove_replicate_links=self.context.config("remove_replicate_links"),
            remove_nodes_with_no_intersection=self.context.config("remove_nodes_with_no_intersection"),
            correct_speeds=self.context.config("correct_speed"),
            ensure_network_connectivity=self.context.config("ensure_network_connectivity"),
        )

        with open("%s/statistics_of_cleaning_network.json" % self.context.path(), "w") as f:
            json.dump(stats, f, indent=4)

    def _correct_link_capacity_if_requested(self):
        if self.context.config("correct_links_capacity"):
            logger.info("Correcting Link Capacity...")
            self.net.links = CapacityCorrector(self.context, self.net).run()

    def _adjust_capacity_outside_border_if_requested(self):
        should_adjust = (isinstance(self.context.config("osm_file"), list)
                            and self.context.config("border_offset") > 0
                            and (self.context.config("capacity_factor_outside_border") < 1))

        if should_adjust:            
            logger.info("Adjusting Capacity Outside Border...")
            self.net.links = CapacityCorrector(self.context, self.net).reduce_capacity_outside_border()

    def _adjust_uphill_speed_if_requested(self):
        if not self.context.config("adjust_speed_uphill"):
            return
        if not self.context.config("assign_elevations"):
            raise ValueError("To correct speeds of uphill links, elevations must be assigned first.")

        logger.info("Adjusting Uphill Speeds...")
        self.net.links = SpeedCorrector(self.context, self.net).run("uphill")

    def _adjust_straightness_speed_if_requested(self):
        if not self.context.config("adjust_speed_straightness"):
            return
        logger.info("Adjusting Straightness Speeds...")
        self.net.links = SpeedCorrector(self.context, self.net).run("straightness")

    def _route_bike_if_requested(self):
        if self.context.config("route_bike"):
            logger.info("Routing Bike...")
            self.net.links = networkCleaner(self.net).add_bike_to_network()

    def _final_cleaning(self):
        logger.info("Final Cleaning of Network...")
        self.net.links["freespeed"] = self.net.links["freespeed"].fillna(0).clip(lower=15/3.6, upper=135/3.6)
        self.net.links["capacity"] = self.net.links["capacity"].fillna(0).clip(lower=300)
        self.net.links["permlanes"] = self.net.links["permlanes"].fillna(0).clip(lower=1, upper=10)

    def _save_processed_network(self):
        logger.info("Saving Processed Network...")
        uncleaned_network_path = self.network_path.replace("converted_network", "converted_network_uncleaned")
        shutil.move(self.network_path, uncleaned_network_path)
        self.net.save(self.network_path)

        assert os.path.exists(self.network_path)
        return self.network_path