import json
import os
import shutil
import pandas as pd
from matsim.readers import read_network
from matsim.scenario.network.utils.network_attribute_assigner import NetworkAttributeAssigner
from matsim.scenario.network.utils.capacity_corrector import CapacityCorrector
from matsim.scenario.network.utils.elevation_estimator import ElevationEstimator
from matsim.scenario.network.utils.network_cleaner import networkCleaner
from matsim.scenario.network.utils.speed_corrector import SpeedCorrector
from matsim.scenario.network.utils.traffic_light_matcher import TrafficLightsMatcher
from matsim.scenario.network.utils.link_geometry import (
    GEOMETRY_COLUMN, attach_link_geometries, write_link_geometries,
)

import logging
logger = logging.getLogger("synpp:\t\t NetworkHandler")

class NetworkHandler:
    def __init__(self, context, network_path: str, detailed_network_path:str):
        self.network_path = network_path
        self.detailed_network_path = detailed_network_path
        self.net = read_network(network_path, context=context)
        self.context = context

    def process_network(self, save_as_pickle:bool = False, network_pickle: str=None):
        self._assign_network_attributes()
        self._assign_elevations_if_requested()
        self._add_traffic_lights_if_requested()
        self._add_tolls_if_requested()
        if os.path.exists(self.detailed_network_path):
            logger.info("Loading detailed geometries before network cleaning...")
            attach_link_geometries(self.net, self.detailed_network_path)
        self._simplify_network_if_requested()
        self._correct_link_capacity_if_requested()
        self._adjust_capacity_outside_border_if_requested()
        self._adjust_uphill_speed_if_requested()
        self._adjust_straightness_speed_if_requested()
        self._adjust_mountain_links_speed_if_requested()
        self._route_bike_if_requested()
        self._final_cleaning()
        return self._save_processed_network(save_as_pickle, network_pickle)

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
        self.net.links = TrafficLightsMatcher(self.net).run(traffic_lights_path, self.detailed_network_path)

    def _add_tolls_if_requested(self):
        if not self.context.config("include_tolls"):
            return
        
        logger.info("Adding Tolls...")
        links = self.net.links.copy()

        tolls_links = self.context.stage("data.tolls.osm_links")
        has_tolls_func = lambda x: x.get("osm:way:id", None) in tolls_links if isinstance(x, dict) else False
        has_tolls = links["attributes"].apply(has_tolls_func)

        price_per_km = max(self.context.config("average_tolls_prices_per_km"), 0.0)
        def add_toll_func(row):
            l = max(row['length'], 1.0) / 1000.0 #convert m to km
            x = dict(row['attributes'])
            x["toll"] = price_per_km * l
            return x

        links.loc[has_tolls, "attributes"] = links.loc[has_tolls, ['length','attributes']].apply(add_toll_func, axis=1)
        self.net.links = links


    def _simplify_network_if_requested(self):
        simplify = self.context.config("simplify_network_in_eqasim")
        filter_roads = self.context.config("only_important_roads_outside_border")
        if not simplify and not filter_roads:
            return

        logger.info("Simplifying Network...")
        self.net, stats = networkCleaner(self.net, self.context).run(
            remove_network_loops=simplify and self.context.config("remove_network_loops"),
            remove_replicate_links=simplify and self.context.config("remove_replicate_links"),
            remove_nodes_with_no_intersection=simplify and self.context.config("remove_nodes_with_no_intersection"),
            correct_speeds=simplify and self.context.config("correct_speed"),
            ensure_network_connectivity=simplify and self.context.config("ensure_network_connectivity"),
            only_important_roads_outside_border=filter_roads,
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
        if not self.context.config("correct_speed_uphill"):
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

    def _adjust_mountain_links_speed_if_requested(self):
        if not self.context.config("adjust_speed_mountain_links"):
            return
        logger.info("Adjusting Mountain Links Speeds...")
        self.net.links = SpeedCorrector(self.context, self.net).run("mountain_links")

    def _route_bike_if_requested(self):
        if self.context.config("route_bike"):
            logger.info("Routing Bike...")
            self.net.links = networkCleaner(self.net).add_bike_to_network()

    def _final_cleaning(self):
        logger.info("Final Cleaning of Network...")
        self.net.links["freespeed"] = self.net.links["freespeed"].fillna(0).clip(lower=15/3.6, upper=135/3.6)
        self.net.links["capacity"] = self.net.links["capacity"].fillna(0).clip(lower=300)
        self.net.links["permlanes"] = self.net.links["permlanes"].fillna(0).clip(lower=1, upper=10)
        # Manual upper bounds take precedence over generic minimums and corrections.
        if self.context.config("correct_capacities_from_manual_list"):
            self.net.links = CapacityCorrector(self.context, self.net).correct_from_manual_list()

    def _save_processed_network(self, save_as_pickle: bool = False, network_pickle: str = None):
        logger.info("Saving Processed Network...")
        if GEOMETRY_COLUMN in self.net.links.columns:
            write_link_geometries(self.net, self.detailed_network_path)
            self.net.links = self.net.links.drop(columns=[GEOMETRY_COLUMN])
        # uncleaned_network_path = self.network_path.replace("converted_network", "converted_network_uncleaned")
        # shutil.move(self.network_path, uncleaned_network_path)
        new_network_path = "%s/converted_network.xml.gz" % self.context.path()
        self.net.save(new_network_path)
        assert os.path.exists(new_network_path)

        if save_as_pickle and network_pickle is not None:
            logger.info("Saving Processed Network as Pickle...")
            pd.to_pickle(self.net, network_pickle)

        return new_network_path
