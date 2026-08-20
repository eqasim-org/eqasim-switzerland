import pandas as pd
import geopandas as gpd
from matsim.scenario.network.utils.speed_factors import SpeedFactorProvider

DEFAULT_PENALTIES = {1 : 0.4000, 2 : 0.1236, 3 :0.0667, 4 :-0.1000, 5 :-0.1000, 11: 0.4000, 12: 0.0179, 13:-0.1000, 
                     14:-0.1000, 15:-0.1000, 21:0.2743, 22:-0.1000, 23:-0.1000, 24:-0.1000, 25:-0.1000}
SPECIFIC_FACTORS = {"ramp":1.1, "trunk":1.3, "normal":1.0}

class RoutingPenaltyProvider:
    def __init__(self, context, links, nodes):
        self.context = context
        self.links = links.copy()
        self.centroids = self.get_centroids(self.links, nodes)
        self.polygone = self.load_polygone()
        self.links_in_polygone = set(self.centroids[self.centroids.geometry.within(self.polygone)].link_id.tolist())
        self.penalties = DEFAULT_PENALTIES

    def process(self):
        if "attributes" not in self.links.columns:
            return self.links

        attrs = self.links["attributes"].tolist()
        link_ids = self.links["link_id"].tolist()
        modes = self.links["modes"].tolist()
        permlanes = self.links["permlanes"].tolist()
        freespeeds = self.links["freespeed"].tolist()

        for attr, link_id, link_modes, lanes, speed in zip(attrs, link_ids, modes, permlanes, freespeeds):
            if isinstance(attr, dict):
                attr["penalty"] = self._get_penalty_from_values(attr, link_id, link_modes, lanes, speed)

        return self.links          

    def _get_penalty_from_values(self, attributes, link_id, modes, number_of_lanes, freespeed):
        base_category = SpeedFactorProvider._get_link_base_category_from_values(
            attributes, modes, number_of_lanes, freespeed
        )
        road_type = self._get_link_type_from_attributes(attributes)
        road_type_factor = SPECIFIC_FACTORS.get(road_type, 1.0)

        if base_category is None:
            return 0.0

        if link_id in self.links_in_polygone:
            penalty = self.penalties.get(base_category + 20, 0.0)
            if penalty >= 0:
                return penalty * road_type_factor

        municipality_type = attributes.get("municipalityType", "outside")
        if municipality_type in ["urbancore", "urban"]:
            return self.penalties.get(base_category + 10, 0.0) * road_type_factor

        return self.penalties.get(base_category, 0.0) * road_type_factor
    
    def read_penalties_from_csv(self, path):
        # csv has these columns: # category;penalty(%)
        df = pd.read_csv(path)
        penalties = dict()
        for _, row in df.iterrows():
            category = row["category"]
            penalty = row["penalty(%)"]            
            penalties[category] = penalty
        
        # make sure all penalties are defined, if not, use the parent's penalty or 0 if not defined
        for cat in range(1, 6):
            if cat not in penalties:
                penalties[cat] = 0.0
            if cat + 10 not in penalties:
                penalties[cat + 10] = penalties.get(cat, 0.0) # urbans, fall back to the non-urban penalty if not defined
            if cat + 20 not in penalties:
                penalties[cat + 20] = -1 # do not consider that case (meaning, will fall back to urban/non-urban penalty)

        self.penalties = penalties
    
    @staticmethod
    def _get_link_type_from_attributes(attributes):
        osm_highway = attributes.get("osm:way:highway", None)
        if osm_highway in ["trunk", "trunk_link"]:
            return "trunk"
        if osm_highway in ["motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"]:
            return "ramp"
        return "normal"

    def load_polygone(self):
        poly_path = self.context.stage("calibration.road_regions.penalty_calibration")
        poly_path = poly_path.split(";")
        
        geometries = []
        for path_i in poly_path:
            gdf = gpd.read_file(path_i)
            geometries.extend(gdf.geometry.tolist())
        unioned = gpd.GeoSeries(geometries).union_all()

        return unioned
        
    @staticmethod
    def get_centroids(links, nodes):
        links_centers = (
            links[["link_id", "from_node", "to_node"]]
            .merge(nodes, left_on="from_node", right_on="node_id", how="left")
            .merge(nodes, left_on="to_node", right_on="node_id", suffixes=("_from_node", "_to_node"), how="left")
        )
        centroids_x = (links_centers.x_from_node + links_centers.x_to_node) / 2
        centroids_y = (links_centers.y_from_node + links_centers.y_to_node) / 2
        geometry = gpd.points_from_xy(centroids_x, centroids_y, crs="EPSG:2056")
        return gpd.GeoDataFrame(links_centers[["link_id"]], geometry=geometry, crs="EPSG:2056")
