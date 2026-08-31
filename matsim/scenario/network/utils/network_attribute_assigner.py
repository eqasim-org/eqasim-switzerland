import geopandas as gpd
from matsim.scenario.network.utils.speed_factors import SpeedFactorProvider
from matsim.scenario.network.utils.routing_penalty import RoutingPenaltyProvider
import logging

logger = logging.getLogger("synpp:\t\t NetworkAttributeAssigner")
class NetworkAttributeAssigner:
    def __init__(self, context, network):
        self.context = context
        self.network = network
        self.links = network.links.copy()        

    def assign_attributes(self):
        logger.info("Assigning Municipality Types...")
        self.links = self.assign_municipality_type()
        self.links = self.assign_zeros_speed_factor()
        logger.info("Assigning Routing Penalties...")
        self.links = RoutingPenaltyProvider(self.context, self.links, self.network.nodes).process()
        logger.info("Assigning Speed Factors...")
        self.links = SpeedFactorProvider(self.context, self.links, self.network.nodes).process()
        return self.links

    def assign_zeros_speed_factor(self):
        if "attributes" not in self.links.columns:
            return self.links

        for attr in self.links["attributes"].tolist():
            if isinstance(attr, dict):
                attr["speedFactor"] = 1.0

        return self.links

    def assign_municipality_type(self):
        regions = self._get_municipality_regions()
        links_types = self._assign_municipality_types(regions)
        links = self.links
        municipality_by_link = links_types.set_index("link_id")["municipality_type"]
        links["municipality_type"] = links["link_id"].map(municipality_by_link).fillna("outside")

        # Update dictionaries in place to avoid expensive row-wise DataFrame apply.
        for attr, municipality_type in zip(links["attributes"].tolist(), links["municipality_type"].tolist()):
            if attr is not None:
                attr["municipalityType"] = municipality_type

        links = links.drop(columns=["municipality_type"])
        return links

    def _get_municipality_regions(self):
        df_types = self.context.stage("data.spatial.municipality_types")
        df_municipalities, _ = self.context.stage("data.spatial.municipalities")
        df = df_types[["municipality_id", "municipality_type"]].merge(df_municipalities, on="municipality_id")
        df = gpd.GeoDataFrame(df, crs="EPSG:2056")
        df = df[["municipality_type", "geometry"]]
        return df.dissolve(by="municipality_type").reset_index()

    def _assign_municipality_types(self, regions):
        links_centers = RoutingPenaltyProvider.get_centroids(self.links, self.network.nodes)

        if links_centers.empty:
            return links_centers.assign(municipality_type="outside")[["link_id", "municipality_type"]]

        joined = gpd.sjoin(
            links_centers[["link_id", "geometry"]],
            regions[["municipality_type", "geometry"]],
            how="left",
            predicate="within",
        )

        links_types = joined[["link_id", "municipality_type"]].drop_duplicates(subset=["link_id"])
        links_types["municipality_type"] = links_types["municipality_type"].cat.add_categories(["outside"]).fillna("outside")
        return links_types
