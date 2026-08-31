from io import StringIO
import geopandas as gpd
import pandas as pd
from shapely import covers

from matsim.scenario.network.utils.speed_factors import SpeedFactorProvider


class RoutingPenaltyProvider:
    def __init__(self, context, links, nodes):
        self.context = context
        self.links = links.copy()
        self.nodes = nodes
        self.penalties = self._read_penalties(StringIO(DEFAULT_LINK_PENALTIES_CSV))
        self.special_region_by_link_id = self._assign_special_regions()

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
        if base_category is None:
            return 0.0

        municipality_type = str(attributes.get("municipalityType", "outside")).strip().lower()
        if municipality_type == "outside":
            return 0.0

        is_urban = municipality_type in ["urbancore", "urban"]
        special_region = self.special_region_by_link_id.get(link_id, 0)
        key = (base_category, is_urban, special_region)

        # The Java calibration merges sparse special-region groups into the
        # corresponding region-0 group. The exported defaults already contain
        # the expanded real keys; this fallback also covers combinations that
        # did not occur in the network used for calibration.
        return self.penalties.get(
            key, self.penalties.get((base_category, is_urban, 0), 0.0)
        )

    def read_penalties_from_csv(self, path):
        self.penalties = self._read_penalties(path)

    @staticmethod
    def _read_penalties(path_or_buffer):
        df = pd.read_csv(path_or_buffer, sep=";")
        required_columns = {
            "linkCategory",
            "isUrban",
            "specialRegion",
            "penalty(%)",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Missing routing-penalty CSV columns: "
                + ", ".join(sorted(missing_columns))
            )

        penalties = {}
        for _, row in df.iterrows():
            key = ( int(row["linkCategory"]),
                    str(row["isUrban"]).strip().lower() == "true",
                    int(row["specialRegion"]))
            penalties[key] = float(row["penalty(%)"])
        return penalties

    def _assign_special_regions(self):
        region_paths = self.context.stage("calibration.road_regions.penalty_calibration")
        region_paths = [path.strip() for path in region_paths.split(";") if path.strip()]
        if not region_paths or self.links.empty:
            return {}

        endpoints = self._get_link_endpoints(self.links, self.nodes)
        from_points = gpd.points_from_xy(
            endpoints.x_from_node, endpoints.y_from_node, crs="EPSG:2056"
        )
        to_points = gpd.points_from_xy(
            endpoints.x_to_node, endpoints.y_to_node, crs="EPSG:2056"
        )

        special_region_by_link_id = {}
        for special_region, path in enumerate(region_paths, start=1):
            geometry = gpd.read_file(path).geometry.union_all()
            in_region = covers(geometry, from_points) & covers(geometry, to_points)
            for link_id in endpoints.loc[in_region, "link_id"]:
                # Match Java's putIfAbsent: the first region in the configured
                # semicolon-separated list wins when regions overlap.
                special_region_by_link_id.setdefault(link_id, special_region)

        return special_region_by_link_id

    @staticmethod
    def _get_link_endpoints(links, nodes):
        return (
            links[["link_id", "from_node", "to_node"]]
            .merge(nodes, left_on="from_node", right_on="node_id", how="left")
            .merge(
                nodes,
                left_on="to_node",
                right_on="node_id",
                suffixes=("_from_node", "_to_node"),
                how="left",
            )
        )

    @staticmethod
    def get_centroids(links, nodes):
        links_centers = RoutingPenaltyProvider._get_link_endpoints(links, nodes)
        centroids_x = (links_centers.x_from_node + links_centers.x_to_node) / 2
        centroids_y = (links_centers.y_from_node + links_centers.y_to_node) / 2
        geometry = gpd.points_from_xy(
            centroids_x, centroids_y, crs="EPSG:2056"
        )
        return gpd.GeoDataFrame(
            links_centers[["link_id"]], geometry=geometry, crs="EPSG:2056"
        )


DEFAULT_LINK_PENALTIES_CSV = """
linkCategory;isUrban;specialRegion;penalty(%)
4;true;13;0.0000
2;true;14;0.0000
4;true;14;0.0000
2;false;1;0.3000
2;false;0;0.0000
2;false;3;0.0000
4;false;0;0.3000
2;false;2;0.0000
1;true;0;0.0895
2;false;5;0.0141
4;false;2;0.0000
3;true;0;0.0000
1;true;2;0.0000
1;true;1;0.0000
1;true;4;0.0000
2;false;9;0.3000
3;true;2;0.0000
1;true;3;0.0000
4;false;9;0.0595
1;true;5;0.0000
3;true;3;0.0000
2;false;13;0.0000
3;true;6;0.0000
1;true;8;0.3000
2;false;12;0.0000
3;true;5;0.0000
4;false;13;0.0000
3;true;8;0.0000
4;false;12;0.1300
2;false;14;0.0000
5;true;5;0.0000
1;true;9;0.0149
4;false;14;0.1810
1;true;14;0.0260
3;true;12;0.0000
3;true;11;0.0000
1;true;13;0.0000
3;true;14;0.0000
3;true;13;0.0000
5;true;13;0.0000
1;false;0;0.0885
3;false;0;0.2797
1;false;1;0.0321
5;false;0;0.0000
3;false;2;0.0000
1;false;3;0.0000
2;true;0;0.0000
2;true;3;0.0000
4;true;0;0.0000
2;true;2;0.0000
2;true;5;0.0000
3;false;9;0.0000
4;true;2;0.0000
3;false;12;0.3000
1;false;14;0.0000
4;true;5;0.0000
1;false;13;0.0000
3;false;14;0.0000
3;false;13;0.0469
2;true;8;0.3000
4;true;6;0.0000
2;true;11;0.3000
4;true;8;0.0000
2;true;13;0.1558
4;true;11;0.0000
"""