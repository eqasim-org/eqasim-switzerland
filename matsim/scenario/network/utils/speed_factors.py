from io import StringIO
import geopandas as gpd
import pandas as pd
from shapely import covers

# The categorization of links is based on the OSM highway tag.
CATEGORY_1_HIGHWAY_TYPES = ["motorway", "motorway_link", "trunk", "trunk_link"]
CATEGORY_2_HIGHWAY_TYPES = ["primary", "primary_link"]
CATEGORY_3_HIGHWAY_TYPES = ["secondary", "secondary_link"]
CATEGORY_4_HIGHWAY_TYPES = ["tertiary", "tertiary_link"]
CATEGORY_5_HIGHWAY_TYPES = ["residential", "unclassified", "living_street", "service", "track"]

class SpeedFactorProvider:
    def __init__(self, context, links, nodes):
        self.context = context
        self.links = links.copy()
        self.nodes = nodes
        self.speed_factors = self._read_speed_factors(StringIO(DEFAULT_SPEED_FACTORS_CSV))
        self.special_region_by_link_id = self._assign_special_regions()

    def process(self):
        if "attributes" not in self.links.columns:
            return self.links

        attrs = self.links["attributes"].tolist()
        link_ids = self.links["link_id"].tolist()
        modes = self.links["modes"].tolist()
        permlanes = self.links["permlanes"].tolist()
        freespeeds = self.links["freespeed"].tolist()

        for attr, link_id, link_modes, lanes, speed in zip(
            attrs, link_ids, modes, permlanes, freespeeds
        ):
            if isinstance(attr, dict):
                attr["speedFactor"] = self._get_speed_factor_from_values(
                    attr, link_id, link_modes, lanes, speed
                )

        return self.links

    def _get_speed_factor_from_values(
        self, attributes, link_id, modes, number_of_lanes, freespeed
    ):
        category = self._get_link_base_category_from_values(
            attributes, modes, number_of_lanes, freespeed
        )
        if category is None:
            return 1.0

        municipality_type = self._get_municipality_type(attributes)
        special_region = self.special_region_by_link_id.get(link_id, 0)
        return self.speed_factors.get(
            (category, municipality_type, special_region), 1.0
        )

    def read_speed_factors_from_csv(self, path):
        self.speed_factors.update(self._read_speed_factors(path))

    @staticmethod
    def _read_speed_factors(path_or_buffer):
        df = pd.read_csv(path_or_buffer, sep=";")
        required_columns = {"category", "municipalityType", "factor"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Missing freespeed-factor CSV columns: "
                + ", ".join(sorted(missing_columns))
            )

        has_special_regions = "specialRegion" in df.columns
        speed_factors = {}
        for _, row in df.iterrows():
            municipality_type = str(row["municipalityType"]).strip().lower()
            special_region = int(row["specialRegion"]) if has_special_regions else 0
            key = (int(row["category"]), municipality_type, special_region)
            speed_factors[key] = float(row["factor"])
        return speed_factors

    def _assign_special_regions(self):
        region_paths = self.context.stage(
            "calibration.road_regions.penalty_calibration"
        )
        region_paths = [
            path.strip() for path in region_paths.split(";") if path.strip()
        ]
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
    def _get_municipality_type(attributes):
        municipality_type = attributes.get("municipalityType")
        if isinstance(municipality_type, str):
            municipality_type = municipality_type.strip().lower()
            if municipality_type:
                return municipality_type
        return "unknown"

    @staticmethod
    def _get_link_base_category_from_values(
        attributes, modes, number_of_lanes, freespeed
    ):
        osm_highway = attributes.get("osm:way:highway", None)
        if osm_highway is None or "car" not in modes:
            return None
        if osm_highway in CATEGORY_1_HIGHWAY_TYPES:
            return 1
        if osm_highway in CATEGORY_2_HIGHWAY_TYPES:
            return 2
        if osm_highway in CATEGORY_3_HIGHWAY_TYPES:
            return 3
        if osm_highway in CATEGORY_4_HIGHWAY_TYPES:
            return 4
        if osm_highway in CATEGORY_5_HIGHWAY_TYPES:
            return 4 if (number_of_lanes > 1 or freespeed > 45 / 3.6) else 5
        return None


DEFAULT_SPEED_FACTORS_CSV = """
category;municipalityType;specialRegion;factor
1;urban;13;1.04
3;urbancore;5;0.84
3;urbancore;6;0.87
3;urbancore;3;0.87
1;urban;11;1.0
1;urban;10;1.13
3;urbancore;4;0.84
3;urbancore;1;0.84
3;urbancore;2;0.87
3;urbancore;0;0.87
1;urban;5;1.04
3;urbancore;13;0.88
1;urban;4;1.09
3;urbancore;11;0.74
4;outside;0;0.91
1;urban;2;1.04
4;outside;1;0.88
1;urban;9;1.12
3;urbancore;9;0.87
3;urbancore;10;0.85
3;urbancore;7;0.85
1;urban;7;1.06
1;urban;6;1.1
1;urban;1;1.02
1;urban;0;1.08
4;outside;6;0.92
4;outside;8;0.76
4;outside;2;0.84
5;rural;5;0.94
4;urbancore;4;0.91
4;urbancore;5;0.92
4;urbancore;2;0.93
4;urbancore;3;0.87
5;rural;2;0.91
4;urbancore;0;0.96
5;rural;1;0.97
4;urbancore;1;1.02
5;rural;0;0.97
4;urbancore;13;0.88
4;urbancore;10;0.94
4;urbancore;11;0.81
5;outside;0;0.9
4;urbancore;9;0.9
4;urbancore;6;0.9
4;urbancore;7;0.93
5;rural;13;1.0
5;rural;9;1.17
5;rural;7;1.01
5;rural;6;0.91
5;outside;6;0.85
5;outside;8;0.8
5;outside;1;0.76
5;outside;2;0.78
5;urbancore;3;0.81
4;rural;6;0.97
4;rural;5;0.95
5;urbancore;4;0.83
5;urbancore;1;0.88
5;urbancore;2;0.87
4;rural;2;0.9
5;urbancore;0;0.88
4;rural;1;0.93
4;rural;0;1.08
5;urbancore;11;0.77
5;urbancore;9;0.92
5;urbancore;10;0.91
5;urbancore;7;0.88
2;outside;2;0.96
5;urbancore;5;0.82
2;outside;1;1.03
5;urbancore;6;1.01
2;outside;0;1.02
4;rural;13;0.94
4;rural;9;1.08
4;rural;7;1.05
1;suburban;0;1.06
1;suburban;1;1.01
1;suburban;8;1.03
1;suburban;9;1.07
2;outside;6;1.03
1;suburban;6;1.02
1;suburban;7;1.04
2;outside;11;0.95
1;suburban;4;1.08
1;suburban;5;1.03
1;suburban;2;1.04
5;urbancore;13;0.88
2;outside;8;0.94
1;suburban;13;1.04
1;suburban;10;1.03
1;suburban;11;1.03
3;rural;6;0.97
3;rural;7;1.09
3;rural;5;1.04
3;rural;2;1.01
3;rural;0;1.05
3;rural;1;1.02
3;outside;2;0.94
3;outside;1;1.03
3;outside;0;0.95
3;rural;13;1.02
2;suburban;0;0.99
2;suburban;7;0.99
3;outside;6;0.92
2;suburban;8;0.97
2;suburban;5;0.98
2;suburban;6;0.96
2;suburban;4;1.0
3;outside;8;0.95
2;suburban;1;0.97
2;suburban;2;1.0
2;suburban;13;0.93
2;suburban;11;0.98
2;suburban;9;1.07
2;suburban;10;0.96
2;rural;7;1.0
5;urban;10;0.99
5;urban;11;0.9
2;rural;5;0.94
5;urban;13;0.97
2;rural;6;0.99
5;urban;6;0.96
5;urban;7;0.92
2;rural;1;0.96
2;rural;2;1.01
5;urban;9;0.99
5;urban;2;0.92
5;urban;3;0.91
2;rural;0;0.98
5;urban;4;0.99
5;urban;5;0.91
5;urban;0;0.93
5;urban;1;0.99
3;suburban;6;1.02
3;suburban;7;1.15
3;suburban;4;1.07
3;suburban;5;1.04
3;suburban;2;1.03
3;suburban;0;1.04
3;suburban;1;1.06
3;suburban;13;1.01
3;suburban;10;1.04
3;suburban;11;1.03
3;suburban;8;1.03
3;suburban;9;1.09
4;urban;11;0.93
4;urban;13;1.04
1;rural;6;1.01
4;urban;7;1.0
4;urban;8;1.04
1;rural;5;1.02
1;rural;2;1.11
4;urban;9;1.04
4;urban;10;1.05
4;urban;3;1.01
1;rural;0;1.05
1;outside;0;0.98
1;rural;1;0.99
4;urban;4;1.03
4;urban;5;0.93
4;urban;6;0.99
4;urban;0;1.02
1;outside;2;0.95
4;urban;1;1.02
4;urban;2;0.93
1;outside;1;0.97
4;suburban;5;1.13
1;outside;8;0.79
4;suburban;6;1.07
1;outside;6;1.0
4;suburban;4;1.13
4;suburban;1;1.13
4;suburban;2;1.07
4;suburban;0;1.14
4;suburban;13;1.1
4;suburban;11;1.13
4;suburban;9;1.2
4;suburban;10;1.12
4;suburban;7;1.13
4;suburban;8;1.13
1;urbancore;7;1.04
3;urban;11;0.91
3;urban;10;0.99
1;urbancore;5;1.07
3;urban;9;1.01
3;urban;8;0.98
1;urbancore;6;1.01
1;urbancore;3;1.11
1;urbancore;4;1.08
3;urban;13;0.95
1;urbancore;1;1.02
1;urbancore;2;1.05
3;urban;3;1.04
3;urban;2;0.9
3;urban;1;0.99
3;urban;0;1.01
3;urban;7;0.97
3;urban;6;0.96
3;urban;5;0.97
1;urbancore;9;1.13
1;urbancore;10;1.13
3;urban;4;1.0
1;urbancore;0;1.07
5;suburban;4;1.0
5;suburban;5;1.01
5;suburban;2;1.0
5;suburban;0;1.02
5;suburban;1;0.99
5;suburban;13;0.99
5;suburban;10;1.04
5;suburban;11;1.02
5;suburban;8;1.02
5;suburban;9;1.1
5;suburban;6;0.99
5;suburban;7;1.02
2;urbancore;6;0.9
2;urbancore;7;0.97
2;urban;11;0.91
2;urbancore;4;0.97
2;urban;10;0.93
2;urbancore;5;0.96
2;urban;9;0.98
2;urbancore;2;0.99
2;urbancore;0;0.98
2;urban;13;0.91
2;urbancore;1;0.94
2;urban;4;0.94
2;urban;2;0.95
2;urbancore;13;1.0
2;urban;1;0.94
2;urban;8;0.96
2;urbancore;10;0.96
2;urban;7;0.94
2;urbancore;11;0.95
2;urban;6;0.98
2;urban;5;0.92
2;urbancore;9;1.04
2;urban;0;0.98
"""