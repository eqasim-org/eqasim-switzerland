import pandas as pd

DEFAULT_SPEED_FACTORS = {
    1: dict(urbancore=0.98*1.05, rural=0.96*1.05, urban=0.98*1.05, suburban=0.98*1.05, outside=1.23*0.8*1.05),
    2: dict(urbancore=0.93*1.03, rural=0.89*1.03, urban=0.89*1.03, suburban=0.93*1.03, outside=1.16*0.8*1.03),
    3: dict(urbancore=0.91*0.85, rural=0.96*1.1, urban=0.94*0.97, suburban=0.99*1.05, outside=1.19*0.8),
    4: dict(urbancore=0.97*0.85, rural=0.98*1.1, urban=0.96*0.97, suburban=1.08*1.05, outside=1.06*0.8),
    5: dict(urbancore=0.91*0.85, rural=0.92*1.1, urban=0.90*0.97, suburban=0.97*1.05, outside=0.98*0.8),
}

# The categorization of links is based on the OSM highway tag
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
        self.speed_factors = DEFAULT_SPEED_FACTORS

    def process(self):
        if "attributes" not in self.links.columns:
            return self.links

        attrs = self.links["attributes"].tolist()
        modes = self.links["modes"].tolist()
        permlanes = self.links["permlanes"].tolist()
        freespeeds = self.links["freespeed"].tolist()

        for attr, link_modes, lanes, speed in zip(attrs, modes, permlanes, freespeeds):
            if isinstance(attr, dict):
                attr["speedFactor"] = self._get_speed_factor_from_values(attr, link_modes, lanes, speed)

        return self.links

    def _get_speed_factor_from_values(self, attributes, modes, number_of_lanes, freespeed):
        category = self._get_link_base_category_from_values(attributes, modes, number_of_lanes, freespeed)
        if category is None:
            return 1.0

        municipality_type = attributes.get("municipalityType", "outside")
        return self.speed_factors.get(category, {}).get(municipality_type, 1.0)

    def read_speed_factors_from_csv(self, path):
        # csv has these columns: # category;municipalityType;factor
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            category = row["category"]
            municipality_type = row["municipalityType"]
            factor = row["factor"]
            self.speed_factors.setdefault(category, {})[municipality_type] = factor
    
    @staticmethod
    def _get_link_base_category_from_values(attributes, modes, number_of_lanes, freespeed):
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
      