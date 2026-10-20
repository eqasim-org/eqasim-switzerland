"""Spatial enrichment and shared preprocessing for secondary destination choice."""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely import line_merge
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import hashlib
import logging

logger = logging.getLogger("synpp: secondary_nn.tools")

OSM_FEATURES = [
    "osm_aerialway_length_m", "osm_ski_piste_length_m", "osm_ski_area_m2",
    "osm_park_area_m2", "osm_garden_area_m2", "osm_viewpoint_count",
    "osm_attraction_count", "osm_cable_station_count", "osm_worship_count",
    "osm_shoreline_length_m", "osm_riverside_path_length_m",
    "osm_public_shore_path_length_m",
]
OSM_KINDS = dict(zip(
    ["aerialway", "piste", "ski_area", "park", "garden", "viewpoint", "attraction",
     "cable_station", "worship", "shoreline", "riverside_path", "shore_path"],
    OSM_FEATURES,
))
MOBILITY_FEATURES = [
    "driving_license", "bike_available", "pt_ga", "pt_halbtax", "pt_regional",
    "home_ovgk_a", "home_ovgk_b", "home_ovgk_c",
    "home_ovgk_d", "home_ovgk_none",
]
NN_ALGORITHMS = {"nn", "neural_network", "v2", "v.2", "locations_v2"}


def empty_features():
    return gpd.GeoDataFrame(columns=["osm_id", "kind", "name", "public", "geometry"],
                            geometry="geometry", crs=2056)


def classify_osm(tags):
    """Return all relevant themes; private objects remain context, not candidates."""
    kinds = []
    if tags.get("aerialway") not in (None, "no", "station", "pylon"):
        kinds.append("aerialway")
    if tags.get("aerialway") == "station":
        kinds.append("cable_station")
    if tags.get("piste:type") not in (None, "no"):
        kinds.append("piste")
        if tags.get("area") == "yes":
            kinds.append("ski_area")
    if tags.get("landuse") == "winter_sports" or tags.get("leisure") == "winter_sports":
        kinds.append("ski_area")
    if tags.get("leisure") in ("park", "garden"):
        kinds.append(tags["leisure"])
    if tags.get("tourism") == "viewpoint":
        kinds.append("viewpoint")
    if tags.get("tourism") == "attraction" and tags.get("name"):
        kinds.append("attraction")
    if tags.get("amenity") == "place_of_worship":
        kinds.append("worship")
    if (tags.get("natural") == "water" and tags.get("water", "lake") not in
            ("river", "stream", "ditch", "canal")) or tags.get("landuse") == "reservoir":
        kinds.append("water")
    if tags.get("waterway") in ("river", "stream", "canal") or tags.get("water") == "river":
        kinds.append("river")
    if tags.get("highway") in ("path", "footway", "pedestrian", "cycleway", "steps", "track"):
        kinds.append("path")
    return kinds


def public_access(tags):
    return (tags.get("access") not in ("no", "private", "customers", "permit")
            and tags.get("foot") not in ("no", "private")
            and not tags.get("disused") == "yes"
            and tags.get("lifecycle") not in ("disused", "abandoned"))



def connected_path_sections(pieces):
    """Unify branches as well as consecutive OSM ways into connected places."""
    merged = gpd.GeoSeries(pieces, crs=2056).union_all()
    segments = gpd.GeoSeries([merged], crs=2056).explode(index_parts=False).reset_index(drop=True)
    segments = segments.loc[segments.length > 0].reset_index(drop=True)
    if segments.empty:
        return []
    left, right = segments.sindex.query(segments, predicate="intersects")
    adjacency = coo_matrix((np.ones(len(left)), (left, right)), shape=(len(segments), len(segments))).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    return [line_merge(segments.loc[labels == label].union_all()) for label in np.unique(labels)]


def derive_water_features(features, distance_m=30.0):
    """Keep full shoreline context, but create visits only along accessible paths."""
    if distance_m <= 0:
        raise ValueError("Water path buffer must be positive")
    if features.empty:
        return features
    result = [features.loc[~features.kind.isin(["water", "river"])].copy()]
    paths = features.loc[(features.kind == "path") & features.public].copy()
    for water_kind, path_kind in [("water", "shore_path"), ("river", "riverside_path")]:
        water = features.loc[features.kind == water_kind].copy()
        if water.empty:
            continue
        if water_kind == "water":
            water.geometry = water.geometry.map(lambda g: g.boundary if g.geom_type in ("Polygon", "MultiPolygon") else g)
            shoreline = water.copy()
            shoreline["kind"] = "shoreline"
            shoreline["public"] = False
            result.append(shoreline)
        if paths.empty:
            continue
        # Intersect each path with nearby water buffers, not the entire water polygon.
        buffers = water.copy()
        buffers.geometry = buffers.geometry.buffer(distance_m)
        nearby = gpd.sjoin(paths, buffers[["geometry"]], predicate="intersects")
        if nearby.empty:
            continue
        pieces = []
        for row in nearby.itertuples():
            piece = row.geometry.intersection(buffers.geometry.loc[row.index_right])
            if not piece.is_empty and piece.length > 0:
                pieces.append(piece)
        if pieces:
            # Connected shoreline sections produce one candidate, regardless of OSM way cuts.
            parts = connected_path_sections(pieces)
            result.append(gpd.GeoDataFrame([
                dict(osm_id=path_kind + ":" + hashlib.sha256(g.normalize().wkb).hexdigest()[:20],
                     kind=path_kind, name="", public=True, geometry=g)
                for g in sorted(parts, key=lambda g: g.wkb_hex) if g.length > 0
            ], geometry="geometry", crs=2056))
    return gpd.GeoDataFrame(pd.concat(result, ignore_index=True), geometry="geometry", crs=2056)


def candidate_points(features):
    """One visit point per place or connected public waterside section."""
    allowed = {"park", "garden", "viewpoint", "attraction", "cable_station", "worship",
               "piste", "ski_area", "riverside_path", "shore_path"}
    places = features.loc[features.public & features.kind.isin(allowed)].copy()
    if places.empty:
        return places
    # Prefer explicit attractions and stations over area interiors when they coincide.
    priority = {"cable_station": 0, "viewpoint": 1, "attraction": 2, "worship": 3}
    places["priority"] = places.kind.map(priority).fillna(4)
    places = places.sort_values(["priority", "osm_id", "kind"]).reset_index(drop=True)
    paths = features.loc[(features.kind == "path") & features.public]
    points = []
    for geometry in places.geometry:
        if geometry.geom_type in ("LineString", "MultiLineString"):
            point = geometry.interpolate(.5, normalized=True)
        elif geometry.geom_type in ("Polygon", "MultiPolygon"):
            nearby = paths.iloc[paths.sindex.query(geometry, predicate="intersects")]
            accessible = nearby.geometry.intersection(geometry)
            accessible = accessible.loc[accessible.length > 0]
            if len(accessible):
                line = accessible.loc[accessible.length.idxmax()]
                point = line.interpolate(.5, normalized=True)
            else:
                point = geometry.representative_point()
        else:
            point = geometry.representative_point()
        points.append(point)
    places.geometry = points
    return places


def add_osm_destinations(destinations, features, municipalities, ovgk, radius_m=50.0):
    if radius_m <= 0:
        raise ValueError("OSM destination deduplication radius must be positive")
    places = candidate_points(features)
    if places.empty:
        return destinations
    # Existing companies of any activity block a duplicate artificial facility.
    company_mask = ((pd.to_numeric(destinations.number_employees, errors="coerce").fillna(0) > 0)
                    | destinations.filter(regex="^offers_").fillna(False).any(axis=1))
    companies = destinations.loc[company_mask]
    tree = cKDTree(np.column_stack([companies.geometry.x, companies.geometry.y])) if len(companies) else None
    xy = np.column_stack([places.geometry.x, places.geometry.y])
    distances = tree.query(xy)[0] if tree is not None else np.full(len(places), np.inf)
    places = places.loc[distances > radius_m].copy()
    # Suppress node/area duplicates and nearby newly proposed places deterministically.
    accepted = []
    if len(places):
        coords = np.column_stack([places.geometry.x, places.geometry.y])
        point_tree = cKDTree(coords)
        blocked = set()
        for i in range(len(places)):
            if i not in blocked:
                accepted.append(i)
                blocked.update(point_tree.query_ball_point(coords[i], radius_m))
    places = places.iloc[accepted].reset_index(drop=True)
    if places.empty:
        return destinations
    rows = gpd.GeoDataFrame(geometry=places.geometry.copy(), crs=2056)
    rows["destination_id"] = "OSM_" + places.osm_id.astype(str) + "_" + places.kind
    if rows.destination_id.isin(destinations.destination_id).any():
        raise ValueError("OSM destination IDs collide with the existing catalogue")
    rows["number_employees"] = 1
    rows["destination_x"] = rows.geometry.x
    rows["destination_y"] = rows.geometry.y
    for column in set(destinations.columns) | {"offers_leisure", "offers_other"}:
        if column.startswith("offers_"):
            rows[column] = column in ("offers_leisure", "offers_other")
    rows["osm_kind"] = places.kind.to_numpy()
    rows["osm_name"] = places.name.to_numpy()
    # Use containing municipalities and the best available overlapping PT class.
    joined = gpd.sjoin(rows[["geometry"]], municipalities[["municipality_id", "municipality_type", "geometry"]], how="left", predicate="intersects")
    joined = joined.loc[~joined.index.duplicated()]
    rows["municipality_id"] = joined.municipality_id.reindex(rows.index).fillna(-1).to_numpy()
    rows["municipality_type"] = joined.municipality_type.reindex(rows.index).fillna("rural").to_numpy()
    rows["ovgk"] = accessibility_at_points(rows.geometry.x, rows.geometry.y, ovgk)
    logger.info("Added %s OSM leisure/other destinations: %s", len(rows), places.kind.value_counts().to_dict())
    return gpd.GeoDataFrame(pd.concat([destinations, rows], ignore_index=True), geometry="geometry", crs=2056)


def accessibility_at_points(x, y, ovgk):
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=2056)
    matched = gpd.sjoin(points, ovgk[["ovgk", "geometry"]].to_crs(2056), how="left", predicate="intersects")
    rank = matched.ovgk.astype("string").map({"A": 0, "B": 1, "C": 2, "D": 3}).fillna(4)
    best = rank.groupby(level=0).min().reindex(points.index, fill_value=4).astype(int)
    return np.array(["A", "B", "C", "D", "None"])[best]


def aggregate_osm_features(cells, features):
    """Measure intersections with the actual clipped coordinate hierarchy polygons."""
    out = pd.DataFrame(0.0, index=cells.index, columns=OSM_FEATURES)
    for kind, column in OSM_KINDS.items():
        objects = features.loc[features.kind == kind, ["osm_id", "geometry"]].reset_index(drop=True)
        if objects.empty:
            continue
        if column.endswith("count"):
            objects.geometry = objects.geometry.representative_point()
        matches = gpd.sjoin(objects, cells[["geometry"]], predicate="intersects")
        if column.endswith("count"):
            # Boundary points count once at each hierarchy level.
            matches = matches.loc[~matches.index.duplicated()]
            values = matches.groupby("index_right").size()
        else:
            values = {}
            for row in matches.itertuples():
                intersection = row.geometry.intersection(cells.geometry.loc[row.index_right])
                value = intersection.area if column.endswith("m2") else intersection.length
                values[row.index_right] = values.get(row.index_right, 0.0) + value
            values = pd.Series(values, dtype=float)
        out[column] = values.reindex(cells.index, fill_value=0.0)
    return out


def attach_candidate_features(attributes, cells):
    indexed = cells.set_index("h3_index")
    for attrs in attributes.values():
        for column in OSM_FEATURES:
            attrs[column] = indexed.loc[attrs["children"], column].to_numpy(dtype=float)


def enrich_candidate_tensor(tensor, candidate_x, candidate_y, valid_mask, cells):
    """Join context by exact candidate centroid; padded alternatives remain zero."""
    centroids = cells["centroid"]
    tree = cKDTree(np.column_stack([centroids.x, centroids.y]))
    shape = valid_mask.shape
    x = np.broadcast_to(candidate_x, shape)
    y = np.broadcast_to(candidate_y, shape)
    distances, indices = tree.query(np.column_stack([x[valid_mask], y[valid_mask]]))
    if (distances > 1e-5).any():
        raise ValueError("Candidate centroids do not match the shared H3 feature table")
    extra = np.zeros((*shape, len(OSM_FEATURES)))
    extra[valid_mask] = cells[OSM_FEATURES].to_numpy(dtype=float)[indices]
    # Scaled environmental quantities precede existing fraction features.
    from .feature_encoding import N_CANDIDATE_DYNAMIC, CANDIDATE_STATIC_SCALED_FEATURES
    insertion = N_CANDIDATE_DYNAMIC + len(CANDIDATE_STATIC_SCALED_FEATURES) - len(OSM_FEATURES)
    return np.concatenate([tensor[:, :, :insertion], extra, tensor[:, :, insertion:]], axis=2)


def mobility_features(persons, constants=None, survey=False, ovgk=None):
    out = pd.DataFrame(index=persons.index)
    def binary(name):
        if name not in persons:
            return pd.Series(False, index=persons.index)
        return pd.to_numeric(persons[name], errors="coerce").fillna(0) > 0
    out["driving_license"] = binary("driving_license")
    out["bike_available"] = pd.to_numeric(persons.bike_availability, errors="coerce").isin([constants.BIKE_AVAILABILITY_ALWAYS, constants.BIKE_AVAILABILITY_SOMETIMES]) if survey else binary("bike_availability")
    subscriptions = persons.get("subscriptions", pd.Series("null", index=persons.index)).fillna("null").astype(str)
    ga = binary("subscriptions_ga") | binary("subscriptions_ga_class") | subscriptions.eq("GA")
    halbtax = binary("subscriptions_halbtax") | subscriptions.isin(["HT", "HTA", "HTA+VA", "VA+HT"])
    regional = binary("subscriptions_verbund") | binary("subscriptions_verbund_class") | binary("subscriptions_strecke") | binary("subscriptions_strecke_class") | subscriptions.isin(["VA", "HTA+VA", "VA+HT"])
    if survey:
        # Match the subscription stage's common five-category representation.
        age = persons.get("age", pd.Series(99, index=persons.index))
        ga = ga | (binary("subscriptions_junior") & (age < 16))
        halbtax = halbtax & ~ga
        regional = regional & ~ga
    if "pt_subscription" in persons:
        category = pd.to_numeric(persons.pt_subscription, errors="raise")
        if not category.isin([0, 1, 2, 3, 4]).all():
            raise ValueError("Unknown synthetic PT subscription category")
        ga = category.eq(1)
        regional = category.isin([2, 4])
        halbtax = category.isin([3, 4])
    out["pt_ga"], out["pt_halbtax"], out["pt_regional"] = ga, halbtax, regional
    classes = accessibility_at_points(persons.home_x, persons.home_y, ovgk) if ovgk is not None else persons.get("ovgk", pd.Series("None", index=persons.index)).fillna("None").to_numpy()
    for category in ["A", "B", "C", "D", "None"]:
        out["home_ovgk_" + category.lower()] = classes == category
    return out[MOBILITY_FEATURES].astype(np.float32)


def training_mobility(context, frame):
    persons = context.stage("data.microcensus.persons")
    features = mobility_features(persons, context.stage("data.constants"), survey=True,
                                 ovgk=context.stage("data.spatial.ovgk"))
    features["person_id"] = persons.person_id.to_numpy()
    return frame[["person_id"]].merge(features, on="person_id", how="left", validate="many_to_one")[MOBILITY_FEATURES].to_numpy()


class OriginContext:
    """Summarize nearby opportunities without observing the chosen destination."""

    def __init__(self, cells, radii=(1500.0, 5000.0)):
        from .feature_encoding import STATIC_CANDIDATE_FEATURES
        self.columns = list(STATIC_CANDIDATE_FEATURES)
        self.radii = tuple(float(r) for r in radii)
        if not self.radii or any(not np.isfinite(r) or r <= 0 for r in self.radii):
            raise ValueError("Origin context radii must be finite and positive")
        self.xy = np.column_stack([cells.centroid.x, cells.centroid.y])
        self.tree = cKDTree(self.xy)
        self.values = cells[self.columns].to_numpy(dtype=float)
        self.cache = {}
        self.names = ["origin_nearest_destination_cell_m", "origin_home_distance_m", "origin_work_distance_m"]
        for radius in self.radii:
            self.names.extend(f"origin_{int(radius)}m_{name}" for name in self.columns)
            self.names.append(f"origin_{int(radius)}m_cells")

    def vector(self, origin_x, origin_y, home_x, home_y, work_x, work_y, has_work):
        # A 250 m origin grid shares summaries across repeated synthetic locations.
        key = (int(np.floor(origin_x / 250)), int(np.floor(origin_y / 250)))
        if key not in self.cache:
            center = (np.asarray(key, dtype=float) + .5) * 250
            summaries = []
            for radius in self.radii:
                indices = self.tree.query_ball_point(center, radius)
                block = self.values[indices]
                values = block.sum(axis=0) if len(indices) else np.zeros(len(self.columns))
                # Fractions are means over cells, quantities are sums.
                for i, column in enumerate(self.columns):
                    if column.startswith("ovgk_share_") or column in ("urban", "urban_core", "outside_fraction"):
                        values[i] /= max(len(indices), 1)
                summaries.extend(values)
                summaries.append(len(indices))
            if len(self.cache) >= 10000:
                self.cache.clear()
            self.cache[key] = np.asarray(summaries)
        nearest = self.tree.query([origin_x, origin_y])[0]
        home = np.hypot(origin_x - home_x, origin_y - home_y)
        work = np.hypot(origin_x - work_x, origin_y - work_y) if has_work else 0.0
        return np.concatenate([[nearest, home, work], self.cache[key]]).astype(np.float32)
