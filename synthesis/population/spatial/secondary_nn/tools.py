"""Spatial enrichment and shared preprocessing for secondary destination choice."""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely import line_merge, is_missing
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
PASSENGER_AERIALWAYS = {
    "cable_car", "gondola", "chair_lift", "mixed_lift", "drag_lift", "t-bar",
    "j-bar", "platter", "rope_tow", "magic_carpet", "zip_line",
}
NN_ALGORITHMS = {"nn", "neural_network", "v2", "v.2", "locations_v2"}


def model_validation_options(context, *, folds=None, strategy=None, min_epochs=None, val_every=1):
    """Read declared controls; independent folds select duration before a full refit."""
    strategy = context.config("secondary_nn_model_validation_strategy") if strategy is None else strategy
    if strategy not in ("fixed", "cross_validation"):
        raise ValueError("Model validation strategy must be fixed or cross_validation")
    cross_validate = strategy == "cross_validation"
    return dict(
        n_val_folds=int(context.config("secondary_nn_model_validation_folds") if folds is None else folds),
        early_stopping_min_epochs=int(context.config("secondary_nn_model_min_epochs") if min_epochs is None else min_epochs),
        cross_validate=cross_validate, validation_seed=int(context.config("random_seed")),
        val_every=1 if cross_validate else int(val_every),
        refit_full=cross_validate or not bool(context.config("secondary_nn_use_target_distance")))


def origin_candidate_indices(tree, origin_x, origin_y, support_mask, *, radius_m,
                             min_candidates, expand_step_m=200., pad_to_min=True):
    """Use identical origin-centred, purpose-eligible H3 neighbourhoods in both phases."""
    support = np.asarray(support_mask, dtype=bool)
    if support.ndim != 1 or not support.any():
        raise ValueError("No eligible destinations for the requested short-range purpose")
    if len(support) != tree.data.shape[0]:
        raise ValueError("Short-range purpose mask must align with the H3 catalogue")
    if not np.isfinite([origin_x, origin_y, radius_m]).all() or radius_m < 0:
        raise ValueError("Short-range origin and radius must be finite, with nonnegative radius")
    if int(min_candidates) < 1:
        raise ValueError("Short-range minimum candidates must be positive")
    center = np.array([[origin_x, origin_y]], dtype=float)

    # Expand only around the origin, never toward the observed destination.

    if expand_step_m is None:
        nearby = tree.query_radius(center, r=radius_m)[0]
        result = nearby[support[nearby]]
        if pad_to_min and len(result) < min_candidates:
            eligible = np.flatnonzero(support)
            # The full nearest ordering is needed only in this compatibility path.
            nearest = tree.query(center, k=len(support))[1][0]
            result = np.union1d(result, nearest[support[nearest]][:min(min_candidates, len(eligible))])
        return np.sort(result).astype(np.int64)
    if not np.isfinite(expand_step_m) or expand_step_m <= 0:
        raise ValueError("Short-range expansion step must be finite and positive")
    required = min(int(min_candidates), int(support.sum()))
    radius = float(radius_m)
    while True:
        nearby = tree.query_radius(center, r=radius)[0]
        result = nearby[support[nearby]]
        if len(result) >= required:
            return np.sort(result).astype(np.int64)
        radius += float(expand_step_m)


def origin_training_choice_sets(frame, cell_ids, tree, support_masks, *, target_column,
                                radius_m, min_candidates, expand_step_m):
    """Audit observations against origin-only queries without inserting their target cells."""
    leaf_index = {str(cell): i for i, cell in enumerate(cell_ids)}
    valid_rows, choices, audit_rows = [], [], []
    for i, (_, row) in enumerate(frame.iterrows()):
        reason, indices = None, np.empty(0, dtype=np.int64)
        chosen = leaf_index.get(str(row[target_column]))
        mask = require_purpose_mask(support_masks, row.purpose)
        if not np.isfinite([row.origin_x, row.origin_y]).all():
            reason = "invalid_origin"
        elif chosen is None:
            reason = "observed_leaf_not_in_catalogue"
        elif not mask[chosen]:
            reason = "observed_leaf_has_no_eligible_facility"
        else:
            indices = origin_candidate_indices(tree, row.origin_x, row.origin_y, mask,
                radius_m=radius_m, min_candidates=min_candidates, expand_step_m=expand_step_m)
            if chosen not in indices:
                reason = "observed_leaf_outside_origin_neighbourhood"
        audit_rows.append(dict(person_id=row.person_id, trip_id=row.trip_id,
                               reason=reason or "included", candidates=len(indices)))
        if reason is None:
            valid_rows.append(i)
            choices.append(indices)
    audit = pd.DataFrame(audit_rows, columns=["person_id", "trip_id", "reason", "candidates"])
    return valid_rows, choices, audit


class DistanceDestinationSampler:
    """Sample actual facilities by target-radius compatibility across origin H3 cells."""

    def __init__(self, cell_ids, pools, coordinates):
        counts = np.asarray([len(pools.get(cell, ())) for cell in cell_ids], dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(counts)]
        self.facilities = [facility for cell in cell_ids for facility in pools.get(cell, ())]
        self.xy = (np.concatenate([coordinates[cell] for cell, count in zip(cell_ids, counts) if count])
                   if counts.any() else np.empty((0, 2)))
        if self.xy.shape != (len(self.facilities), 2) or not np.isfinite(self.xy).all():
            raise ValueError("Short-range facility coordinates must be finite and aligned with their pools")

    def distribution(self, cells, origin_x, origin_y, target_distance, *, sigma_fraction=.1, sigma_min_m=25.):
        if (not np.isfinite([origin_x, origin_y, target_distance, sigma_fraction, sigma_min_m]).all()
                or target_distance < 0 or sigma_fraction <= 0 or sigma_min_m <= 0):
            raise ValueError("Distance sampling needs finite coordinates, a nonnegative target and positive widths")
        cells = np.asarray(cells, dtype=np.int64)
        if len(np.unique(cells)) != len(cells):
            raise ValueError("Short-range candidate cells must be unique")
        starts = self.offsets[cells]
        counts = self.offsets[cells + 1] - starts
        total = int(counts.sum())
        if not total:
            raise ValueError("No purpose-eligible facilities in the short-range origin neighbourhood")

        # Gather variable cell pools without scanning the whole facility catalogue per trip.

        indices = np.repeat(starts - (np.cumsum(counts) - counts), counts) + np.arange(total)
        distances = np.hypot(self.xy[indices, 0] - origin_x, self.xy[indices, 1] - origin_y)
        sigma = max(sigma_min_m, sigma_fraction * target_distance)
        error = np.abs(distances - target_distance)

        # Subtract the best radial error so distant/sparse pools never become a uniform fallback.

        relative_error = (error - error.min()) / sigma
        log_weight = -.5 * relative_error * ((error + error.min()) / sigma)
        weights = np.exp(log_weight)
        return indices, weights / weights.sum()

    def sample(self, cells, origin_x, origin_y, target_distance, rng, **widths):
        indices, probability = self.distribution(cells, origin_x, origin_y, target_distance, **widths)
        return self.facilities[int(rng.choice(indices, p=probability))]


def empty_features():
    return gpd.GeoDataFrame(columns=["osm_id", "kind", "name", "public", "access_reason",
                                         "access", "foot", "garden_type", "geometry"],
                            geometry="geometry", crs=2056)


def classify_osm(tags):
    """Identify themes before applying the common public-access filter."""
    kinds = []
    if tags.get("aerialway") in PASSENGER_AERIALWAYS:
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
    """Whether a mapped place is eligible for visitor destination assignment."""
    return osm_access_decision(tags)[0]


def osm_access_decision(tags, kind=None):
    """Use access restrictions first, then evidence specific to the feature type."""
    values = {key: str(value).strip().lower() for key, value in tags.items()}
    kinds = {kind} if kind else set(classify_osm(tags))
    garden_type = values.get("garden:type") or values.get("garden", "")

    # Closed sites and residential grounds never become visitor destinations.

    if any(values.get(key) == "yes"
           for key in ("disused", "abandoned", "construction", "locked", "membership")):
        return False, "closed_or_members_only"
    if values.get("lifecycle") in ("disused", "abandoned", "construction"):
        return False, "inactive"
    if kinds & {"garden", "park"}:
        residential_buildings = {"house", "detached", "semidetached_house", "terrace",
                                 "apartments", "residential", "bungalow", "static_caravan"}
        if (garden_type in {"residential", "residental", "private"}
                or values.get("building") in residential_buildings
                or values.get("landuse") == "residential"):
            return False, "residential_grounds"
    if kinds & {"aerialway", "cable_station"}:
        if (values.get("usage") in {"industrial", "freight"}
                or values.get("aerialway:goods") == "only"
                or values.get("aerialway:access") == "no"):
            return False, "non_visitor_aerialway"

    # Explicit restrictions override names, fees, ownership and default access.

    ski_facility = bool(kinds & {"aerialway", "cable_station", "piste", "ski_area"})
    if (values.get("access:conditional") or values.get("foot:conditional")
            or (ski_facility and values.get("ski:conditional"))):
        return False, "conditional_access"
    effective = values.get("foot") or values.get("access", "")
    # A lift may prohibit pedestrians while explicitly admitting skiers.
    if ski_facility and values.get("ski") in {"yes", "designated", "permissive", "official", "customers"}:
        effective = values["ski"]
    if effective in {"yes", "public", "permissive", "designated", "official"}:
        return True, "explicit_access"
    visitor_service = bool(kinds & {"aerialway", "cable_station", "piste", "ski_area",
                                   "attraction", "worship"})
    visitor_garden = ("garden" in kinds
                      and garden_type in {"botanical", "arboretum", "community"})
    if effective == "customers" and (visitor_service or visitor_garden):
        return True, "visitor_customers"
    if effective:
        return False, "restricted_access"

    # Gardens need more evidence than their name; untyped backyards stay excluded.

    if "garden" in kinds:
        if visitor_garden:
            return True, "visitor_garden_type"
        named = bool(str(tags.get("name", "")).strip())
        public_operator = values.get("operator:type") in {"public", "government", "municipal"}
        if named and public_operator:
            return True, "named_garden_public_operator"
        if named and values.get("tourism") == "attraction":
            return True, "named_garden_attraction"
        if named and not garden_type:
            return False, "garden_needs_path"
        return False, "uncertain_garden"

    # Visitor facilities do not require public ownership or free admission.

    if visitor_service or kinds & {"park", "viewpoint"}:
        return True, "visitor_feature_default"
    if "path" in kinds and values.get("highway") in {"path", "footway", "pedestrian", "steps"}:
        return True, "pedestrian_path_default"
    if kinds & {"water", "river"}:
        return True, "water_context"
    return False, "uncertain_access"


def confirm_garden_access(features):
    """Recover named gardens with a visitor path entering their interior."""
    if "access_reason" not in features:
        return features
    pending = features.access_reason.eq("garden_needs_path") & features.kind.eq("garden")
    gardens = features.loc[pending & features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gardens.empty:
        return features

    # Restricted grounds must not supply evidence through an intersecting path.

    eligible = public_osm_features(features.loc[~pending])
    paths = eligible.loc[eligible.kind.eq("path")]
    if paths.empty:
        return features
    interiors = gardens.geometry.buffer(-1.0)
    path_rows, garden_rows = interiors.sindex.query(paths.geometry, predicate="intersects")
    result = features.copy()
    for path_row, garden_row in zip(path_rows, garden_rows):
        if paths.geometry.iloc[path_row].intersection(interiors.iloc[garden_row]).length > 0:
            index = gardens.index[garden_row]
            result.loc[index, "public"] = True
            result.loc[index, "access_reason"] = "named_garden_with_path"
    return result


def public_osm_features(features):
    """Keep eligible visitor records outside restricted or uncertain mapped areas."""
    if "public" not in features:
        raise ValueError("OSM features require public-access flags")
    permitted = features["public"].fillna(False).astype(bool)
    result = features.loc[permitted].copy().reset_index(drop=True)
    restricted = features.loc[~permitted & features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if result.empty or restricted.empty:
        return result

    # Exclude overlapping areas whose public access cannot be confirmed.
    rows, areas = restricted.sindex.query(result.geometry, predicate="intersects")
    geometry = result.geometry.to_numpy(copy=True)
    matches = pd.DataFrame({"row": rows, "area": areas})
    for row, indices in matches.groupby("row", sort=False):
        exclusion = restricted.geometry.iloc[indices.area.to_numpy()].union_all()
        geometry[row] = geometry[row].difference(exclusion)
    result.geometry = geometry
    return result.loc[~result.geometry.is_empty & ~is_missing(result.geometry.to_numpy())].reset_index(drop=True)


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
    """Derive public waterside paths and only their accessible shoreline sections."""
    if distance_m <= 0:
        raise ValueError("Water path buffer must be positive")
    features = public_osm_features(confirm_garden_access(features))
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
        if paths.empty:
            continue
        # Intersect each path with nearby water buffers, not the entire water polygon.
        buffers = water.copy()
        buffers.geometry = buffers.geometry.buffer(distance_m)
        nearby = gpd.sjoin(paths, buffers[["geometry"]], predicate="intersects")
        if nearby.empty:
            continue
        pieces = []
        shore_sections = {}
        for row in nearby.itertuples():
            piece = row.geometry.intersection(buffers.geometry.loc[row.index_right])
            if not piece.is_empty and piece.length > 0:
                pieces.append(piece)
                if water_kind == "water":
                    section = water.geometry.loc[row.index_right].intersection(row.geometry.buffer(distance_m))
                    if not section.is_empty and section.length > 0:
                        shore_sections.setdefault(row.index_right, []).append(section)
        if shore_sections:
            # Count shoreline length only where the adjoining path is public.
            shoreline = water.loc[list(shore_sections)].copy()
            shoreline.geometry = [gpd.GeoSeries(shore_sections[index], crs=2056).union_all()
                                  for index in shoreline.index]
            shoreline["kind"] = "shoreline"
            shoreline["public"] = True
            shoreline["access_reason"] = "pedestrian_shore_access"
            result.append(shoreline)
        if pieces:
            # Connected shoreline sections produce one candidate, regardless of OSM way cuts.
            parts = connected_path_sections(pieces)
            result.append(gpd.GeoDataFrame([
                dict(osm_id=path_kind + ":" + hashlib.sha256(g.normalize().wkb).hexdigest()[:20],
                     kind=path_kind, name="", public=True,
                     access_reason="pedestrian_waterside_access", geometry=g)
                for g in sorted(parts, key=lambda g: g.wkb_hex) if g.length > 0
            ], geometry="geometry", crs=2056))
    return gpd.GeoDataFrame(pd.concat(result, ignore_index=True), geometry="geometry", crs=2056)


def candidate_points(features):
    """Create visit points from the already eligible OSM stage output."""
    allowed = {"park", "garden", "viewpoint", "attraction", "cable_station", "worship",
               "piste", "ski_area", "riverside_path", "shore_path"}
    places = features.loc[features.kind.isin(allowed)].copy()
    if places.empty:
        return places
    # Prefer explicit attractions and stations over area interiors when they coincide.
    priority = {"cable_station": 0, "viewpoint": 1, "attraction": 2, "worship": 3}
    places["priority"] = places.kind.map(priority).fillna(4)
    places = places.sort_values(["priority", "osm_id", "kind"]).reset_index(drop=True)
    paths = features.loc[features.kind == "path"]
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
    """Measure the already eligible OSM stage output in clipped H3 polygons."""
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

    def __init__(self, cells, radii=(1500.0, 5000.0, 20000.0), purpose_counts=None):
        from .feature_encoding import STATIC_CANDIDATE_FEATURES
        self.columns = list(STATIC_CANDIDATE_FEATURES)
        self.radii = tuple(float(r) for r in radii)
        if not self.radii or any(not np.isfinite(r) or r <= 0 for r in self.radii):
            raise ValueError("Origin context radii must be finite and positive")
        self.xy = np.column_stack([cells.centroid.x, cells.centroid.y])
        self.tree = cKDTree(self.xy)
        self.values = cells[self.columns].to_numpy(dtype=float)
        self.cache = {}
        self.purpose_values = {}
        self.purpose_trees = {}
        for purpose, fallback in {'work': 'num_statent', 'home': 'num_statent', 'other': 'num_statent',
                                  'shop': 'shop', 'leisure': 'leisure', 'education': 'education'}.items():
            counts = cells[fallback].to_numpy(dtype=float) if purpose_counts is None else purpose_counts[purpose].reindex(cells.h3_index, fill_value=0).to_numpy(dtype=float)
            self.purpose_values[purpose] = counts
            self.purpose_trees[purpose] = cKDTree(self.xy[counts > 0]) if (counts > 0).any() else None
        self.names = ["origin_nearest_destination_cell_m", "origin_home_distance_m", "origin_work_distance_m",
                      "origin_nearest_purpose_opportunity_m", "origin_has_purpose_opportunities"]
        for radius in self.radii:
            self.names.extend(f"origin_{int(radius)}m_{name}" for name in self.columns)
            self.names.extend([f"origin_{int(radius)}m_cells", f"origin_{int(radius)}m_purpose_opportunities"])

    def vector(self, origin_x, origin_y, home_x, home_y, work_x, work_y, has_work, purpose="other"):
        # A 250 m origin grid shares summaries across repeated synthetic locations.
        purpose = base_activity_purpose(purpose)
        key = (int(np.floor(origin_x / 250)), int(np.floor(origin_y / 250)), purpose)
        if key not in self.cache:
            center = (np.asarray(key[:2], dtype=float) + .5) * 250
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
                summaries.extend([len(indices), self.purpose_values[purpose][indices].sum()])
            if len(self.cache) >= 10000:
                self.cache.clear()
            self.cache[key] = np.asarray(summaries)
        nearest = self.tree.query([origin_x, origin_y])[0]
        home = np.hypot(origin_x - home_x, origin_y - home_y)
        work = np.hypot(origin_x - work_x, origin_y - work_y) if has_work else 0.0
        tree = self.purpose_trees[purpose]
        purpose_distance = tree.query([origin_x, origin_y])[0] if tree is not None else 0.
        return np.concatenate([[nearest, home, work, purpose_distance, float(tree is not None)], self.cache[key]]).astype(np.float32)



def base_activity_purpose(purpose):
    """Share facility offers between primary and secondary activity labels."""
    return str(purpose).removesuffix("_secondary")


def destination_offer_masks(destinations):
    """Use base offers, preserving the catalogue's existing home-visit offer."""
    masks = {}
    for purpose in ("other", "shop", "leisure", "work", "education", "home"):
        column = f"offers_{purpose}"
        # The catalogue stores home visits separately from household home anchors.
        if purpose == "home" and column not in destinations:
            column = "offers_home_secondary"
        if column not in destinations:
            raise ValueError(f"Missing destination offer column: {column}")
        masks[purpose] = destinations[column].fillna(False).to_numpy(dtype=bool)
    return masks


def destination_support_masks(candidate_ids, destination_index, purposes):
    """Mark cells with a real eligible facility and leave padding unsupported."""
    masks = {"statent": np.zeros(len(candidate_ids), dtype=bool)}
    for purpose in purposes:
        base = base_activity_purpose(purpose)
        pools = destination_index[base]
        supported = np.asarray([bool(pools.get(cell)) for cell in candidate_ids], dtype=bool)
        masks[base] = supported
        masks[str(purpose)] = supported
        masks["statent"] |= supported
    return masks


def require_purpose_mask(masks, purpose):
    """Never widen a purpose-specific choice set to arbitrary facilities."""
    base = base_activity_purpose(purpose)
    if base not in masks:
        raise ValueError(f"No destination support mask for purpose {purpose!r}")
    return masks[base]


def selector_origin_context(context):
    """Count purpose-eligible facilities in each coordinate leaf for origin context."""
    h3_data, levels, _ = context.stage('synthesis.population.spatial.secondary_nn.h3')
    destinations = context.stage('synthesis.population.destinations')
    masks = destination_offer_masks(destinations)
    assignments = h3_data['destinations'].set_index('destination_id')['level_2']
    counts = {}
    for purpose, mask in masks.items():
        leaves = assignments.reindex(destinations.loc[mask, 'destination_id'])
        counts[purpose] = leaves.value_counts()
    return OriginContext(levels['level_2'], radii=context.config('secondary_nn_selector_context_radii_m'),
                         purpose_counts=counts)
