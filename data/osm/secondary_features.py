"""Extract visitor destinations using access restrictions and feature-specific evidence."""

import logging
from pathlib import Path
import tempfile

import geopandas as gpd
import osmium
import pandas as pd
from shapely import from_wkb, make_valid
from shapely.ops import transform
from pyproj import Transformer

from synthesis.population.spatial.secondary_nn.tools import (
    classify_osm, derive_water_features, empty_features, osm_access_decision,
)

logger = logging.getLogger("synpp: osm.secondary_features")


def configure(context):
    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("secondary_nn_osm_water_path_buffer_m", 30.0)


def execute(context):

    # Read the configured Swiss extracts and their access restrictions.

    files = context.config("osm_file")
    files = files if isinstance(files, list) else [files]
    frames = []
    for name in files:
        # for the moment, we only process switzerland
        if "switzerland" not in name:
            continue
        # check if file exists
        path = Path(context.config("data_path")) / "osm" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        # append the features
        frames.append(read_features(path, context.path()))

    if not frames:
        raise ValueError("osm_file must include a Swiss extract with switzerland in its filename")

    features = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=2056)
    features = features.drop_duplicates(["osm_id", "kind"]).reset_index(drop=True)

    # Suppress separately tagged member polygons already represented by a relation.
    relations = features.loc[features.osm_id.str.startswith("r")]
    if len(relations):
        matches = gpd.sjoin(features, relations[["kind", "geometry"]], predicate="within", lsuffix="object", rsuffix="relation")
        duplicate = matches.loc[(matches.kind_object == matches.kind_relation)
                                & matches.osm_id.str.startswith("w")].index
        features = features.drop(index=duplicate.unique()).reset_index(drop=True)

    # Resolve visitor eligibility and exclude residential grounds before deriving waterside features.

    features = derive_water_features(features, float(context.config("secondary_nn_osm_water_path_buffer_m")))

    # Save eligible visitor features and access evidence for all model levels.

    features.to_parquet(Path(context.path()) / "secondary_features.parquet", index=False)
    logger.info("OSM secondary features by theme: %s", features.kind.value_counts().to_dict())
    logger.info("OSM visitor access evidence: %s", features.access_reason.value_counts().to_dict())
    return features


def read_features(path, cache_directory):
    factory = osmium.geom.WKBFactory()
    project = Transformer.from_crs(4326, 2056, always_xy=True).transform
    area_kinds = {"park", "garden", "water", "ski_area", "worship", "attraction"}
    rows = []
    failures = 0
    # Disk-backed node locations bound RAM while processing the Swiss extract.
    with tempfile.TemporaryDirectory(dir=cache_directory, prefix="osm_locations_") as directory:
        processor = (osmium.FileProcessor(str(path))
                     .with_locations(f"sparse_file_array,{directory}/nodes")
                     .with_areas(osmium.filter.KeyFilter("natural", "water", "landuse", "leisure", "tourism", "amenity", "piste:type"))
                     .with_filter(osmium.filter.KeyFilter("aerialway", "piste:type", "leisure", "tourism", "amenity", "natural", "water", "waterway", "landuse", "highway")))
        for obj in processor:
            if obj.is_relation():
                continue
            tags = dict(obj.tags)
            kinds = classify_osm(tags)
            if not kinds:
                continue
            try:
                if obj.is_node():
                    geometry = from_wkb(factory.create_point(obj))
                    osm_id = f"n{obj.id}"
                elif obj.is_area():
                    geometry = from_wkb(factory.create_multipolygon(obj))
                    osm_id = ("w" if obj.from_way() else "r") + str(obj.orig_id())
                elif obj.is_way():
                    # Areas are assembled separately, including multipolygon relations.
                    if obj.is_closed() and set(kinds).issubset(area_kinds):
                        continue
                    geometry = from_wkb(factory.create_linestring(obj))
                    osm_id = f"w{obj.id}"
                else:
                    continue
                geometry = make_valid(transform(project, geometry))
                if geometry.is_empty:
                    continue
                for kind in kinds:
                    # Mixed-tag closed ways may supply both a path and an area.
                    if obj.is_way() and obj.is_closed() and kind in area_kinds:
                        continue
                    # Area assembly also visits closed line features; retain their line representation.
                    if obj.is_area() and kind in ("aerialway", "piste", "path", "river"):
                        continue
                    # Use the original lake boundary for shoreline measurements.
                    measured = geometry.boundary if kind == "water" and geometry.geom_type in ("Polygon", "MultiPolygon") else geometry
                    if not measured.is_empty:
                        permitted, reason = osm_access_decision(tags, kind)
                        rows.append(dict(osm_id=osm_id, kind=kind, name=tags.get("name", ""),
                                         public=permitted, access_reason=reason,
                                         access=tags.get("access", ""), foot=tags.get("foot", ""),
                                         garden_type=tags.get("garden:type", tags.get("garden", "")),
                                         geometry=measured))
            except (RuntimeError, osmium.InvalidLocationError, ValueError):
                failures += 1
    logger.info("Extracted %s features from %s; %s invalid/incomplete geometries", len(rows), path.name, failures)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=2056) if rows else empty_features()
