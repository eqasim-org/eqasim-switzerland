"""Keep directed detailed shapes aligned with network cleaning and contraction."""
import csv
import logging
from math import dist
from pathlib import Path

from shapely import from_wkt, to_wkt, linestrings, is_geometry, get_type_id, is_empty, get_coordinates
from shapely.geometry import LineString
import numpy as np
import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)
GEOMETRY_COLUMN = "_detailed_geometry"
ENDPOINT_TOLERANCE = 0.1  # Same tolerance (metres) as pt2matsim LinkGeometryIndex.


def _aligned(shape, start, end, link_id):
    """Validate the CRS/direction, then snap exporter rounding to network nodes."""
    coords = [(x, y) for x, y, *_ in shape.coords]
    if (dist(coords[0], start) > ENDPOINT_TOLERANCE
            or dist(coords[-1], end) > ENDPOINT_TOLERANCE):
        raise ValueError(f"Detailed geometry endpoints/direction do not match link {link_id}; "
                         "use the geometry exported with this network and its CRS")
    coords[0], coords[-1] = start, end
    return LineString(coords)


def attach_link_geometries(network, path):
    """Read the original export once before cleaning, preserving intermediate points.

    Missing/degenerate exporter rows use original link endpoints, matching the
    mapper's fallback. Valid shapes with wrong endpoints fail rather than hiding
    a mismatched network or CRS. Geometry travels with its dataframe row through
    filtering and merging, so removed links cannot reappear in the output CSV.
    """
    wanted = set(network.links.link_id.astype(str))
    shapes = {}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"LinkId", "Geometry"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Geometry CSV needs LinkId and Geometry columns: {path}")
        for row in reader:
            link_id = row["LinkId"]
            if link_id not in wanted:
                continue
            if link_id in shapes:
                raise ValueError(f"Duplicate detailed geometry for link {link_id}")
            shape = from_wkt(row["Geometry"], on_invalid="ignore")
            shapes[link_id] = shape if isinstance(shape, LineString) and not shape.is_empty else None
    nodes = {str(row.node_id): (float(row.x), float(row.y))
             for row in network.nodes[["node_id", "x", "y"]].itertuples(index=False)}
    geometries = []
    fallback_count = 0
    for row in network.links[["link_id", "from_node", "to_node"]].itertuples(index=False):
        start, end = nodes[str(row.from_node)], nodes[str(row.to_node)]
        shape = shapes.pop(str(row.link_id), None)
        if shape is None:
            shape = LineString([start, end])
            fallback_count += 1
        geometries.append(_aligned(shape, start, end, row.link_id))
    network.links[GEOMETRY_COLUMN] = geometries
    logger.info("Loaded detailed link geometries; %d links use endpoint fallback.", fallback_count)


def merge_link_geometries(shapes):
    """Concatenate shapes in directed chain order, keeping every intermediate bend."""
    coords = []
    for shape in shapes:
        part = list(shape.coords)
        if coords:
            if dist(coords[-1], part[0]) > ENDPOINT_TOLERANCE:
                raise ValueError("Cannot merge detailed geometries: directed chain is discontinuous")
            coords.extend(part[1:])
        else:
            coords.extend(part)
    return LineString(coords)


def write_link_geometries(network, path):
    """Validate and export one directed geometry per final link, atomically."""
    nodes = {str(row.node_id): (float(row.x), float(row.y))
             for row in network.nodes[["node_id", "x", "y"]].itertuples(index=False)}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["LinkId", "Geometry"])
            columns = ["link_id", "from_node", "to_node", GEOMETRY_COLUMN]
            for link_id, source, target, shape in network.links[columns].itertuples(index=False, name=None):
                shape = _aligned(shape, nodes[str(source)], nodes[str(target)], link_id)
                writer.writerow([link_id, to_wkt(shape, rounding_precision=-1)])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_link_geometries_with_additions(source_path, output_path, new_shapes, reversed_links=None):
    """Preserve mapped shapes and append new PT links without loading the whole CSV."""
    shapes = dict(new_shapes)
    reversed_links = reversed_links or {}
    with open(source_path, newline="", encoding="utf-8-sig") as source, \
            open(output_path, "w", newline="", encoding="utf-8") as output:
        reader = csv.DictReader(source)
        if not {"LinkId", "Geometry"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Geometry CSV needs LinkId and Geometry columns: {source_path}")
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            writer.writerow(row)
            reverse_id = reversed_links.get(row["LinkId"])
            if reverse_id in shapes:
                shape = from_wkt(row["Geometry"], on_invalid="ignore")
                if isinstance(shape, LineString) and not shape.is_empty:
                    shapes[reverse_id] = LineString(list(shape.coords)[::-1])
        for link_id, shape in shapes.items():
            writer.writerow({"LinkId": link_id, "Geometry": to_wkt(shape, rounding_precision=-1)})



def get_network_geometry(network, links=None):
    """Return directed shapes in LV95, preserving subset row order and labels.

    Detailed curves take precedence. Only missing shapes use node endpoints;
    an invalid supplied shape raises rather than silently flattening a curve.
    """
    links = network.links if links is None else links
    crs = getattr(network, "network_attrs", {}).get("coordinateReferenceSystem") or "EPSG:2056"
    if links.empty:
        return gpd.GeoSeries([], index=links.index, crs=crs).to_crs(2056)
    shapes = links[GEOMETRY_COLUMN].tolist() if GEOMETRY_COLUMN in links else [None] * len(links)
    missing = np.array([shape is None or shape is pd.NA or (
        isinstance(shape, float) and np.isnan(shape)) for shape in shapes])
    if missing.any():
        nodes = network.nodes[["node_id", "x", "y"]].copy()
        nodes["node_id"] = nodes.node_id.astype(str)
        if nodes.node_id.duplicated().any():
            raise ValueError("Cannot construct link geometry: duplicate network node IDs")
        nodes = nodes.set_index("node_id")[["x", "y"]]
        selected = links.iloc[np.flatnonzero(missing)]
        try:
            endpoints = np.stack([nodes.loc[selected.from_node.astype(str)].to_numpy(dtype=float),
                                  nodes.loc[selected.to_node.astype(str)].to_numpy(dtype=float)], axis=1)
        except KeyError as error:
            raise ValueError("Cannot construct link geometry: missing endpoint node") from error
        for position, shape in zip(np.flatnonzero(missing), linestrings(endpoints)):
            shapes[position] = shape
    shapes = np.asarray(shapes, dtype=object)
    invalid = ~is_geometry(shapes)
    if not invalid.any():
        invalid = (get_type_id(shapes) != 1) | is_empty(shapes)
    if invalid.any():
        raise ValueError(f"Invalid directed geometry for link {links.link_id.iloc[np.flatnonzero(invalid)[0]]}")
    coordinates, positions = get_coordinates(shapes, return_index=True)
    finite = np.isfinite(coordinates).all(axis=1)
    if not finite.all():
        raise ValueError(f"Invalid directed geometry for link {links.link_id.iloc[positions[~finite][0]]}")
    return gpd.GeoSeries(shapes, index=links.index, crs=crs).to_crs(2056)
