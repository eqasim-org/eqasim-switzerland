"""Cache the shared Swiss boundary for pre-assignment demand clipping."""

from pathlib import Path

import shapely
from pyproj import Transformer
from shapely.ops import transform

from sumo.tables import as_polars


def configure(context):
    context.stage("data.spatial.swiss_border")


def execute(context):

    # Convert the external geometry once and keep the clipping coordinates metric.

    source = context.stage("data.spatial.swiss_border")
    crs = getattr(source, "crs", None)
    if crs is None:
        raise ValueError("The Swiss boundary must declare its coordinate reference system")
    frame = as_polars(source.to_frame(name="geometry"))
    polygons = shapely.from_wkb(frame["geometry"].to_numpy())
    if not len(polygons) or not shapely.is_valid(polygons).all() or shapely.is_empty(polygons).any():
        raise ValueError("The Swiss boundary must contain valid, nonempty polygons")
    region = transform(Transformer.from_crs(crs, "EPSG:2056", always_xy=True).transform,
                       shapely.union_all(polygons))
    if region.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("The Swiss boundary must be polygonal")

    # Return a small geometry manifest instead of a pickled geospatial table.

    output = Path(context.path()).resolve() / "switzerland.wkb"
    output.write_bytes(shapely.to_wkb(region))
    return {"geometry": str(output), "crs": "EPSG:2056"}
