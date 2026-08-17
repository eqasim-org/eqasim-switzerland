"""Input helpers for optional detailed network geometry files."""

from shapely import wkt
from shapely.errors import ShapelyError


def safe_wkt_load(value):
    try:
        geometry = wkt.loads(value)
        if geometry.geom_type == "LineString" and len(geometry.coords) > 1:
            return geometry
    except (ShapelyError, AttributeError, TypeError):
        pass
    return None
