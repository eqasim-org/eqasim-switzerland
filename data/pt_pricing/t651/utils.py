import geopandas as gpd
import shapely.geometry as geo
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np

def alpha_shape(points, alpha):
    if len(points) < 4:
        return None

    coords = np.array([[p.x, p.y] for p in points])
    tri = Delaunay(coords)

    triangles = coords[tri.simplices]
    a = []
    for triangle in triangles:
        pa, pb, pc = triangle
        ab = np.linalg.norm(pa - pb)
        bc = np.linalg.norm(pb - pc)
        ca = np.linalg.norm(pc - pa)
        s = (ab + bc + ca) / 2.0
        area = max(s * (s - ab) * (s - bc) * (s - ca), 0)
        if area == 0:
            continue
        circum_r = ab * bc * ca / (4.0 * np.sqrt(area))
        if circum_r < 1.0 / alpha:
            a.append([(pa[0], pa[1]), (pb[0], pb[1])])
            a.append([(pb[0], pb[1]), (pc[0], pc[1])])
            a.append([(pc[0], pc[1]), (pa[0], pa[1])])

    m = geo.MultiLineString([tuple(p) for p in a])
    triangles = list(polygonize(m))
    return unary_union(triangles)


def compute_alpha_shape(group, alpha=1000):
    points = list(group.values)
    return alpha_shape(points, alpha)


def extract_polygon(geom):
    if isinstance(geom, (geo.Polygon, geo.MultiPolygon)):
        return geom
    elif isinstance(geom, geo.GeometryCollection):
        # Filter only Polygon-like geometries
        polys = [g for g in geom.geoms if isinstance(g, (geo.Polygon, geo.MultiPolygon))]
        if len(polys) == 1:
            return polys[0]
        elif len(polys) > 1:
            return geo.MultiPolygon(polys)
    return None 


def create_shapes(gtfs_networks, spatial_zones):
    gtfs_found          = gtfs_networks[gtfs_networks["zones"].notna()]

    df_shapes = gpd.GeoDataFrame(gtfs_found.groupby(
        ["tarif network", "local network", "zones"]
    )["geometry"].agg(
        lambda group: compute_alpha_shape(group.drop_duplicates(), alpha=0.0005)
    ).reset_index(name="geometry")).dropna()

    df_shapes["geometry"] = df_shapes["geometry"].apply(extract_polygon)
    df_shapes.crs = {"init" : "EPSG:2056"}

    for authority, zones in spatial_zones.items():
        for zone_id, zone_shape in zones.items():
            df_shapes.loc[(df_shapes["tarif network"]==authority) & (df_shapes["zones"]==zone_id), "geometry"] = zone_shape

    df_shapes.to_file("/home/asallard/Documents/WP4/Pricing/Switzerland_pt_zones/output/shp/zones_pt.shp")

    return df_shapes