import geopandas as gpd
import shapely.geometry as geo
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import logging

logger = logging.getLogger("synpp")
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

    return df_shapes


def add_zpass_from_zones(gtfs_network):

    logger.info("Starting to create the Z-Pass network")

    awelle_zvv = {
        "ZVV": [116, 115, 114, 113, 112, 111, 110, 117, 118, 
                121, 120, 123, 122, 124,
                130, 131, 132, 133, 134, 135, 140, 141, 142, 143,
                150, 151, 152, 153, 154, 155, 156, 160, 161, 162, 163, 164, 170, 171, 172, 173, 180, 181, 182, 183, 184],
        "Awelle": [560, 561, 562, 563, 564, 565, 550, 551, 552, 570, 571, 572, 573, 574, 530, 531, 532, 533, 534, 535,
                   510, 511, 512, 513, 514, 518, 522]
    }

    szzg_zvv = {
        "ZVV": [116, 115, 114, 113, 112, 111, 110, 117, 118, 121, 120, 123, 124, 122, 130, 131, 132, 133, 134, 135, 140, 141, 142, 143,
                150, 151, 152, 153, 154, 155, 156, 160, 161, 162, 163, 164, 170, 171, 172, 173, 180, 181, 182, 183, 184],
        "ZVB": [610, 611, 612, 613, 621, 622, 623, 624, 625, 626, 631, 632, 633, 636, 637, 638, 639],
        "TVSZ": [670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688,
                 689, 691, 692]
    }

    ostwind_zvv = {
        "ZVV": [116, 115, 114, 113, 112, 111, 110, 117, 118, 121, 120, 123, 124, 122, 130, 131, 132, 133, 134, 135, 140, 141, 142, 143,
                150, 151, 152, 153, 154, 155, 156, 160, 161, 162, 163, 164, 170, 171, 172, 173, 180, 181, 182, 183, 184],
        "Ostwind": [810, 
                    820, 821, 822, 
                    830, 833, 834, 835, 837, 838, 
                    840, 845, 847, 848,
                    959, 958, 953, 954,
                    920, 921, 922, 923, 924, 925,
                    916, 915, 917, 918, 919,
                    974, 975, 976, 977,
                    990, 991, 992, 993, 994, 995, 996, 997, 998, 999,
                    901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912
                    ]
    }

    networks = [awelle_zvv, szzg_zvv, ostwind_zvv]
    networks_names = ["ZPassAwelle", "ZPassSchwyzZug", "ZPassOstwind"]
    networks_zones = {}

    zone_to_rows = {}

    for idx, zones_str in gtfs_network["zones"].dropna().items():
        for z in zones_str.split(", "):   
            zone_to_rows.setdefault(z, []).append(idx)

    for i in range(3):
        network           = networks[i]
        network_name      = networks_names[i]
        network_zone_list = []
        for key, zonelist in network.items():
            network_zone_list += [key + ":" + str(zone) for zone in zonelist]

        networks_zones[network_name] = network_zone_list

    for netw_name, zone_list in networks_zones.items():
        for thezone in zone_list:
            prefix  = thezone.split(":")[0]
            zone_nb = thezone.split(":")[1]
            new_zone_id = netw_name + ":" + zone_nb
            local_network_id = "ZPass" + prefix

            rows = zone_to_rows.get(thezone, [])
            if not rows:
                continue

            gtfs_network.loc[rows, "tarif network"] = gtfs_network.loc[rows, "tarif network"].apply(lambda x: list(set(x + [netw_name])) )
            gtfs_network.loc[rows, "local network"] = gtfs_network.loc[rows, "local network"].apply(lambda x: list(set(x + [local_network_id])))
            gtfs_network.loc[rows, "zones"]         = gtfs_network.loc[rows, "zones"].apply(lambda x: ", ".join(dict.fromkeys(x.split(", ") + [new_zone_id])))

        logger.info(f"{netw_name} processed")

    return gtfs_network