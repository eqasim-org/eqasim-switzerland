"""The fare zones abroad: the communes of the list of zones abroad and the zones of the stops in them.

The zone polygons themselves are made by the stage data.pt_pricing.zones (from the municipalities, the zones of the
stops and these communes abroad). The zones abroad (the French zones of Unireso) come from a list of communes and their
zones (unireso_france_zones.csv: insee, commune, zone, and optionally network, else Unireso) and the outlines of these
communes (communes_france.gpkg: insee, name, geometry), in the Swiss projection (LV95, EPSG:2056).
"""
import logging

import geopandas as gpd
import pandas as pd

logger = logging.getLogger("synpp")

FOREIGN_NETWORK = "Unireso"   # the network of the communes abroad when their list has no network column


def foreign_communes(communes_path, zones_path):
    """The communes abroad and their zones, one row per commune and zone: id ("FR" and the INSEE code), name, network,
    zone and geometry (EPSG:2056); empty without both files. A listed commune without an outline is left out, with a
    warning."""
    if not (communes_path and zones_path):
        return gpd.GeoDataFrame({"id": [], "name": [], "network": [], "zone": []}, geometry = [], crs = 2056)
    listed = pd.read_csv(zones_path, dtype = str).apply(lambda s: s.str.strip())
    if "network" not in listed:
        listed["network"] = FOREIGN_NETWORK
    communes = gpd.read_file(communes_path).to_crs(2056)[["insee", "name", "geometry"]]
    missing = sorted(set(listed["insee"]) - set(communes["insee"]))
    if missing:
        logger.warning(f"  {len(missing)} communes of {zones_path} have no outline in {communes_path}: {', '.join(missing)}")
    result = communes.merge(listed[["insee", "network", "zone"]], on = "insee")
    result["id"] = "FR" + result["insee"]
    return result[["id", "name", "network", "zone", "geometry"]]


def foreign_stop_zones(net, zones, communes_path, zones_path):
    """The zones of the stops in the communes abroad, as a DataFrame stop_id, network, zone: the zone of the commune of
    each stop, or, for a commune listed in several zones, the one whose polygon (of zones, the zone polygons) holds the
    stop (else the first listed). Empty without both files."""
    abroad = foreign_communes(communes_path, zones_path)
    if not len(abroad):
        return pd.DataFrame({"stop_id": [], "network": [], "zone": []})
    points = gpd.GeoDataFrame({"stop_id": net.stop_ids.astype(str)},
                              geometry = gpd.GeoSeries(gpd.points_from_xy(net.stop_lon, net.stop_lat), crs = 4326).to_crs(2056).values, crs = 2056)
    hits = gpd.sjoin(points, abroad[["network", "zone", "geometry"]], predicate = "within").drop(columns = "index_right")
    several = hits.duplicated(["stop_id", "network"], keep = False)
    if several.any():   # a commune in several zones: the zone whose polygon holds the stop
        inside = gpd.sjoin(hits[several].drop(columns = "zone"), zones[["network", "zone", "geometry"]], predicate = "within")
        inside = inside[inside["network_left"] == inside["network_right"]].rename(columns = {"network_left": "network"})
        chosen = hits[several].merge(inside[["stop_id", "network", "zone"]], on = ["stop_id", "network", "zone"])
        first = hits[several].sort_values("zone").drop_duplicates(["stop_id", "network"])
        hits = pd.concat([hits[~several], chosen, first]).drop_duplicates(["stop_id", "network"])
    return pd.DataFrame(hits[["stop_id", "network", "zone"]]).reset_index(drop = True)
