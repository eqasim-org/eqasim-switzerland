from matsim.readers import read_network
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiLineString
from shapely import from_wkt, get_point, get_x, get_y
import logging
import os
import pickle
import time
from ..paths import  configure_simulation_path, get_simulation_path

logger = logging.getLogger("synpp")

def configure(context):
    configure_simulation_path(context)
    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", True)

def execute(context):
    simulation_path = get_simulation_path(context)
    network_file = os.path.join(simulation_path, "output_network.xml.gz")

    if not os.path.exists(network_file) and not context.config("analysis.counts.simulation_path"):
        network_file = os.path.join(
            context.config("output_path"),
            context.config("output_id"),
            context.config("output_prefix") + "network.xml.gz",
        )

    network_geometry_file = _resolve_detailed_network_path(context, simulation_path)

    assert os.path.exists(network_file), f"Network file not found at {network_file}"
    logger.info("\t LOADING NETWORK FROM: %s" % network_file)

    # create the network object, overwrite is set to True to ensure that the latest network is loaded when this stage is run again
    network = RoadNetwork(network_file, network_geometry_file, overwrite=True, cache_dir= context.path(), context=context)

    return network


class RoadNetwork:
    def __init__(self, network_file: str, geometry_file: str=None, overwrite=True, cache_dir:str="cache", context=None):
        self.network_file = network_file
        self.geometry_file = geometry_file
        self._in_cache_class_file = os.path.join(cache_dir,"network.direct_ids.cache.pkl")

        start_time = time.time()
        if not overwrite and os.path.exists(self._in_cache_class_file):
            self._load_from_cache()
            logger.info(f"Loading from cache took {(time.time()-start_time):.1f} seconds.\n")
        else:
            # Load the network
            self.net = self._load_network(context=context)
            self.net_geo = self.read_network_geometry(geometry_file)
            self._save_to_cache()
            logger.info(f"Reading and processing took {(time.time()-start_time):.1f} seconds.\n")

    def _save_to_cache(self):
        logger.info("Saving network to cache...")
        with open(self._in_cache_class_file, 'wb') as f:
            pickle.dump(self, f)

    def _load_from_cache(self):
        logger.info("Loading network from cache...")
        with open(self._in_cache_class_file, 'rb') as f:
            cached = pickle.load(f)
            self.__dict__.update(cached.__dict__)

    def get_geometry(self):
        if not hasattr(self, "net_geo"):
            self.net_geo = self.read_network_geometry()
        return self.net_geo

    @property
    def links(self):
        return self.net.links

    @property
    def nodes(self):
        return self.net.nodes

    @property
    def link_attrs(self):
        return self.net.link_attrs

    def get_link(self, link_id):
        return self.links.loc[self.links.link_id == str(link_id)].iloc[0]

    def get_link_geometry(self, link_id):
        geometries = self.get_link_geometries([link_id])
        return None if geometries.empty else geometries.geometry.iloc[0]

    def get_link_geometries(self, link_ids):
        """Select current network shapes by their actual simulation link IDs."""
        ids = list(dict.fromkeys(str(link_id) for link_id in link_ids))
        geometry = self.get_geometry().set_index("link_id", drop=False)
        missing = set(ids) - set(geometry.index)
        if missing:
            raise KeyError(f"Unknown network geometry IDs: {sorted(missing)[:10]}")
        return geometry.loc[ids].reset_index(drop=True)

    def get_bearing(self, link_id):
        geo = self.get_link_geometry(link_id)
        if geo is None or geo.is_empty:
            return np.nan
        if isinstance(geo, MultiLineString):
            start = geo.geoms[0].coords[0]
            end = geo.geoms[-1].coords[-1]
        else:
            start = geo.coords[0]
            end = geo.coords[-1]
        return (np.degrees(np.arctan2(end[0] - start[0], end[1] - start[1])) + 360) % 360

    def plot(self, *args, **kwargs):
        return self.get_geometry().plot(*args, **kwargs)

    @property
    def geometry(self):
        return self.get_geometry().geometry

    @property
    def crs(self):
        return self.get_geometry().crs

    def _load_network(self, file_path: str=None, context=None):
        if file_path is None:
            file_path = self.network_file

        logger.info(f"Reading network file: {file_path}")
        net = read_network(file_path, context=context)

        if "attributes" not in net.links and len(net.link_attrs):
            net.put_attributes_in_links()

        net.filter_car_links(inplace=True)

        net.links["highway"] = net.links.attributes.apply(
            lambda x: x["osm:way:highway"] if (isinstance(x,dict) and "osm:way:highway" in x) else np.nan
        )
        net.links["osm_id"] = net.links.attributes.apply(
                    lambda x: x['osm:way:id'] if (isinstance(x,dict) and 'osm:way:id' in x) else np.nan
                )
        net.links["link_id"] = net.links["link_id"].astype(str)
        return net

    def get_ways(self, geo_df: gpd.GeoDataFrame=None, road_types=None):
        if geo_df is None:
            geo_df = self.get_geometry()
        if road_types is None:
            road_types = ["motorway", "trunk", "primary", "secondary"]

        motorway_ids = self.net.links[self.net.links.highway.isin(road_types)].link_id
        return geo_df[geo_df.link_id.astype(str).isin(motorway_ids)]

    def read_network_geometry(self, file_path=None, crs='EPSG:2056'):
        """Load the paired detailed export; never reconstruct old link histories."""
        file_path = file_path or self.geometry_file
        if not file_path:
            raise ValueError("Count matching requires the detailed geometry exported with its network")
        df = pd.read_csv(file_path, dtype={"LinkId": str}, usecols=["LinkId", "Geometry"])
        df = df.rename(columns={"LinkId": "link_id"})
        df = df.loc[df.link_id.isin(self.links.link_id)].copy()
        if df.link_id.duplicated().any():
            raise ValueError("Duplicate link IDs in detailed network geometry")
        missing = set(self.links.link_id) - set(df.link_id)
        if missing:
            raise ValueError(f"Detailed geometry is missing {len(missing)} network links: {sorted(missing)[:10]}")
        df['geometry'] = from_wkt(df.pop('Geometry').to_numpy(), on_invalid='raise')
        geometry = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
        if (geometry.geometry.is_empty | geometry.geometry.isna() | (geometry.geom_type != 'LineString')).any():
            raise ValueError("Detailed network geometry contains empty or non-LineString shapes")
        # The final split fragment retains its parent's ID; detect stale full shapes too.
        links = self.links.set_index('link_id').loc[geometry.link_id]
        nodes = self.nodes.set_index('node_id')
        for column, endpoint in [('from_node', 0), ('to_node', -1)]:
            xy = nodes.loc[links[column], ['x', 'y']].to_numpy(dtype=float)
            points = get_point(geometry.geometry.to_numpy(), endpoint)
            distances = np.hypot(get_x(points) - xy[:, 0], get_y(points) - xy[:, 1])
            if (~np.isfinite(distances) | (distances > 0.1)).any():
                raise ValueError("Detailed geometry endpoints do not match the network; use its matching mapped export")
        return geometry.reset_index(drop=True)

    def add_attribute_to_geometry(self, attribute:str):
        att = self.net.links[["link_id", attribute]]

        # Double check
        att.loc[:, "link_id"] = att["link_id"].astype(str)
        self.net_geo.loc[:, "link_id"] = self.net_geo["link_id"].astype(str)

        #Merge
        self.net_geo = self.net_geo.merge(att, on="link_id", how="left")



def _resolve_detailed_network_path(context, simulation_path):
    """Find the detailed pt2matsim geometry belonging to the simulation network."""
    configured_path = context.config("analysis.counts.detailed_network_path")
    if configured_path:
        configured_path = os.path.abspath(os.path.expanduser(configured_path))
        if not os.path.exists(configured_path):
            raise FileNotFoundError(
                f"Configured detailed network geometry not found at {configured_path}"
            )
        return configured_path

    if not context.config("export_detailed_network"):
        return None

    filename = f"{context.config('output_prefix')}detailed_network.csv"
    candidates = [
        os.path.join(os.path.dirname(simulation_path), filename),
        os.path.join(simulation_path, filename),
        os.path.join(os.path.dirname(simulation_path), "detailed_network.csv"),
    ]
    if not context.config("analysis.counts.simulation_path"):
        candidates.append(
            os.path.join(
                context.config("output_path"),
                context.config("output_id"),
                filename,
            )
        )
    for candidate in dict.fromkeys(candidates):
        if os.path.exists(candidate):
            logger.info("\t LOADING DETAILED NETWORK GEOMETRY FROM: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Detailed geometry for this network was not found. Set "
        "analysis.counts.detailed_network_path to its matching mapped export."
    )
