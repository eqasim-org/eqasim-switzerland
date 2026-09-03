from matsim.readers import read_network
import numpy as np
import geopandas as gpd
import pandas as pd
from .geometry_io import safe_wkt_load
from shapely.geometry import LineString, MultiLineString
import logging
import os
import pickle
import time

from ..paths import (
    DETAILED_NETWORK_PATH_CONFIG,
    SIMULATION_PATH_CONFIG,
    configure_simulation_path,
    get_simulation_path,
)

logger = logging.getLogger("synpp")


def configure(context):
    configure_simulation_path(context)
    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", False)


def execute(context):  
    simulation_path = get_simulation_path(context)
    network_file = os.path.join(simulation_path, "output_network.xml.gz")

    if not os.path.exists(network_file) and not context.config(SIMULATION_PATH_CONFIG):
        network_file = os.path.join(
            context.config("output_path"),
            context.config("output_id"),
            context.config("output_prefix") + "network.xml.gz",
        )
        
    network_geometry_file = _resolve_detailed_network_path(context, simulation_path)
            
    assert os.path.exists(network_file), f"Network file not found at {network_file}"
    logger.info("\t LOADING NETWORK FROM: %s" % network_file)
    
    # create the network object, overwrite is set to True to ensure that the latest network is loaded when this stage is run again
    network = RoadNetwork(network_file, network_geometry_file, overwrite=True, cache_dir= context.path())

    return network


def _resolve_detailed_network_path(context, simulation_path):
    """Find the detailed pt2matsim geometry belonging to the simulation network."""
    configured_path = context.config(DETAILED_NETWORK_PATH_CONFIG)
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
    if not context.config(SIMULATION_PATH_CONFIG):
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

    logger.warning(
        "Detailed network geometry was requested but not found. Counts maps will "
        "fall back to straight MATSim node-to-node link geometries. Set %s to the "
        "pt2matsim detailed_network CSV to plot real road shapes.",
        DETAILED_NETWORK_PATH_CONFIG,
    )
    return None


class RoadNetwork:
    def __init__(self, network_file: str, geometry_file: str=None, overwrite=True, cache_dir:str="cache"):
        self.network_file = network_file
        self.geometry_file = geometry_file
        self._in_cache_class_file = os.path.join(cache_dir,"network.class.cache.pkl")
        
        start_time = time.time()
        if not overwrite and os.path.exists(self._in_cache_class_file):
            self._load_from_cache()
            logger.info(f"Loading from cache took {(time.time()-start_time):.1f} seconds.\n")
        else:
            # Load the network
            self.net = self._load_network()                
            # The links that were removed when removing unecessary nodes are replicated (usefull for matching later)
            self.replicate_merged_links()        
            # Load the geometry if it is given otherwise created geopandas
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
    
    def get_link(self, link_id, in_simulation_link=False):
        # in_simulation_link: whether the link that isconsidered in MATSim simulation
        # It can be different because some links are just merged
        link = self.links[self.links.link_id==link_id].iloc[0]
        if in_simulation_link:
            original_link = link.replicate_of
            if not pd.isna(original_link):
                link = self.links[self.links.link_id==original_link].iloc[0]        
        return link
    
    def get_link_geometry(self, link_id, in_simulation_link=False):
        geometries = self.get_link_geometries(
            [link_id], expand_merged=in_simulation_link
        ).geometry.tolist()
        if not geometries:
            return None

        parts = []
        for geometry in geometries:
            if isinstance(geometry, LineString):
                parts.append(geometry)
            elif isinstance(geometry, MultiLineString):
                parts.extend(geometry.geoms)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else MultiLineString(parts)

    def get_detailed_link_ids(self, link_ids):
        """Expand matched IDs to the complete detailed shape of simulation links.

        Network simplification keeps the first old link ID as the MATSim link ID
        and creates replicate_of rows for the remaining old IDs. Both forms must
        therefore resolve to the same ordered old_link_id chain.
        """
        requested_ids = list(dict.fromkeys(str(link_id) for link_id in link_ids))
        links = self.links
        link_ids_as_string = links["link_id"].astype(str)
        columns = ["link_id", "replicate_of", "attributes"]

        requested_links = links.loc[
            link_ids_as_string.isin(requested_ids), columns
        ].copy()
        requested_links["link_id"] = requested_links["link_id"].astype(str)
        requested_links = requested_links.drop_duplicates("link_id").set_index("link_id")

        simulation_ids = {
            str(link["replicate_of"])
            if pd.notna(link["replicate_of"])
            else link_id
            for link_id, link in requested_links.iterrows()
        }
        simulation_links = links.loc[
            link_ids_as_string.isin(simulation_ids), columns
        ].copy()
        simulation_links["link_id"] = simulation_links["link_id"].astype(str)
        simulation_links = (
            simulation_links.drop_duplicates("link_id").set_index("link_id")
        )

        detailed_ids = []
        for link_id in requested_ids:
            if link_id not in requested_links.index:
                logger.warning("Link %s is not present in the MATSim network.", link_id)
                continue

            link = requested_links.loc[link_id]
            simulation_link_id = (
                str(link["replicate_of"])
                if pd.notna(link["replicate_of"])
                else link_id
            )
            simulation_link = simulation_links.loc[simulation_link_id]
            attributes = simulation_link.get("attributes")
            old_link_ids = (
                attributes.get("old_link_id")
                if isinstance(attributes, dict)
                else None
            )
            detailed_ids.extend(
                str(old_link_ids).split("_") if old_link_ids else [simulation_link_id]
            )

        # Keep chain order while avoiding duplicate paths when several stations
        # are assigned to the same simplified simulation link.
        return list(dict.fromkeys(detailed_ids))

    def get_link_geometries(self, link_ids, expand_merged=False):
        """Return real link shapes, optionally expanding simplified MATSim links."""
        requested_ids = [str(link_id) for link_id in link_ids]
        geometry_ids = (
            self.get_detailed_link_ids(requested_ids)
            if expand_merged
            else list(dict.fromkeys(requested_ids))
        )

        geometry = self.get_geometry()
        link_ids_as_string = geometry["link_id"].astype(str)
        selected = geometry.loc[link_ids_as_string.isin(geometry_ids)].copy()
        selected["link_id"] = selected["link_id"].astype(str)
        selected = selected.drop_duplicates("link_id")
        geometry_order = {
            link_id: index for index, link_id in enumerate(geometry_ids)
        }
        selected["_geometry_order"] = selected["link_id"].map(geometry_order)
        selected = selected.sort_values("_geometry_order")

        missing_ids = set(geometry_ids).difference(selected["link_id"])
        if missing_ids:
            logger.warning(
                "%d detailed link geometries are unavailable; using available "
                "simulation-link geometry as a fallback.",
                len(missing_ids),
            )
            simulation_ids = self.get_in_simulation_links(requested_ids)
            simulation_ids = [
                str(link_id) for link_id in simulation_ids if pd.notna(link_id)
            ]
            fallback = geometry.loc[
                link_ids_as_string.isin(simulation_ids)
                & ~link_ids_as_string.isin(selected["link_id"])
            ].copy()
            fallback["link_id"] = fallback["link_id"].astype(str)
            fallback_order = {
                link_id: len(geometry_order) + index
                for index, link_id in enumerate(simulation_ids)
            }
            fallback["_geometry_order"] = fallback["link_id"].map(fallback_order)
            selected = pd.concat([selected, fallback], ignore_index=True)

        selected = selected.sort_values("_geometry_order")
        selected = selected.drop(columns="_geometry_order", errors="ignore")
        return gpd.GeoDataFrame(selected, geometry="geometry", crs=geometry.crs)

    def get_bearing(self, link_id, in_simulation_link=False):
        geo = self.get_link_geometry(link_id, in_simulation_link)
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
       
    
    def _load_network(self, file_path: str=None):
        if file_path is None:
            file_path = self.network_file
        
        logger.info(f"Reading network file: {file_path}")
        net = read_network(file_path)
        
        if "attributes" not in net.links and len(net.link_attrs):
            link_attrs = net.link_attrs.groupby('link_id') \
                .apply(lambda x: dict(zip(x['name'], x['value']))) \
                .reset_index(name='attributes')
            net.links = net.links.merge(link_attrs, on="link_id", how="left")
            net.links.loc[net.links["attributes"].isna(), "attributes"] = None

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
    
    def read_network_geometry(self, file_path: str=None, crs='EPSG:2056') -> gpd.GeoDataFrame:
        if file_path is None:
            file_path = self.geometry_file
        
        if file_path:
            logger.info(f"Reading geometry from file: {file_path}")
            df = pd.read_csv(file_path)
            df['Geometry'] = df['Geometry'].apply(safe_wkt_load)
            df = df.dropna(subset=['Geometry']).rename(columns={"Geometry": "geometry", "LinkId": "link_id"})
            df["link_id"] = df["link_id"].astype(str)
            return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
        else:
            logger.info("The geometry is created from links and nodes coordinates")
            return self.net.as_geo(projection=crs)
        
    def add_attribute_to_geometry(self, attribute:str):
        att = self.net.links[["link_id", attribute]]
        
        # Double check
        att.loc[:, "link_id"] = att["link_id"].astype(str)
        self.net_geo.loc[:, "link_id"] = self.net_geo["link_id"].astype(str)
        
        #Merge
        self.net_geo = self.net_geo.merge(att, on="link_id", how="left")
        
    def replicate_merged_links(self):
        # Extract relevant columns
        df = self.links[["link_id", "attributes"]].copy()
        
        # Extract and filter old_link_id
        df["old_link_id"] = df["attributes"].map(lambda x: x.get("old_link_id", None) if isinstance(x, dict) else None)
        df = df[df["old_link_id"].notna()]
        df["old_link_id"] = df["old_link_id"].str.split("_")
        
        # Assertion
        assert all(df["link_id"] == df["old_link_id"].str[0]), "Old link id may be corrupted!"
        
        
        # Create DataFrame of replicate link_ids
        df["replicate_ids"] = df["old_link_id"].apply(lambda x: x[1:] if len(x) > 1 else [])
        df_repl = df[["link_id", "replicate_ids"]].explode("replicate_ids")
        df_repl = df_repl[df_repl["replicate_ids"].notna()].copy()
        
        # Get rows to replicate from original DataFrame
        df_links = self.links.set_index("link_id")
        base_rows = df_links.loc[df_repl["link_id"]].reset_index()
        base_rows["replicate_of"] = base_rows["link_id"]
        base_rows["link_id"] = df_repl["replicate_ids"].values
        
        # Combine original and replicated
        df_links = df_links.reset_index()
        
        df_links["replicate_of"]  = np.nan        
        
        df_links = pd.concat([df_links, base_rows], ignore_index=True)

        self.net.links = df_links            
    
    def get_in_simulation_link(self, link_id:list):
        """This function return the  links that are used in the simulation, 
        in case the link is just a replicate of another one that was removed during network cleaning"""
        replicate_of = self.links[self.links.link_id==link_id].iloc[0].replicate_of        
        if not pd.isna(replicate_of):
            return replicate_of
        else:
            return link_id
    
    def get_in_simulation_links(self, link_ids: list):
        """
        Efficiently returns the original simulation links for a list of link_ids.
        If a link was a replicate, returns its original link (replicate_of),
        otherwise returns the link_id itself.
        """
        # Create a DataFrame for input link_ids
        link_df = pd.DataFrame({'link_id': link_ids})
        
        # Merge with self.links to get the 'replicate_of' info
        merged = link_df.merge(
            self.links[['link_id', 'replicate_of']],
            on='link_id',
            how='left'
        )
    
        # Use replicate_of if it exists, else use original link_id
        merged['simulation_link'] = merged['replicate_of'].where(
            ~merged['replicate_of'].isna(),
            merged['link_id']
        )
    
        return merged['simulation_link'].tolist()
