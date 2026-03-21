from matsim.readers import read_network
import numpy as np
import geopandas as gpd
import pandas as pd
from .utils.geometry_utils import safe_wkt_load
from .utils import Functions as F
from shapely.geometry import LineString
import logging
import os
import pickle
import time

logger = logging.getLogger("synpp")


def configure(context):
    # context.stage("matsim.output")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("output_prefix", "switzerland_")

def execute(context):  
    # this part ensure dependency (this stage run after matsim.output)
    # _ = context.stage("matsim.output")      

    output_path = context.config("output_path")
    output_id   = context.config("output_id")
    simulation_directory = context.config("simulation_directory")
    
    network_file = os.path.join(output_path, output_id, simulation_directory, "output_network.xml.gz")
    if not os.path.exists(network_file):
        network_file = os.path.join(output_path, output_id, context.config("output_prefix") + "network.xml.gz")
    assert os.path.exists(network_file), f"Network file not found at {network_file}"

    network_geometry_file = os.path.join(output_path, output_id, 
                                         "%sdetailed_network.csv" % context.config("output_prefix"))
    
    # create the network object, overwrite is set to True to ensure that the latest network is loaded when this stage is run again
    network = Network(network_file, network_geometry_file, overwrite=True, cache_dir= context.path())

    return network


class Network:
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
        if not in_simulation_link:
            geo = self.get_geometry()
            return geo[geo["link_id"]==link_id].geometry
        else:
            link = self.links[self.links["link_id"]==link_id].iloc[0] 
            replicate_of = link.replicate_of
            if pd.isna(replicate_of):
                return self.get_link_geometry(link_id,in_simulation_link=False)
            else:
                links = link["attributes"]["old_link_id"].split('_')
                first_link_geom = self.get_link_geometry(links[0]  ,in_simulation_link=False)
                last_link_geom  = self.get_link_geometry(links[-1] ,in_simulation_link=False)
                
                return LineString([first_link_geom.coords[0], last_link_geom.coords[1]])
    
    def get_bearing(self, link_id, in_simulation_link=False):
        geo = self.get_link_geometry(link_id, in_simulation_link)
        return F.compute_bearing(geo)
                       
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
            
        net.links["highway"] = net.links.attributes.apply(
            lambda x: x["osm:way:highway"] if (isinstance(x,dict) and "osm:way:highway" in x) else np.nan
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