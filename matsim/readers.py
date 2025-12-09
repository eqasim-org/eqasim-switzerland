# -*- coding: utf-8 -*-

import xopen
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import pandas as pd
from matsim.writers import NetworkWriter
import gzip
import multiprocessing
import io

import logging
logger = logging.getLogger(__name__)

"""
It is partially based on the implementation from:
https://github.com/matsim-vsp/matsim-python-tools/blob/master/matsim/Network.py

Additional network readers may be integrated in the future to improve reading efficiency.
"""

class Network:

    _crsTag = 'coordinateReferenceSystem'

    def __init__(self, nodes, links, node_attrs, link_attrs, net_attrs=None):
        self.nodes = nodes
        self.links = links
        self.link_attrs = link_attrs
        self.node_attrs = node_attrs

        self.network_attrs = {}
        if net_attrs: self.network_attrs = net_attrs
        
        # This will add attributes as a column in the links dataframe
        self.put_attributes_in_links()
    
    def put_attributes_in_links(self):
        # As a single column containing dicts
        link_attrs = self.link_attrs.groupby('link_id').apply(lambda x: dict(zip(x['name'], x['value']))).reset_index(name='attributes')
        self.links = self.links.merge(link_attrs, on="link_id",how="left")
        self.links.loc[self.links["attributes"].isna(), "attributes"] = None
        
    def __len__(self):
        return len(self.links)
    
    def __str__(self):
        return 'Network: {nodes} nodes, {links} links, {crs}'.format(
            nodes=len(self.nodes),
            links=len(self.links),
            crs=Network._crsTag in self.network_attrs and self.network_attrs[Network._crsTag] or 'No CRS')
    
    def as_geo(self, projection=None):
        import geopandas as gpd
        import shapely.geometry as shp
    
        """Return a GeoPandas GeoDataFrame containing link geometries suitable for plotting."""
    
        # Project the coords, if CRS is specified somehow
        if projection:
            crs = {'init': projection}
        elif Network._crsTag in self.network_attrs:
            crs = self.network_attrs[Network._crsTag]
        else:
            crs = None
    
        # attach xy to links
        full_net = (self.links
        .merge(self.nodes,
                left_on='from_node',
                right_on='node_id')
        .merge(self.nodes,
                left_on='to_node',
                right_on='node_id',
                suffixes=('_from_node', '_to_node'))
        )
    
        # create the geometry column from coordinates
        geometry = [shp.LineString([(ox,oy), (dx,dy)]) for ox, oy, dx, dy in zip(full_net.x_from_node, full_net.y_from_node, full_net.x_to_node, full_net.y_to_node)]
    
        # build the geopandas geodataframe
        geo_net = (gpd.GeoDataFrame(full_net,
            geometry=geometry,
            crs = crs)
            .drop(columns=['x_from_node','y_from_node','node_id_from_node','node_id_to_node','x_to_node','y_to_node'])
            )
    
        return geo_net
    
    def as_nx(self):
        import networkx as nx 
        G = nx.Graph()  
        links = self.links
        nodes = self.nodes  

        # Add nodes with coordinates        
        G.add_nodes_from(
            zip(nodes['node_id'], 
                ({'x': x, 'y': y} for x, y in zip(nodes['x'], nodes['y'])))
        )

        # Add edges with attributes
        G.add_edges_from(
            zip(links['from_node'], links['to_node'],
                ({'link_id': lid, 'free_travel_time': tt} for lid, tt in zip(links['link_id'], links['free_travel_time'])))
        )

        return G

    def save(self,file_path,write_attrbs=True):        
        # set the types
        self.nodes = self.nodes.astype({'node_id':str,'x':float,'y':float})
        if "z" in self.nodes:
            self.nodes = self.nodes.astype({'z':float})

        self.links = self.links.astype({'link_id':str,
                                        'from_node':str, 
                                        'to_node':str, 
                                        'length':float,     
                                        'freespeed':float, 
                                        'capacity':int, 
                                        'permlanes':int,      
                                        'oneway':int,
                                        'modes':str,
                                        })
        
        if ("attributes" not in self.links) and (len(self.link_attrs)) and write_attrbs:
            self.put_attributes_in_links()

        # save the network in xml file
        file_open = gzip.open if file_path.endswith('.gz') else open
        with file_open(file_path, 'wb+') as f_write:
            writer = NetworkWriter(f_write)
            writer.start_network()
            writer.add_nodes(self.nodes['node_id'],
                             self.nodes.x, 
                             self.nodes.y, 
                             self.nodes.z if "z" in self.nodes else None)
            writer.add_links(self.links['link_id'],
                             self.links['from_node'],
                             self.links['to_node'],
                             self.links['length'],
                             self.links['freespeed'],
                             self.links['capacity'],
                             self.links['permlanes'],
                             self.links['oneway'],
                             self.links['modes'],
                             self.links['attributes'] if "attributes" in self.links else None,
                             write_attrbs = write_attrbs)
            writer.end_network()

    


def parse_nodes(content, skip_attributes):
    # Parse nodes from XML bytes
    tree = ET.iterparse(io.BytesIO(content), events=['start', 'end'])
    node_ids = []
    node_xs = []
    node_ys = []
    node_zs = []
    node_attrs = []
    network_attrs = {}
    current_id = None
    for xml_event, elem in tree:
        if elem.tag == 'node' and xml_event == 'start':
            atts = elem.attrib
            current_id = str(atts['id'])
            node_ids.append(current_id)
            node_xs.append(float(atts['x']))
            node_ys.append(float(atts['y']))
            node_zs.append(float(atts.get('z', 0.0)) if 'z' in atts else None)
        elif elem.tag == 'attribute' and xml_event == 'end':
            if elem.attrib['name'] == Network._crsTag:
                network_attrs[Network._crsTag] = elem.text
            elif not skip_attributes:
                atts = {}
                atts['node_id'] = current_id
                atts['name'] = elem.attrib['name']
                atts['value'] = elem.text
                if 'class' in elem.attrib:
                    if elem.attrib['class'] == 'java.lang.Long':
                        atts['value'] = int(elem.text)
                    elif elem.attrib['class'] == 'java.lang.Double':
                        atts['value'] = float(elem.text)
                    elif elem.attrib['class'] == 'java.lang.Integer':
                        atts['value'] = int(elem.text)
                node_attrs.append(atts)
        if elem.tag == 'node' and xml_event == 'end':
            elem.clear()
    nodes = pd.DataFrame({'node_id': node_ids, 'x': node_xs, 'y': node_ys, 'z': node_zs})
    if nodes.z.isna().all():
        nodes = nodes.drop(columns=['z'])

    node_attrs = pd.DataFrame.from_records(node_attrs)
    return nodes, node_attrs, network_attrs

def parse_links(content, skip_attributes):
    # Parse links from XML bytes
    tree = ET.iterparse(io.BytesIO(content), events=['start', 'end'])
    link_ids = []
    link_froms = []
    link_tos = []
    link_lengths = []
    link_freespeeds = []
    link_capacities = []
    link_permlanes = []
    link_modes = []
    link_oneways = []
    link_attrs = []
    current_id = None
    for xml_event, elem in tree:
        if elem.tag == 'link' and xml_event == 'start':
            atts = elem.attrib
            current_id = str(atts['id'])
            link_ids.append(current_id)
            link_froms.append(str(atts['from']))
            link_tos.append(str(atts['to']))
            link_lengths.append(float(atts['length']))
            link_freespeeds.append(float(atts['freespeed']))
            link_capacities.append(float(atts['capacity']))
            link_permlanes.append(float(atts['permlanes']))
            link_modes.append(str(atts.get('modes', '')))
            link_oneways.append(int(atts.get('oneway', 1)))
        elif elem.tag == 'attribute' and xml_event == 'end' and not skip_attributes:
            atts = {}
            atts['link_id'] = current_id
            atts['name'] = elem.attrib['name']
            atts['value'] = elem.text
            if 'class' in elem.attrib:
                if elem.attrib['class'] == 'java.lang.Long':
                    atts['value'] = int(elem.text)
                elif elem.attrib['class'] == 'java.lang.Double':
                    atts['value'] = float(elem.text)
                elif elem.attrib['class'] == 'java.lang.Integer':
                    atts['value'] = int(elem.text)
            link_attrs.append(atts)
        if elem.tag == 'link' and xml_event == 'end':
            elem.clear()
    links = pd.DataFrame({
        'link_id': link_ids, 'from_node': link_froms, 'to_node': link_tos,
        'length': link_lengths, 'freespeed': link_freespeeds, 'capacity': link_capacities,
        'permlanes': link_permlanes, 'modes': link_modes, 'oneway': link_oneways
    })

    link_attrs = pd.DataFrame.from_records(link_attrs)
    return links, link_attrs

def read_network(filename, skip_attributes=False):
    """Read a MATSim network.xml.gz file. Returns a Network object with dataframes
    for nodes, links, node_attributes, and link_attributes. If the network has a CRS
    projection set, it will be available in network_attrs."""
    with xopen.xopen(filename, 'rb') as f:
        content = f.read()
    
    # Find the start of <links>
    links_start = content.find(b'<links')
    if links_start == -1:
        raise ValueError("Could not find <links> tag in file.")
    
    nodes_content = content[:links_start] + b'</network>'  # Close the nodes part
    links_content = b'<network>' + content[links_start:]  # Start the links part
    
    with multiprocessing.Pool(2) as pool:
        nodes_result = pool.apply_async(parse_nodes, (nodes_content, skip_attributes))
        links_result = pool.apply_async(parse_links, (links_content, skip_attributes))
        nodes, node_attrs, network_attrs = nodes_result.get()
        links, link_attrs = links_result.get()
    
    # Ensure types
    nodes["node_id"] = nodes["node_id"].astype(str)
    links["link_id"] = links["link_id"].astype(str)
    if not link_attrs.empty:
        link_attrs["link_id"] = link_attrs["link_id"].astype(str)
    
    return Network(nodes, links, node_attrs, link_attrs, network_attrs)