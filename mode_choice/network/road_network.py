"""
I made this as a stage because it takes time to read the network from xml, but it is faster to read the pickle file,
in case it was already created.
"""
import xopen
import xml.etree.ElementTree as ET
import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)

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

def read_network(filename, skip_attributes=False):
    """Read a MATSim network.xml.gz file. Returns a Network object with dataframes
    for nodes, links, node_attributes, and link_attributes. If the network has a CRS
    projection set, it will be available in network_attrs."""
    tree = ET.iterparse(xopen.xopen(filename, 'r'), events=['start', 'end'])
    nodes = []
    links = []
    node_attrs = []
    link_attrs = []

    network_attrs = {}

    attributes = node_attrs
    attr_label = 'node_id'
    current_id = None

    for xml_event, elem in tree:
        # the nodes element CLOSES at the end of the nodes, followed by links:
        if elem.tag == 'links' and xml_event == 'start':
            attributes = link_attrs
            attr_label = 'link_id'

        elif elem.tag == 'node' and xml_event == 'start':
            atts = elem.attrib
            current_id = atts['id']

            atts['node_id'] = atts.pop('id')
            atts['x'] = float(atts['x'])
            atts['y'] = float(atts['y'])
            if 'z' in atts: atts['z'] = float(atts['z'])

            nodes.append(atts)

        elif elem.tag == 'link' and xml_event == 'start':
            atts = elem.attrib
            current_id = atts['id']

            atts['link_id'] = atts.pop('id')
            atts['from_node'] = atts.pop('from')
            atts['to_node'] = atts.pop('to')

            atts['length'] = float(atts['length'])
            atts['freespeed'] = float(atts['freespeed'])
            atts['capacity'] = float(atts['capacity'])
            atts['permlanes'] = float(atts['permlanes'])

            if 'volume' in atts: atts['volume'] = float(atts['volume'])

            links.append(atts)


        elif elem.tag == 'attribute' and xml_event == 'end':
            if elem.attrib['name'] == Network._crsTag:
                network_attrs[Network._crsTag] = elem.text

            elif not skip_attributes:
                atts = {}
                atts[attr_label] = current_id
                atts['name'] = elem.attrib['name']
                atts['value'] = elem.text

                # TODO: pandas will make the value column "object" since we're mixing types
                if 'class' in elem.attrib:
                    if elem.attrib['class'] == 'java.lang.Long':
                        atts['value'] = int(elem.text)
                    if elem.attrib['class'] == 'java.lang.Double':
                        atts['value'] = float(elem.text)
                    if elem.attrib['class'] == 'java.lang.Integer':
                        atts['value'] = int(elem.text)

                attributes.append(atts)

        # clear the element when we're done, to keep memory usage low
        if elem.tag in ['node', 'link'] and xml_event == 'end':
            elem.clear()

    nodes = pd.DataFrame.from_records(nodes)
    links = pd.DataFrame.from_records(links)
    node_attrs = pd.DataFrame.from_records(node_attrs)
    link_attrs = pd.DataFrame.from_records(link_attrs)
    
    # make sure all ids are str
    nodes["node_id"] = nodes["node_id"].astype(str)
    links["link_id"] = links["link_id"].astype(str)
    link_attrs["link_id"] = link_attrs["link_id"].astype(str)
    
    return Network(nodes, links, node_attrs, link_attrs, network_attrs)

def configure(context):
    context.config("data_path")
    context.config("dmc_network_file", 
                   default=os.path.join(context.config("data_path"), "dmc", "switzerland_network.xml.gz"))

def execute(context):
    network_file = context.config("dmc_network_file")
    
    logger.info(f"\t Reading network from {network_file}")
    net = read_network(network_file)
    
    # make sure id is str
    net.links['link_id'] = net.links['link_id'].astype(str)
    net.links['from_node'] = net.links['from_node'].astype(str)
    net.links['to_node'] = net.links['to_node'].astype(str)
    net.nodes['node_id'] = net.nodes['node_id'].astype(str)

    # filter out non-car links
    car_links = net.links["modes"].str.split(',').map(lambda x: "car" in x)
    net.links = net.links[car_links].reset_index(drop=True)
    net.nodes = net.nodes[net.nodes['node_id'].isin(
        pd.unique(net.links['from_node'].tolist() + net.links['to_node'].tolist())
    )].reset_index(drop=True)        

    # del unecessary data
    del net.link_attrs
    del net.node_attrs
    net.links = net.links[['link_id', 'from_node', 'to_node', 'length', 'freespeed', 'modes']]

    logger.info(f"\t Loaded {len(net.links)} links and {len(net.nodes)} nodes")
    return net