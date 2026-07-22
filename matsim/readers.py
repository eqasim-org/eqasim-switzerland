# -*- coding: utf-8 -*-

import xopen
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import pandas as pd
from matsim.writers import NetworkWriter
import gzip
import io
import networkx as nx
import numpy as np
import logging
logger = logging.getLogger(__name__)

###################################################################################
########################     Network Reader     ###################################
###################################################################################

#TODO: optimize reading with multiprocessing
#TODO: optimize saving with multiprocessing, buffered batch writing

class Network:

    _crsTag = 'coordinateReferenceSystem'

    def __init__(self, nodes, links, node_attrs, link_attrs, net_attrs=None, only_car_links=False):
        self.nodes = nodes
        self.links = links
        self.link_attrs = link_attrs
        self.node_attrs = node_attrs

        self.network_attrs = {}
        if net_attrs: self.network_attrs = net_attrs
        
        # This will add attributes as a column in the links dataframe
        if "attributes" not in self.links.columns and not self.link_attrs.empty:
            self.put_attributes_in_links()

        # keep only car links
        if only_car_links:
            self.links = self.links[self.links['modes'].str.split(",").map(lambda x: "car" in x)].reset_index(drop=True)
            
            used_node_ids = set(self.links['from_node']) | set(self.links['to_node'])
            self.nodes = self.nodes[self.nodes['node_id'].isin(used_node_ids)].reset_index(drop=True)

    
    def put_attributes_in_links(self):
        if self.link_attrs.empty:
            self.links["attributes"] = None
            return

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
    
    def as_nx(self, only_car_links=False):         
        G = nx.DiGraph()  
        links = self.links
        nodes = self.nodes  
        
        if only_car_links:
            links = links[links['modes'].str.split(",").map(lambda x: "car" in x)].reset_index(drop=True)

        lengths = links['length'].values
        free_travel_times = lengths / links['freespeed'].values
        speed_factors = np.ones_like(lengths)
        if "attributes" in links:
            speed_factors = links["attributes"].apply(lambda attr: attr.get("speedFactor", 1.0) if isinstance(attr, dict) else 1.0).values       
        # Add nodes with coordinates        
        G.add_nodes_from(
            zip(nodes['node_id'], 
                ({'x': x, 'y': y} for x, y in zip(nodes['x'], nodes['y'])))
        )

        # Add edges with attributes
        G.add_edges_from(
            zip(links['from_node'], links['to_node'],
                ({'link_id': lid, 'travel_time': tt, "length": length, "speed_factor": sf} for lid, tt, length, sf in zip(links['link_id'], free_travel_times, lengths, speed_factors)))
        )

        return G

    def as_igraph(self, only_car_links=False):
        import igraph as ig
        links = self.links
        nodes = self.nodes

        if only_car_links:
            links = links[links['modes'].str.split(",").map(lambda x: "car" in x)].reset_index(drop=True)

        # Calculate travel times
        lengths = links['length'].values
        free_travel_times = lengths / links['freespeed'].values
        speed_factors = np.ones_like(lengths)
        if "attributes" in links:
            speed_factors = links["attributes"].apply(lambda attr: attr.get("speedFactor", 1.0) if isinstance(attr, dict) else 1.0).values                
        # Create igraph
        g = ig.Graph(directed=True)

        # Add vertices with node_id and coordinates
        g.add_vertices(len(nodes))
        g.vs['node_id'] = nodes['node_id'].values
        g.vs['x'] = nodes['x'].values
        g.vs['y'] = nodes['y'].values

        # Create node_id → index mapping for edge construction
        node_id_to_index = {nid: idx for idx, nid in enumerate(nodes['node_id'])}

        # Build edge list using vertex indices
        edge_list = [
            (node_id_to_index[src], node_id_to_index[tgt])
            for src, tgt in zip(links['from_node'], links['to_node'])
        ]

        # Add edges with attributes
        g.add_edges(edge_list)
        g.es['link_id'] = links['link_id'].values
        g.es['travel_time'] = free_travel_times
        g.es['length'] = lengths
        g.es['speed_factor'] = speed_factors

        return g

    def as_pandana(self, only_car_links=False, use_speed_factor=False, directed=True):
        import pandana as pdna
        links = self.links.reset_index(drop=True).copy()
        nodes = self.nodes.reset_index(drop=True).copy()

        if only_car_links:
            links = links[links['modes'].str.split(",").map(lambda x: "car" in x)].reset_index(drop=True)
            nodes = nodes[nodes['node_id'].isin(pd.unique(links['from_node'].tolist() + links['to_node'].tolist()))].reset_index(drop=True) 

        # Calculate travel times
        lengths = links['length'].values
        free_travel_times = lengths / links['freespeed'].values

        if use_speed_factor:
            speed_factors = np.ones_like(lengths)
            if "attributes" in links:
                speed_factors = links["attributes"].apply(lambda attr: attr.get("speedFactor", 1.0) if isinstance(attr, dict) else 1.0).values
            free_travel_times = free_travel_times / speed_factors
        
        # get indices        
        node_id_map = dict(zip(nodes["node_id"], nodes.index))
        from_nodes = links["from_node"].map(lambda x: node_id_map.get(x)).astype('int32')
        to_nodes = links["to_node"].map(lambda x: node_id_map.get(x)).astype('int32')

        # Create pandana network
        weights = pd.DataFrame(dict(length=lengths, travel_time=free_travel_times))
        net = pdna.Network(nodes['x'], nodes['y'], from_nodes, to_nodes,
                           twoway=not directed,
                           edge_weights = weights)
        
        return net

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











def _iterparse(source):
    # lxml supports huge_tree for large XML files; stdlib ElementTree does not.
    try:
        return ET.iterparse(source, events=['start', 'end'], huge_tree=True)
    except TypeError:
        return ET.iterparse(source, events=['start', 'end'])


def _local_tag(tag):
    if '}' in tag:
        return tag.rsplit('}', 1)[1]
    return tag


def _coerce_attribute_value(text, class_name):
    if text is None:
        return None
    if class_name in ('java.lang.Long', 'java.lang.Integer'):
        return int(text)
    if class_name == 'java.lang.Double':
        return float(text)
    return text


def _parse_network_stream(stream, skip_attributes, parse_nodes=True, parse_links=True):
    tree = _iterparse(stream)

    node_ids = []
    node_xs = []
    node_ys = []
    node_zs = []

    link_ids = []
    link_froms = []
    link_tos = []
    link_lengths = []
    link_freespeeds = []
    link_capacities = []
    link_permlanes = []
    link_modes = []
    link_oneways = []

    node_attrs = []
    link_attrs = []
    network_attrs = {}

    current_node_id = None
    current_link_id = None

    for xml_event, elem in tree:
        tag = _local_tag(elem.tag)

        if xml_event == 'start' and tag == 'node' and parse_nodes:
            atts = elem.attrib
            current_node_id = str(atts['id'])
            node_ids.append(current_node_id)
            node_xs.append(float(atts['x']))
            node_ys.append(float(atts['y']))
            node_zs.append(float(atts.get('z', 0.0)) if 'z' in atts else None)

        elif xml_event == 'start' and tag == 'link' and parse_links:
            atts = elem.attrib
            current_link_id = str(atts['id'])
            link_ids.append(current_link_id)
            link_froms.append(str(atts['from']))
            link_tos.append(str(atts['to']))
            link_lengths.append(float(atts['length']))
            link_freespeeds.append(float(atts['freespeed']))
            link_capacities.append(float(atts['capacity']))
            link_permlanes.append(float(atts['permlanes']))
            link_modes.append(str(atts.get('modes', '')))
            link_oneways.append(int(atts.get('oneway', 1)))

        elif xml_event == 'end' and tag == 'attribute':
            name = elem.attrib.get('name')
            class_name = elem.attrib.get('class')

            if name == Network._crsTag:
                network_attrs[Network._crsTag] = elem.text
            elif not skip_attributes:
                value = _coerce_attribute_value(elem.text, class_name)

                if current_link_id is not None and parse_links:
                    link_attrs.append({
                        'link_id': current_link_id,
                        'name': name,
                        'value': value,
                    })
                elif current_node_id is not None and parse_nodes:
                    node_attrs.append({
                        'node_id': current_node_id,
                        'name': name,
                        'value': value,
                    })

        elif xml_event == 'end' and tag == 'node':
            current_node_id = None

        elif xml_event == 'end' and tag == 'link':
            current_link_id = None

        if xml_event == 'end':
            elem.clear()

    nodes = pd.DataFrame({'node_id': node_ids, 'x': node_xs, 'y': node_ys, 'z': node_zs})
    if not nodes.empty and nodes.z.isna().all():
        nodes = nodes.drop(columns=['z'])

    links = pd.DataFrame({
        'link_id': link_ids,
        'from_node': link_froms,
        'to_node': link_tos,
        'length': link_lengths,
        'freespeed': link_freespeeds,
        'capacity': link_capacities,
        'permlanes': link_permlanes,
        'modes': link_modes,
        'oneway': link_oneways,
    })

    node_attrs_df = pd.DataFrame.from_records(node_attrs, columns=['node_id', 'name', 'value'])
    link_attrs_df = pd.DataFrame.from_records(link_attrs, columns=['link_id', 'name', 'value'])

    return nodes, links, node_attrs_df, link_attrs_df, network_attrs


def parse_nodes(content, skip_attributes):
    # Parse nodes from XML bytes
    nodes, _, node_attrs, _, network_attrs = _parse_network_stream(
        io.BytesIO(content), skip_attributes, parse_nodes=True, parse_links=False
    )
    return nodes, node_attrs, network_attrs


def parse_links(content, skip_attributes):
    # Parse links from XML bytes
    _, links, _, link_attrs, _ = _parse_network_stream(
        io.BytesIO(content), skip_attributes, parse_nodes=False, parse_links=True
    )
    return links, link_attrs


def read_network(filename, skip_attributes=False, only_car_links=False):
    """Read a MATSim network.xml.gz file. Returns a Network object with dataframes
    for nodes, links, node_attributes, and link_attributes. If the network has a CRS
    projection set, it will be available in network_attrs."""
    with xopen.xopen(filename, 'rb') as f:
        nodes, links, node_attrs, link_attrs, network_attrs = _parse_network_stream(
            f, skip_attributes, parse_nodes=True, parse_links=True
        )

    # Ensure types
    nodes["node_id"] = nodes["node_id"].astype(str)
    links["link_id"] = links["link_id"].astype(str)
    if not link_attrs.empty:
        link_attrs["link_id"] = link_attrs["link_id"].astype(str)

    return Network(nodes, links, node_attrs, link_attrs, network_attrs, only_car_links=only_car_links)





###################################################################################
########################      Plans Reader      ###################################
###################################################################################

class Plans:
    def __init__(self, persons, plans, activities, legs, routes):
        self.persons = persons
        self.plans = plans
        self.activities = activities
        self.legs = legs
        self.routes = routes

def plan_reader(filename, selected_plans_only = False):
    person = None
    tree = ET.iterparse(xopen.xopen(filename), events=['start','end'])
    
    for xml_event, elem in tree:
        if elem.tag == 'person' and xml_event == 'start':
            # keep track of whether a person node has any plans
            this_person_has_plans = False

            if person: person.clear() # clear memory
            person = elem

        elif elem.tag == 'plan' and xml_event == 'end':
            this_person_has_plans = True

            # filter out unselected plans if asked to do so
            if selected_plans_only and elem.attrib['selected'] == 'no': continue

            yield (person, elem)

            # free memory. Otherwise the data is kept in memory
            elem.clear()
        
        elif elem.tag == 'person' and xml_event == 'end':
            # if this person has no plans, then yield the person with a None plan.
            if not this_person_has_plans:
                yield (person, None)

# Parses attributes of an element and adds them to the given dictionary
def _parseAttributes(elem, dict):
    for attrib in elem.attrib:
        dict[attrib] = elem.attrib[attrib]
    return dict

# Returns dataframes with the following relations between them:
# Person : None
# Plan : person_id
# Activity : plan_id
# Leg : plan_id
# Route :leg_id
# The column names of the dataframes are the same as the attribute names (<name:'value'> and <attribute> are parsed)
def plan_reader_dataframe(filename, selected_plans_only = False):
    tree = ET.iterparse(xopen.xopen(filename), events=['start','end'])
    
    persons = []
    plans = []
    activities = []
    legs = []
    routes = []
    
    current_person = {}
    current_plan = {}
    current_activity = {}
    current_leg = {}
    current_route = {}
    
    # Indicates current parent element while parsing <attribute> element
    is_parsing_person = False
    is_parsing_activity = False
    is_parsing_leg = False
    
    current_person_id = None
    current_plan_id = 0
    current_activity_id = 0
    current_leg_id = 0
    current_route_id = 0
    
    for xml_event, elem in tree:
        if elem.tag in ['person', 'leg', 'activity', 'plan', 'route'] and xml_event == 'end':
            if is_parsing_person:
                persons.append(current_person)
                current_person = {}
                is_parsing_person = False
            
            if is_parsing_activity:
                activities.append(current_activity)
                current_activity = {}
                is_parsing_activity = False
            
            if is_parsing_leg:
                legs.append(current_leg)
                current_leg = {}
                is_parsing_leg = False
            
            if elem.tag == 'plan':
                if elem.attrib['selected'] == 'no' and selected_plans_only: continue
                plans.append(current_plan)
                current_plan = {}
                
            if elem.tag == 'route':
                routes.append(current_route)
                current_route = {}
            
            elem.clear()
        
        # PERSON
        elif elem.tag == 'person':
            current_person['id'] = elem.attrib['id']
            current_person_id = elem.attrib['id']
            is_parsing_person = True
        
        # PLAN
        elif elem.tag == 'plan':
            if elem.attrib['selected'] == 'no' and selected_plans_only: continue
            current_plan_id += 1
            
            current_plan['id'] = current_plan_id
            current_plan['person_id'] = current_person_id
            current_plan = _parseAttributes(elem, current_plan)
        
        # ACTIVITY
        elif elem.tag == 'activity':
            is_parsing_activity = True
            current_activity_id += 1
            
            current_activity['id'] = current_activity_id
            current_activity['plan_id'] = current_plan_id
            current_activity = _parseAttributes(elem, current_activity)
            
        
        # LEG
        elif elem.tag == 'leg':
            is_parsing_leg = True
            current_leg_id += 1
            
            current_leg['id'] = current_leg_id
            current_leg['plan_id'] = current_plan_id
            current_leg = _parseAttributes(elem, current_leg)
        
        
        # ROUTE
        elif elem.tag == 'route':
            current_route_id += 1
            
            current_route['id'] = current_route_id
            current_route['leg_id'] = current_leg_id
            current_route['value'] = elem.text
            current_route = _parseAttributes(elem, current_route)
        
        
        # ATTRIBUTES
        elif elem.tag == 'attribute' and xml_event == 'end':
            attribs = elem.attrib
            
            if is_parsing_activity:
                current_activity[attribs['name']] = elem.text
                
            elif is_parsing_leg:
                current_leg[attribs['name']] = elem.text
            
            elif is_parsing_person: # Parsing person
                current_person[attribs['name']] = elem.text
    
    persons = pd.DataFrame.from_records(persons)
    plans = pd.DataFrame.from_records(plans)
    activities = pd.DataFrame.from_records(activities)
    legs = pd.DataFrame.from_records(legs)
    routes = pd.DataFrame.from_records(routes)
    
    return Plans(persons, plans, activities, legs, routes)
    






