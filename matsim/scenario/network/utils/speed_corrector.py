import numpy as np
from matsim.readers import Network
import shapely.vectorized
import geopandas as gpd
import shapely.geometry as shp
import logging

logger = logging.getLogger("synpp")

class SpeedCorrector:
    """
    Class to correct the capacity of the links in the network.
    
    Parameters:
        network (Network): The network object containing the links.
    """
    
    def __init__(self, context, network:Network):
        self.context = context
        self.network = network        

    def get_regions(self):
        """
        This method retrieves and processes municipality data to create regions based on municipality types.
        It merges municipality types with their geometries and dissolves them by type to create aggregated regions
        for further processing.
        """
        # Retrieve municipality types data from the context
        df1 = self.context.stage("data.spatial.municipality_types")
        # Retrieve municipalities data from the context (ignoring the second return value)
        df2, _ = self.context.stage("data.spatial.municipalities")        
        # Merge the two dataframes on 'municipality_id' to combine type and geometry info
        df = df1[['municipality_id', 'municipality_type']].merge(df2, on="municipality_id")
        # Convert to GeoDataFrame with the specified CRS
        df = gpd.GeoDataFrame(df, crs="EPSG:2056")
        # Select only the relevant columns
        df = df[['municipality_type', 'geometry']]
        # Dissolve by municipality_type to aggregate geometries, then reset index
        merged = df.dissolve(by="municipality_type").reset_index()
        # Return only the municipality_type and geometry columns
        return merged

    def get_centroids(self):
        """
        This method calculates the centroids of all links in the network.
        It merges the link data with node coordinates to compute the midpoints.
        """
        links_centers = (self.network.links[["link_id", "from_node", "to_node"]]
                        .merge(self.network.nodes, left_on='from_node', right_on='node_id', how="left")
                        .merge(self.network.nodes, left_on='to_node', right_on='node_id', suffixes=('_from_node', '_to_node'), how="left")
                        )
        centroids_x = (links_centers.x_from_node + links_centers.x_to_node) / 2
        centroids_y = (links_centers.y_from_node + links_centers.y_to_node) / 2
        geometry    = gpd.points_from_xy(centroids_x, centroids_y, crs="EPSG:2056")
        links_centers = gpd.GeoDataFrame(links_centers[["link_id"]], geometry=geometry, crs="EPSG:2056")
        return links_centers
    
    def assign_municipality_types(self, regions):    
        """
        This method assigns municipality types to the links in the network based on their geographical location.
        It calculates the centroid of each link and checks which municipality polygon it falls within.
        If a link's centroid does not fall within any municipality, it is assigned the type 'outside'.
        """    
        links_centers = self.get_centroids()
        
        # We can simply use sjoin, but this is faster for a limited number of regions
        links_centers["municipality_type"] = None
        for _, region in regions.iterrows():
            mask = shapely.vectorized.contains(region.geometry, links_centers.geometry.x.tolist(), links_centers.geometry.y.tolist())
            links_centers.loc[mask, "municipality_type"] = region.municipality_type
        
        # Assign 'outside' to links that do not fall within any municipality
        links_centers["municipality_type"] = links_centers["municipality_type"].fillna("outside")

        return links_centers[["link_id", "municipality_type"]]
    
    def assign_municipality_types_to_network(self):
        logger.info("Assigning municipality types to network links...")
        regions = self.get_regions()            
        links_types = self.assign_municipality_types(regions)
        links = self.network.links.copy()
        links = links.merge(links_types, on="link_id", how="left")
        links.apply(lambda row: row['attributes'].update({'municipalityType': row['municipality_type']}), axis=1)
        links = links.drop(columns=["municipality_type"])
        return links

    def run_municipality_type_based_correction(self):
        """        
        This method corrects the free speed of links in the network based on the municipality type they are located in.
        The correction is applied only to car links and only if their free speed is below a specified speed limit.
        The municipality types considered are 'urban', 'suburban', 'rural', and 'outside'.
        """ 
        municipality_type_is_assigned = self.network.links['attributes'].apply(lambda attrs: 'municipalityType' in attrs).any()
        if not municipality_type_is_assigned:
            logger.info("Municipality types not assigned to links. Assigning now...")
            self.network.links = self.assign_municipality_types_to_network()
            
        # Merge the municipality types with the network links
        links = self.network.links.copy()
        links["municipality_type"] = links['attributes'].apply(lambda attrs: attrs.get('municipalityType', 'outside'))
        car_links = self.network.links.modes.str.contains(r"\bcar\b")
        
        # Apply speed correction if enabled
        if self.context.config("correct_speed"):
            speed_limit_for_correction = self.context.config("speed_limit_for_correction")
            speed_factors = {
                    "urbancore": self.context.config("speed_factor_urbancore"),
                    "urban": self.context.config("speed_factor_urban"),
                    "suburban": self.context.config("speed_factor_suburban"),
                    "rural": self.context.config("speed_factor_rural"),
                    "outside": 1.0
                }
            missing_types = [t for t in links['municipality_type'].unique() if t not in speed_factors]
            assert not missing_types, f"Speed factors not defined for municipality types: {missing_types}"

            def correct_speed(row):
                if row['freespeed'] < speed_limit_for_correction:
                    factor = speed_factors.get(row['municipality_type'], 1.0)
                    return row['freespeed'] * factor
                else:
                    return row['freespeed']

            links.loc[car_links, 'freespeed'] = links[car_links].apply(correct_speed, axis=1)
                
        links = links.drop(columns=["municipality_type"])
        return links

    def assign_slopes(self):
        """
        This method calculates the elevation gain for each link in the network based on the elevations of its from_node and to_node.
        It adds a new column 'elevation_gain' to the links DataFrame.
        """
        nodes_elevation = self.network.nodes[['node_id', 'z']]
        links = (self.network.links[["link_id", "length", "from_node", "to_node"]]
                    .merge(nodes_elevation, left_on='from_node', right_on='node_id', how="left")
                    .merge(nodes_elevation, left_on='to_node', right_on='node_id', suffixes=('_from_node', '_to_node'), how="left")
                        )
        no_elevation = ((links['z_from_node'].isna()) | (links['z_to_node'].isna()) | 
                        (links['z_from_node']==0) | (links['z_to_node']==0) | (links['length']==0))
        
        links['slope'] = 0.0
        links.loc[~no_elevation, "slope"] = ((links.loc[~no_elevation,'z_to_node'] - links.loc[~no_elevation,'z_from_node']) 
                                             / links.loc[~no_elevation,'length']) * 100
        return links[['link_id', 'slope']]
    
    def run_uphill_based_correction(self):
        """
        This method corrects the free speed of links in the network based on their gradient.
        """
        # Merge the slopes with the network links
        links = self.network.links.copy()
        slopes = self.assign_slopes()
        links = links.merge(slopes, on="link_id", how="left")

        # Apply speed correction to uphill car links
        car_links = self.network.links.modes.str.contains(r"\bcar\b")        
        max_gradient_threshold = self.context.config("max_gradient_threshold")
        speed_factor_uphill = self.context.config("speed_factor_uphill")

        def correct_speed(row):
            gradient = abs(row['slope'] / 100) # convert percentage to decimal
            if gradient > max_gradient_threshold:
                if row['slope'] > 0:
                    return row['freespeed'] * speed_factor_uphill
                else:
                    return row['freespeed'] * ((1+speed_factor_uphill)/2) # less reduction for downhill
            else:
                return row['freespeed']

        links.loc[car_links, 'freespeed'] = links[car_links].apply(correct_speed, axis=1)
        # drop the slope column after correction
        links = links.drop(columns=["slope"])
        return links
    
    def get_swiss_border(self):
        """
        This method retrieves the Swiss border geometry from the context.
        """
        border = self.context.stage("data.spatial.swiss_border")
        border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
        return border.geometry.iloc[0]

    def run_outside_border_correction(self):
        """
        This method is a placeholder for outside border correction.
        Currently, it does not apply any correction and simply returns the original links.
        """
        # Identify links outside the Swiss border
        border = self.get_swiss_border()
        centroids = self.get_centroids()
        outside_mask = ~shapely.vectorized.contains(border, centroids.geometry.x.tolist(), centroids.geometry.y.tolist())
        links_outside_border = set(centroids.loc[outside_mask, "link_id"].unique())
        
        # Apply speed correction to car links outside the border
        speed_factor = self.context.config("speed_factor_outside_border")
        links = self.network.links.copy()
        car_links = self.network.links.modes.str.contains(r"\bcar\b")        
        mask = car_links & links['link_id'].isin(links_outside_border)
        
        links.loc[mask, 'freespeed'] = links.loc[mask, 'freespeed'] * speed_factor
        return links
    
    def run(self, correction_type="municipality_type"):
        """
        This method runs the speed correction process based on the specified correction type.
        Currently, it supports 'municipality_type' and 'uphill' correction types.
        
        Parameters:
            correction_type (str): The type of correction to apply. Default is 'municipality_type'.
        """
        assert correction_type in ["municipality_type","uphill","outside_border","motorway"], f"Unsupported correction type: {correction_type}"

        if correction_type == "municipality_type":
            return self.run_municipality_type_based_correction()
        elif correction_type == "uphill":
            return self.run_uphill_based_correction()
        elif correction_type == "outside_border":           
            return self.run_outside_border_correction()
        elif correction_type=="motorway":
            return self.run_motorway_correction()
    
    def run_motorway_correction(self):
        """
        This method corrects the free speed of motorway links in the network.
        The correction is applied only to car links classified as motorways.
        """ 
        links = self.network.links.copy()
        car_links = links.modes.str.contains(r"\bcar\b")
        motorway_links = links['freespeed']>=(80/3.6)
        
        # Apply speed correction        
        speed_factor_motorway = self.context.config("speed_factor_motorway")
        mask = car_links & motorway_links
        links.loc[mask, 'freespeed'] = links.loc[mask, 'freespeed'] * speed_factor_motorway
                
        return links
        


