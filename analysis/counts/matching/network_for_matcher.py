import os
import geopandas as gpd
import shapely


def configure(context):
    context.stage("analysis.counts.matching.network")
    context.config("polygone_for_cutting_network_for_matcher")

def execute(context):
    network = context.stage("analysis.counts.matching.network")
    links   = network.links

    # get osm id
    links["osm_id"] = links["attributes"].apply(lambda x: x.get('osm:way:id'))

    # merge links with nodes coords
    links = links.merge(network.net_geo, on="link_id", how="left")
    links = (links.merge(network.net.nodes, left_on='from_node', right_on='node_id')
                  .merge(network.net.nodes, left_on='to_node', right_on='node_id', suffixes=('_from_node', '_to_node')))

    # get only relevant columns
    cols = ['link_id', 'osm_id', 'x_from_node', 'y_from_node', 'x_to_node', 'y_to_node', 'geometry']
    links = gpd.GeoDataFrame(links[cols], crs="EPSG:2056")

    # geometry used to cut
    polygone_file = context.config("polygone_for_cutting_network_for_matcher")
    gdf = gpd.read_file(polygone_file)
    if not isinstance(gdf, gpd.GeoDataFrame):
        assert "geometry" in gdf.columns, "We expect to have 'geometry' column in the polygone file"
        if type(gdf['geometry'].iloc[0]) == str:
            gdf['geometry'] = gdf['geometry'].apply(lambda x: shapely.wkt.loads(x))
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    # if multiple polygones or multiple points, we create a union of all geometries, then a polygon that would cover
    # all these geometries and buffer it with 1 km to get a bigger area to cut the network
    geo_union = gdf.to_crs("EPSG:2056").geometry.union_all()
    cutting_area = geo_union.convex_hull.buffer(1_000)
    links = links[links.geometry.intersects(cutting_area)].copy()

    links = links.to_crs(epsg=4326)
    output_path = os.path.join(context.path, "zurich_network.gpkg")
    links.to_file(output_path)