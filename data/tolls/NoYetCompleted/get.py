import os
import re
import unicodedata
import logging
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import contextily as ctx

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.tolls.aprr")
    context.stage("data.tolls.area")
    context.stage("data.tolls.atmb")
    context.stage("data.tolls.aprr_area_open_system")

    context.stage("matsim.scenario.network.convert_osm")
    context.config("data_path")

def execute(context):
    # load the network
    network, detailed_network = load_network(context)
    network.filter_car_links(inplace=True)

    # pandana graph, need you later to route
    graph = network.as_pandana(only_car_links=False, use_speed_factor=False, directed=True)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(network.nodes.node_id)}
    detailed_to_coarse_link_id = get_detailed_to_coarse_link_id(network)

    # read the tolls geometries
    aprr, geo_aprr = context.stage("data.tolls.aprr")
    area, geo_area = context.stage("data.tolls.area")
    atmb, geo_atmb = context.stage("data.tolls.atmb")
    prices_aprr_area_open_system = context.stage("data.tolls.aprr_area_open_system")

    # prepare geometries and plot the map
    locations = pd.concat([geo_atmb, geo_aprr, geo_area], ignore_index=True)
    locations = gpd.GeoDataFrame(locations, geometry="geometry", crs="EPSG:4326")
    plot_map(locations, title="Toll locations", save_path="%s/toll_locations.png" % context.path())

    # get local network and synchronize projections
    locations_2056 = locations.to_crs("EPSG:2056")
    xmin, ymin, xmax, ymax = locations_2056.buffer(5000).total_bounds
    local_net = detailed_network.cx[xmin:xmax, ymin:ymax]

    local_net = local_net.to_crs(epsg=4326)
    locations = locations.to_crs(epsg=4326)

    # Attach nearest link attributes to every location
    matched_links = gpd.sjoin_nearest(locations.rename_geometry("location_geom"),
                                      local_net[["link_id", "geometry"]],
                                      how="left",
                                      distance_col="distance"
                                     ).drop(columns=["index_right"]
                                     ).merge(local_net, on="link_id", how="left").set_geometry("geometry")

    # there are tolls in open system (atmb), prices already in the dataframe, and other need to be linked to the prices dataframe
    open_system, closed_system = match(matched_links, network, graph, node_to_idx, detailed_to_coarse_link_id, aprr, area, prices_aprr_area_open_system)
    
    # resolve problems where links are overlapping
    open_system, closed_system = resolve_overlapping_links_problems(open_system, closed_system, network, detailed_network, node_to_idx, detailed_to_coarse_link_id)

    # plot in htm file
    plot_in_html(
        locations,
        local_net,
        matched_links,
        save_path="%s/toll_locations_map.html" % context.path(),
        open_system=open_system,
        closed_system=closed_system,
    )
    
    return (open_system, closed_system)








def load_network(context):
    network = pd.read_pickle("%s/network.pkl" % context.path("matsim.scenario.network.convert_osm"))
    detailed_network = pd.read_csv("%s/detailed_network.csv" % context.path("matsim.scenario.network.convert_osm"))
    detailed_network = detailed_network.rename(columns={"LinkId":"link_id", "Geometry":"geometry"})
    def try_load(x):
        try:
            return wkt.loads(x)
        except Exception as e:
            return str(x)
    detailed_network["geometry"] = detailed_network["geometry"].apply(try_load)
    detailed_network = detailed_network[~detailed_network.geometry.apply(lambda x: isinstance(x,str))]
    detailed_network = gpd.GeoDataFrame(detailed_network,geometry='geometry', crs="epsg:2056")
    return network, detailed_network


def normalize_station_name(name):
    """Normalize French toll station names to maximize exact matches."""
    if pd.isna(name):
        return None

    name = str(name).upper().strip()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))

    # Standardize separators
    name = name.replace("-", " ").replace("'", " ").replace("é", "E").replace("è", "E").replace("ê", "E").replace("à", "A").replace("ç", "C")

    # Expand abbreviations
    name = re.sub(r"\bST\b", "SAINT", name)
    name = re.sub(r"\bSTE\b", "SAINTE", name)

    # Remove punctuation and collapse whitespace
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def plot_map(gdf, title="Map", figsize=(10, 10), save_path=None):

    df = gdf.copy().to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=figsize)
    df.plot(ax=ax, markersize=10, color="navy", edgecolor="white")
    ctx.add_basemap(ax=ax, crs="EPSG:4326", zoom=10)
    plt.axis("off")
    plt.title(title, fontsize=15)
    plt.savefig(save_path)


def plot_in_html(locations, net, matched_net, save_path, open_system=None, closed_system=None):
    import plotly.graph_objects as go

    def build_line_trace(gdf, with_hover=False, id_col="link_id", name_col="name"):
        """
        Flatten all LineString geometries in a GeoDataFrame into a single
        lon/lat trace (with None separators between lines).

        If with_hover=True, also builds a per-vertex customdata array so
        that hovering over a segment shows that line's id_col/name_col.
        """
        lons, lats = [], []
        customdata = [] if with_hover else None

        for row in gdf.itertuples():
            line = row.geometry
            x, y = line.xy
            x, y = list(x), list(y)
            n = len(x)

            lons.extend(x)
            lats.extend(y)

            if with_hover:
                link_id = getattr(row, id_col, None)
                name = getattr(row, name_col, None)
                # one [link_id, name] pair per vertex of this line
                customdata.extend([[link_id, name]] * n)

            # separator between distinct lines
            lons.append(None)
            lats.append(None)
            if with_hover:
                customdata.append([None, None])

        return lons, lats, customdata

    # Road network (no hover needed)
    lons, lats, _ = build_line_trace(net, with_hover=False)

    # Matched road network (with link_id / name on hover)
    lons_matched, lats_matched, customdata_matched = build_line_trace(
        matched_net, with_hover=True, id_col="link_id", name_col="name"
    )

    def add_link_category_trace(fig, matched_net, link_ids, color, trace_name, width=5):
        if not link_ids:
            return

        mask = matched_net["link_id"].isin(link_ids)
        if not mask.any():
            return

        cat_lons, cat_lats, cat_customdata = build_line_trace(
            matched_net.loc[mask], with_hover=True, id_col="link_id", name_col="name"
        )
        fig.add_trace(
            go.Scattermap(
                lon=cat_lons,
                lat=cat_lats,
                mode="lines",
                line=dict(color=color, width=width),
                customdata=cat_customdata,
                hovertemplate=(
                    "link_id: %{customdata[0]}<br>"
                    "name: %{customdata[1]}<extra></extra>"
                ),
                name=trace_name,
            )
        )

    # Prefer the non-deprecated GeoPandas API when available
    try:
        center = locations.union_all().centroid
    except AttributeError:
        center = locations.unary_union.centroid

    fig = go.Figure()

    # Road network
    fig.add_trace(
        go.Scattermap(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(color="navy", width=1),
            hoverinfo="skip",
            name="Road network",
        )
    )

    if closed_system is not None and not closed_system.empty:
        origin_ids = set(closed_system["origin_link_id"].dropna())
        destination_ids = set(closed_system["destination_link_id"].dropna())

        open_ids = set()
        if open_system is not None and not open_system.empty:
            open_ids = set(open_system["link_id"].dropna())

        both_ids = origin_ids & destination_ids
        origin_only_ids = origin_ids - destination_ids
        destination_only_ids = destination_ids - origin_ids

        add_link_category_trace(
            fig,
            matched_net,
            origin_only_ids,
            color="darkorange",
            trace_name="Origin links",
            width=5,
        )
        add_link_category_trace(
            fig,
            matched_net,
            destination_only_ids,
            color="royalblue",
            trace_name="Destination links",
            width=5,
        )
        add_link_category_trace(
            fig,
            matched_net,
            both_ids,
            color="mediumorchid",
            trace_name="Origin & destination links",
            width=6,
        )
        add_link_category_trace(
            fig,
            matched_net,
            open_ids,
            color="crimson",
            trace_name="Open-system links",
            width=6,
        )
    else:
        fig.add_trace(
            go.Scattermap(
                lon=lons_matched,
                lat=lats_matched,
                mode="lines",
                line=dict(color="brown", width=4),
                customdata=customdata_matched,
                hovertemplate=(
                    "link_id: %{customdata[0]}<br>"
                    "name: %{customdata[1]}<extra></extra>"
                ),
                name="Matched Road network",
            )
        )

    # Locations
    fig.add_trace(
        go.Scattermap(
            lon=locations.geometry.x.to_numpy(),
            lat=locations.geometry.y.to_numpy(),
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Locations",
        )
    )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center.y, lon=center.x),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
    )

    fig.write_html(save_path)


def get_correct_links(graph, network, node_to_idx, detailed_to_coarse_link_id, origin_links, destination_links):
    ol = [detailed_to_coarse_link_id.get(str(link_id), link_id) for link_id in origin_links]
    dl = [detailed_to_coarse_link_id.get(str(link_id), link_id) for link_id in destination_links]

    # Candidate origin links end at their to_node
    origin_df = network.links.loc[network.links["link_id"].isin(ol),["link_id", "to_node"]]

    # Candidate destination links start at their from_node
    destination_df = network.links.loc[network.links["link_id"].isin(dl),["link_id", "from_node"]]

    best_origin = None
    best_destination = None
    best_distance = np.inf

    for _, o in origin_df.iterrows():
        o_idx = node_to_idx.get(o["to_node"])
        if o_idx is None:
            continue

        for _, d in destination_df.iterrows():
            d_idx = node_to_idx.get(d["from_node"])
            if d_idx is None:
                continue

            try:
                dist = graph.shortest_path_length(o_idx, d_idx, imp_name="travel_time")

                if np.isfinite(dist) and dist < best_distance:
                    best_distance = dist
                    best_origin = o["link_id"]
                    best_destination = d["link_id"]

            except Exception as e:
                logger.warning(f"Failed routing {o['link_id']} -> {d['link_id']}: {e}")

    if best_origin is not None and best_destination is not None:
        best_origin = origin_links[ol.index(best_origin)]
        best_destination = destination_links[dl.index(best_destination)]

    return best_origin, best_destination, best_distance


def get_detailed_to_coarse_link_id(network):
    detailed = network.links["attributes"].apply(lambda x: x.get("old_link_id", np.nan))
    coarse = network.links["link_id"]
    mapping = pd.DataFrame({"detailed_link_id":detailed, "coarse_link_id":coarse})
    mapping = mapping.dropna(subset=["detailed_link_id"])
    mapping["detailed_link_id"] = mapping["detailed_link_id"].astype(str).str.split('_')
    mapping = mapping.explode("detailed_link_id").reset_index(drop=True)
    return mapping.set_index("detailed_link_id")["coarse_link_id"].to_dict()


def match(matched_links, network, graph, node_to_idx, detailed_to_coarse_link_id, aprr, area, aprr_area_open_system):
    # first, get prices of open system
    aprr_area_open_system["name_norm"] = aprr_area_open_system["name"].apply(normalize_station_name)
    dict_prices_open_system = aprr_area_open_system.set_index("name_norm")["price"].to_dict()
    matched_links["name_norm"] = matched_links["name"].apply(normalize_station_name)
    matched_links["price"] = matched_links[["name_norm","price"]].apply(lambda x: dict_prices_open_system.get(x["name_norm"], x["price"]), axis=1)
    
    # there are tolls in open system (atmb), prices already in the dataframe, and other need to be linked to the prices dataframe
    open_system = matched_links.loc[matched_links.price.notna(), ["link_id", "price"]].dropna(subset=["link_id"])
    open_system = open_system.drop_duplicates().reset_index(drop=True)
    unsolved = matched_links.loc[matched_links.price.isna()].reset_index(drop=True)

    # prices obtained from pdfs & normalize the names
    prices = pd.concat([aprr, area], ignore_index=True)
    prices["origin_norm"] = prices["origin"].apply(normalize_station_name)
    prices["destination_norm"] = prices["destination"].apply(normalize_station_name)

    unsolved = unsolved[["location_geom","link_id","geometry","name_norm"]]
    prices = prices[["origin_norm","destination_norm","distance","price"]]

    # start merging
    prices = prices.dropna(subset=["origin_norm", "destination_norm"])

    # Build lookup: each normalized name -> list of candidate link_ids
    link_lookup = unsolved.groupby("name_norm")["link_id"].apply(lambda x: sorted(set(x))).to_dict()
    available_names = list(link_lookup.keys())
    prices = prices[prices["origin_norm"].isin(available_names) & prices["destination_norm"].isin(available_names)]

    results = []
    for _, row in prices.iterrows():
        origin_norm = row["origin_norm"]
        destination_norm = row["destination_norm"]

        if pd.isna(origin_norm) or pd.isna(destination_norm):
            continue

        origin_links = link_lookup.get(origin_norm, [])
        destination_links = link_lookup.get(destination_norm, [])

        if len(origin_links) == 0 or len(destination_links) == 0:
            logger.warning(
                "No matched links for origin='%s' or destination='%s'",
                origin_norm,
                destination_norm,
            )
            continue

        origin_link, destination_link, tt = get_correct_links(graph, network, node_to_idx, detailed_to_coarse_link_id, origin_links, destination_links)

        if origin_link is None or destination_link is None:
            logger.warning(
                "Routing could not select valid links for origin='%s', destination='%s'",
                origin_norm,
                destination_norm
            )
            continue

        results.append({
            **row.to_dict(),
            "origin_link_id": origin_link,
            "destination_link_id": destination_link,
            "travel_time": tt,
        })

    closed_system = pd.DataFrame(results)

    return open_system, closed_system

def resolve_overlapping_links_problems(open_system, closed_system, network, detailed_network, node_to_idx, detailed_to_coarse_link_id):
    """
    This function resolves problems where links (from the network) are overlapping, meaning they have the same geometry,  but in opposite directions.
    Thus, a different link_id. We need to make sure in that case the correct point is assigned to the correct link_id
    """

    return open_system, closed_system

