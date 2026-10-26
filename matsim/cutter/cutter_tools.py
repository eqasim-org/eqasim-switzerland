import os
import pandas as pd
import geopandas as gpd
from shapely import contains_xy
from matsim.readers import read_network

############ Other functions for the cutter ######################

def cut_and_save_network_geometry(geometry, local_network, output_path, context=None):
    """Keep detailed geometry rows whose LinkId exists in the cut network."""
    df = pd.read_csv(geometry, dtype={"LinkId": str}, keep_default_na=False)
    links = read_network(str(local_network), skip_attributes=True, context=context).links
    df = df.loc[df["LinkId"].isin(links.link_id.astype(str))]
    df.to_csv(output_path, index=False)


def cut_csv_to_region(csv_path, region_path, output_path = None, only_origin=False,
                      origin_x='origin_x', origin_y='origin_y', 
                      destination_x='destination_x', destination_y='destination_y'):
    assert os.path.exists(csv_path)
    assert os.path.exists(region_path)
    
    # Read the file
    df = pd.read_csv(csv_path)
    region = gpd.read_file(region_path).union_all()

    # Check points entirely within the region
    within_region = contains_xy(region, df[origin_x], df[origin_y])
    if not only_origin:
        within_region &= contains_xy(region, df[destination_x], df[destination_y])

    df = df[within_region].reset_index(drop=True)
    assert len(df), f"There is no point left from: {csv_path}, within this region: {region_path}"

    # save or return
    if output_path is not None:
        df.to_csv(output_path, index=False, sep=",")
    else:
        return df


def cut_csv_to_network(csv_path, network_path, output_path = None, link_id_col = "linkId", context=None):
    assert os.path.exists(csv_path)
    assert os.path.exists(network_path)

    # Read the file
    df = pd.read_csv(csv_path)
    links = read_network(network_path, skip_attributes=True, context=context).links

    # Check points entirely within the region
    network_links = set(links.link_id.astype(str).unique())
    within_region = df[link_id_col].astype(str).isin(network_links)

    df = df[within_region].reset_index(drop=True)
    assert len(df), f"There is no point left from: {csv_path}, within this region: {network_path}"

    # save or return
    if output_path is not None:
        df.to_csv(output_path, index=False, sep=",")
    else:
        return df
