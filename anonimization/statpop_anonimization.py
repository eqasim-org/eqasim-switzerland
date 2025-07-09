import pickle
import os 
import numpy as np
import pandas as pd
import geopandas as gpd
import unicodedata
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm

def remove_accents(text):
    if isinstance(text, str):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    return text

def add_wgs84_coordinates(gdf, drop_geometry=True):
    gdf = gdf.copy()
    
    # convert geometry to WGS84
    wgs84_geom = gdf.geometry.to_crs(epsg=4326)
    
    gdf['longitude'] = wgs84_geom.x
    gdf['latitude'] = wgs84_geom.y
    
    if drop_geometry:
        gdf = gdf.drop(columns=['geometry'])
    
    return gdf

def add_canton_name(dataset, x_col, y_col, coord_system=2056, distance=3500, add_coords=False):
    """
    Adds the cantons of a datapoint based on coordinates.
    Adapted from Andrew 

    Args:
        x_col: column for x-coordinate
        y_col: column for y-coordinate
        coord_system: input coordinate system (default 2056 for LV95)
        distance: maximum distance for nearest canton matching
        add_coords: if True, adds longitude/latitude columns
    """
    if x_col not in dataset.columns or y_col not in dataset.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must exist in the provided file.")

    geojson_path = "/cluster/work/ivt_vpl/anding/data/TLM_KANTONSGEBIET.json"
    canton_boundaries = gpd.read_file(geojson_path).to_crs(epsg=coord_system)

    geometry = gpd.points_from_xy(dataset[x_col], dataset[y_col])

    dataset_gdf = gpd.GeoDataFrame(dataset, geometry=geometry, crs=f"EPSG:{coord_system}")
    
    if add_coords:
        dataset_gdf = add_wgs84_coordinates(dataset_gdf, drop_geometry=False)

    print("Finished assigning points!")

    within_canton = dataset_gdf.sjoin(canton_boundaries[['KANTONSNUMMER', 'NAME', 'geometry']], how="left", predicate='within')

    print("Finished within canton matches!")
    print("Checking non-matches...")

    non_match = within_canton.loc[within_canton['NAME'].isna()]
    match = within_canton.loc[within_canton['NAME'].notna()]

    non_match = non_match.drop(columns=["index_right", 'KANTONSNUMMER', 'NAME'], errors="ignore")
    match_closest = non_match.sjoin_nearest(canton_boundaries[['KANTONSNUMMER', 'NAME', 'geometry']], how="left", max_distance=distance, distance_col="distance")

    print("Non-matches finished!")
    print("Concatenating results...")

    result = pd.concat([match, match_closest], ignore_index=True)

    result_filt = result.drop(columns=["geometry", "index_right"], errors="ignore")
    result_filt = result_filt.rename(columns={
        "NAME": "canton_name",
        "KANTONSNUMMER": "canton_id"
    })

    missing_matches = len(result_filt.loc[result_filt['canton_name'].isna()])

    if missing_matches > 0:
        print(f'Warning: {missing_matches} trips not assigned a canton (try increasing the distance parameter)')

    assert len(dataset) == len(result_filt), "Input/Output number of rows not matching"

    result_df = pd.DataFrame(result_filt)
    result_df["canton_name"] = result_df["canton_name"].apply(remove_accents)
    return result_df

def process_filename(file_path):
    temp = file_path.split(".")[2:] # skip synthesis.population
    temp[-2] = temp[-2].split("__")[0] # skip the hash the end
    result = "_".join(temp[:-1]) # skip the .p ending
    return result

def get_pkl_data(directory, prefix):
    """
    Reads in the .pkl files and simplifies their names
    """
    files = dict()
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.p'):
            with open(directory + '/' + filename, 'rb') as file:
                data = pickle.load(file)
                processed_name = process_filename(filename)
                files[processed_name] = data
    return files

def preprocess_statpop_files(directory, save_directory=None, prefix='data.statpop.statpop'):
    prefix = 'data.statpop.statpop__4d761c3eb424c15c341c91a7b666fefa'
    print("Reading the .pkl files...")
    files = get_pkl_data(directory, prefix)

    for name, file in files.items(): 
        print(name, type(file), len(file))

        result_df = add_canton_name(file, 'home_x', 'home_y')
        result_df = result_df[result_df['canton_name'] == 'Zurich']
        if save_directory:
            output_path = os.path.join(save_directory, f'statpop_anonimized_zurich.csv')
            result_df.to_csv(output_path, index=False)

def analyze_statpop_data(anon_file_path, original_file_path):
    """
    Basic statistics comparison between statpop anonimized data
    and the original data. This only shows generic characteristics of the dataset.
    """
    print(f"Reading anonymized data from {anon_file_path}...")
    df_anon = pd.read_csv(anon_file_path)
    print(f"Reading original data from {original_file_path}...")
    df_orig = pd.read_csv(original_file_path)
    
    # Basic dataset info
    print("\nBasic Dataset Information:")
    print("                     Anonymized    Original")
    print(f"Total records:      {len(df_anon):,}        {len(df_orig):,}")
    print(f"Unique households:  {df_anon['household_id'].nunique():,}        {df_orig['household_id'].nunique():,}")
    
    # Demographics
    print("\nDemographic Analysis:")
    print("\nGender Distribution (%):")
    gender_anon = df_anon['sex'].value_counts(normalize=True).round(3) * 100
    gender_orig = df_orig['sex'].value_counts(normalize=True).round(3) * 100
    comparison = pd.DataFrame({
        'Anonymized': gender_anon,
        'Original': gender_orig
    })
    print(comparison)
    
    print("\nAge Statistics:")
    age_stats = pd.DataFrame({
        'Anonymized': df_anon['age'].describe().round(1),
        'Original': df_orig['age'].describe().round(1)
    })
    print(age_stats)
    
    print("\nHousehold Size Distribution:")
    household_comp = pd.DataFrame({
        'Anonymized': df_anon['household_size'].value_counts().sort_index(),
        'Original': df_orig['household_size'].value_counts().sort_index()
    })
    print(household_comp)
    
    # Marital Status
    print("\nMarital Status Distribution (%):")
    marital_anon = df_anon['marital_status'].value_counts(normalize=True).round(3) * 100
    marital_orig = df_orig['marital_status'].value_counts(normalize=True).round(3) * 100
    marital_comp = pd.DataFrame({
        'Anonymized': marital_anon,
        'Original': marital_orig
    })
    print(marital_comp)
    
    # Nationality Analysis
    print("\nTop 10 Nationalities Comparison:")
    nationality_comp = pd.DataFrame({
        'Anonymized': df_anon['nationality'].value_counts().head(10),
        'Original': df_orig['nationality'].value_counts().head(10)
    })
    print(nationality_comp)
    
    # Spatial Analysis
    print("\nMunicipality Type Distribution:")
    muni_comp = pd.DataFrame({
        'Anonymized': df_anon['municipality_type'].value_counts(),
        'Original': df_orig['municipality_type'].value_counts()
    })
    print(muni_comp)
    
    print("\nPopulation Density Statistics:")
    density_stats = pd.DataFrame({
        'Anonymized': df_anon['population_density'].describe().round(1),
        'Original': df_orig['population_density'].describe().round(1)
    })
    print(density_stats)
    
    # Household Analysis
    print("\nHousehold Size Distribution (by household heads):")
    hh_heads_anon = df_anon[df_anon['is_head']]['household_size'].value_counts().sort_index()
    hh_heads_orig = df_orig[df_orig['is_head']]['household_size'].value_counts().sort_index()
    hh_comp = pd.DataFrame({
        'Anonymized': hh_heads_anon,
        'Original': hh_heads_orig
    })
    print(hh_comp)
    
    # Spatial Analysis - Calculate and plot distances between households
    distances_df = calculate_household_distances(df_anon, df_orig)
    
    # Calculate statistics per zone
    zone_stats = distances_df.groupby('zone_id')['distance'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    plt.boxplot([distances_df[distances_df['zone_id'] == zone]['distance'] 
                for zone in zone_stats['zone_id'].unique()])
    plt.xlabel('Zone ID')
    plt.ylabel('Distance (meters)')
    plt.title('Distribution of Household Distances by Zone')
    plt.xticks(range(1, len(zone_stats) + 1), zone_stats['zone_id'], rotation=45)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(zone_stats['count'], zone_stats['mean'], alpha=0.5)
    plt.xlabel('Number of Households in Zone')
    plt.ylabel('Mean Distance (meters)')
    plt.title('Mean Distance vs Zone Size')
    
    plt.tight_layout()
    plt.savefig('household_distances_analysis.png')
    plt.close()
    
    print("\nSpatial Analysis Summary:")
    print(f"Total households compared: {len(distances_df):,}")
    print(f"Average distance between original and anonymized locations: {distances_df['distance'].mean():.2f} meters")
    print(f"Median distance: {distances_df['distance'].median():.2f} meters")
    print(f"Standard deviation: {distances_df['distance'].std():.2f} meters")
    
    return df_anon, df_orig

def calculate_household_distances(df_anon, df_orig):
    # Filter and reduce columns
    df_anon = df_anon.loc[df_anon['household_size'] < 3, ['household_id', 'home_x', 'home_y', 'home_zone_id']]
    df_orig = df_orig.loc[df_orig['household_size'] < 3, ['household_id', 'home_x', 'home_y']]

    # Create GeoDataFrames with proper CRS
    gdf_anon = gpd.GeoDataFrame(
        df_anon,
        geometry=gpd.points_from_xy(df_anon.home_x, df_anon.home_y),
        crs="EPSG:2056"
    )
    gdf_anon = gdf_anon.rename_geometry("geometry_anon")

    gdf_orig = gpd.GeoDataFrame(
        df_orig,
        geometry=gpd.points_from_xy(df_orig.home_x, df_orig.home_y),
        crs="EPSG:2056"
    )
    gdf_orig = gdf_orig.rename_geometry("geometry_orig")

    # Merge datasets
    merged_df = pd.merge(
        gdf_anon,
        gdf_orig,
        on='household_id'
    )

    # Calculate distance between geometries (in meters, since EPSG:2056)
    merged_df['distance'] = merged_df.apply(
        lambda row: row.geometry_anon.distance(row.geometry_orig),
        axis=1
    )
    print(merged_df[['household_id', 'distance']].head())
    print(merged_df['distance'].describe())
    return merged_df[['home_zone_id', 'distance']].rename(columns={'home_zone_id': 'zone_id'})


def geo_indistinguishability_noise_meters(epsilon, sensitivity=1000.0):
    """
    Generate noise dx, dy in meters (Swiss coordinate units)
    sensitivity defaults to 1000 meters (1 km) here to scale properly.
    """
    # Sample radius from exponential distribution (in meters)
    r = np.random.exponential(scale=sensitivity / epsilon)

    # Sample angle uniformly
    theta = np.random.uniform(0, 2 * np.pi)

    # Convert polar to Cartesian offsets in meters
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return dx, dy

def test_adding_noise(df):
    df_spatial = pd.DataFrame(df[["person_id", "home_x", "home_y"]])
    df_spatial = gpd.GeoDataFrame(
        df_spatial,
        geometry=gpd.points_from_xy(df_spatial['home_x'], df_spatial['home_y'], crs="EPSG:2056")
    )   

    # add noise to swiss coordinates
    epsilon = 1.0
    dx_dy = [geo_indistinguishability_noise_meters(epsilon) for _ in range(len(df_spatial))]
    dxs, dys = zip(*dx_dy)

    df_spatial['new_home_x'] = df_spatial['home_x'] + np.array(dxs)
    df_spatial['new_home_y'] = df_spatial['home_y'] + np.array(dys)

    # Update geometry to noisy coordinates
    df_spatial = gpd.GeoDataFrame(
        df_spatial,
        geometry=gpd.points_from_xy(df_spatial['home_x'], df_spatial['home_y'], crs="EPSG:2056")
    )   

    # Calculate distances between original and noisy coordinates
    distances = np.sqrt((df_spatial['new_home_x'] - df_spatial['home_x'])**2 + 
                       (df_spatial['new_home_y'] - df_spatial['home_y'])**2)

    # Calculate and print average distance
    avg_distance = distances.mean()
    std_distance = distances.std()
    print(f"Standard deviation of distances: {std_distance:.2f} meters")
    print(f"Average distance between original and noisy coordinates: {avg_distance:.2f} meters")

if __name__ == '__main__':
    # Then analyze both datasets
    anon_file = '/cluster/home/chaoch/ch/ch-zh-synpop/statpop_anonimized_zurich.csv'
    orig_file = '/cluster/home/chaoch/ch/ch-zh-synpop/statpop_original_zurich.csv'
    df_anon, df_orig = analyze_statpop_data(anon_file, orig_file)

    # df_orig = pd.read_csv(orig_file)
    # test_adding_noise(df_orig)
    # preprocess_statpop_files(directory='/cluster/project/cmdp/chaoch/switzerland_data/cache', 
                            #  save_directory='/cluster/home/chaoch/ch/ch-zh-synpop')