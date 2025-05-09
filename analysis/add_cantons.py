import pandas as pd
import geopandas as gpd
import pandas as pd
import geopandas as gpd
import unicodedata

def remove_accents(text):
    if isinstance(text, str):
        return ''.join(
            c for c in unicodedata.normalize('NFKD', text)
            if not unicodedata.combining(c)
        )
    return text

def add_canton_name(dataset, x_col, y_col, coord_system = 2056, distance=3500):
    """
    Adds the cantons of a datapoint based on coordinates.
    Adapted from Andrew 

    - x_col: column for x-coordinate
    - y_col: column for y-coordinate
    """
    if x_col not in dataset.columns or y_col not in dataset.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must exist in the provided file.")

    geojson_path = "/cluster/work/ivt_vpl/anding/data/TLM_KANTONSGEBIET.json"
    canton_boundaries = gpd.read_file(geojson_path).to_crs(epsg=coord_system)

    geometry = gpd.points_from_xy(dataset[x_col], dataset[y_col])

    dataset_gdf = gpd.GeoDataFrame(dataset, geometry=geometry, crs=f"EPSG:{coord_system}")

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


def add_geo(file_path, x_col, y_col, coord_system = 2056, distance=3500, export=True):
    dataset = pd.read_csv(file_path, sep=';')

    if x_col not in dataset.columns or y_col not in dataset.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must exist in the provided file.")

    geojson_path = "/cluster/work/ivt_vpl/anding/data/TLM_KANTONSGEBIET.json"
    canton_boundaries = gpd.read_file(geojson_path).to_crs(epsg=coord_system)

    geometry = gpd.points_from_xy(dataset[x_col], dataset[y_col])

    dataset_gdf = gpd.GeoDataFrame(dataset, geometry=geometry, crs=f"EPSG:{coord_system}")

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

    if export:
        print("Exporting results...")

        output_path = file_path.replace(".csv", "_geo.csv")
        result_filt.to_csv(output_path, index=False)

        print(f"DataFrame written to: {output_path}")

    return pd.DataFrame(result_filt)

def main():
    file_path = input("Enter the file path to the CSV file: ")
    
    x_col = input("Enter the column name for the x-coordinate: ").strip()
    y_col = input("Enter the column name for the y-coordinate: ").strip()

    coord_system_input = input("Enter the EPSG code for the coordinate system (default: 2056): ").strip()
    try:
        coord_system = int(coord_system_input) if coord_system_input else 2056
    except ValueError:
        print("Invalid EPSG code input. Using default value of 2056.")
        coord_system = 2056

    distance = input("Enter maximum distance for matching (default: 3500): ").strip()
    distance = int(distance) if distance.isdigit() else 3500

    try:
        result = add_geo(file_path, x_col=x_col, y_col=y_col, coord_system = coord_system, distance=distance)
        print("Processing complete!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()