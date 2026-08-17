import logging
import geopandas as gpd
import osmium
from pathlib import Path
from multiprocessing import get_context

logger = logging.getLogger("synpp")

def configure(context):
    context.config("osm_corrected_files", default=["geneva_segments.geojson", "lausanne_osm_roads_cleaned.geojson"])
    context.config("data_path")


def execute(context):
    osm_corrected_files = context.config("osm_corrected_files")
    data_path = context.config("data_path")

    if not isinstance(osm_corrected_files, list):
        osm_corrected_files = [osm_corrected_files]

    corrected_file_paths = [Path(data_path) / "osm" / f for f in osm_corrected_files]
    logger.info(f"Processing {len(corrected_file_paths)} corrected OSM files for speed corrections:")
    for f in corrected_file_paths:
        assert f.exists(), f"Corrected OSM file {f} does not exist."
        logger.info(f"\t OSM ways speed correction file: {f}.")

    # get the dictionary of speeds (process the files in parallel if there are multiple files)
    n_processes = min(len(corrected_file_paths), 4)  # Limit to 4 processes or the number of files

    mp_ctx = get_context("spawn")
    with mp_ctx.Pool(processes=n_processes, maxtasksperchild=1) as pool:
        results = pool.imap_unordered(process_file, corrected_file_paths, chunksize=1)
        speed_dict = [res for res in context.progress(results, total=len(corrected_file_paths),
                                              label=f"Processing OSM files for traffic lights ({n_processes} parallel processes) ...")]

    # Combine the results into a single dictionary
    combined_speed_dict = {}
    for d in speed_dict:
        combined_speed_dict.update(d["maxspeed"])
    
    logger.info(f"Total number of ways with speed corrections: {len(combined_speed_dict)}")
    return combined_speed_dict




############################# helper functions #############################
def process_file(file_path):
    logger.info(f"Processing file: {file_path}")
    if file_path.suffix == ".pbf":
        # Process PBF file
        return process_pbf_file(file_path)
    elif file_path.suffix == ".geojson":
        # Process GeoJSON file
        return process_geojson_file(file_path)
    else:
        logger.warning(f"Unsupported file format for {file_path}. Skipping.")
        return dict()  # Return an empty DataFrame for unsupported formats


############################# geojson file #########################
def process_geojson_file(file_path):
    logger.info(f"Processing GeoJSON file: {file_path}")
    df = gpd.read_file(file_path)
    if "modified_speed" in df.columns:
        return process_geogeojson_type1(df, file_path)
    elif "implicit_kmh" in df.columns:
        return process_geogeojson_type2(df, file_path)
    else:
        raise ValueError(f"GeoJSON file {file_path} does not contain 'modified_speed' or 'implicit_kmh' columns.")

def process_geogeojson_type1(df, file_path):
    df = df[df["modified_speed"].notnull()]
    df = df[df["modified_speed"].astype(bool)]
    df = df[df["maxspeed"].notnull()]
    df = df[df["maxspeed"]>0].reset_index(drop=True)

    if len(df) != df["osm_id"].nunique():
        logger.warning(f"Duplicate osm_id values found in GeoJSON file {file_path}.")
        df = df[~df["osm_id"].duplicated(keep=False)]
    
    df = df[["osm_id","maxspeed"]]
    df["osm_id"] = df["osm_id"].astype(int)
    return df[["osm_id","maxspeed"]].set_index("osm_id").to_dict()

def process_geogeojson_type2(df, file_path):
    df = df[df["implicit_kmh"].notnull()]
    df = df[df["implicit_kmh"] != df["explicit_kmh"]]
    df = df[df["implicit_kmh"] >0].reset_index(drop=True)

    if len(df) != df["osm_id"].nunique():
        logger.warning(
            f"Duplicate osm_id values found in GeoJSON file {file_path}. "
            "Selecting implicit_kmh for the longest segment."
        )

        df = (df.loc[df.groupby("osm_id")["seg_len"].idxmax(), ["osm_id", "implicit_kmh"]]
                .reset_index(drop=True))

    df["osm_id"] = df["osm_id"].astype(int)
    df = df.rename(columns={"implicit_kmh": "maxspeed"})
    return df[["osm_id","maxspeed"]].set_index("osm_id").to_dict()






############################# pbf file #########################
def process_pbf_file(file_path):
    assert file_path.suffix == ".pbf", "File must be a .pbf file"
    logger.info(f"Processing PBF file: {file_path}")
    
    osm_reader = OSMSpeedReader()
    osm_reader.apply_file(file_path)

    return osm_reader.speeds

# Read OSM maxspeed values
class OSMSpeedReader(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.speeds = {}

    def way(self, w):
        if "maxspeed" in w.tags:
            self.speeds[w.id] = w.tags["maxspeed"]

