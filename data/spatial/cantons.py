import geopandas as gpd
import unicodedata

def configure(context):
    context.config("data_path")


def execute(context):
    # Load data
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/spatial/canton/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp" % data_path,
        encoding="latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    df = df.rename({"KANTONSNUM": "canton_id", "NAME": "canton_name"}, axis=1)
    df = df[["canton_id", "canton_name", "geometry"]]

    df = process_canton_names(df)
    return df


SP_REGION_1 = [25, 12, 13, 1, 2, 14, 9]
SP_REGION_2 = [21, 26, 15, 16, 22, 11, 24, 3, 6, 7]
SP_REGION_3 = [17, 19, 10, 23, 20, 5, 18, 4, 8]


def impute_sp_region(df):
    assert ("canton_id" in df.columns)
    assert ("sp_region" not in df.columns)

    df["sp_region"] = 0
    df.loc[df["canton_id"].isin(SP_REGION_1), "sp_region"] = 1
    df.loc[df["canton_id"].isin(SP_REGION_2), "sp_region"] = 2
    df.loc[df["canton_id"].isin(SP_REGION_3), "sp_region"] = 3

    # TODO: There are some municipalities that are not included in the shape
    # file above. Hence, they get region 0. Should be fixed in the future.
    # Especially, we need a consistent spatial system. It probably would make
    # more sense to impute the SP region in another way

    # assert(not np.any(df["sp_region"] == 0))
    return df

def process_canton_names(df, col='canton_name', out_col='canton_name_en'):
    def fix_and_ascii(s):
        try:
            fixed = s.encode('latin1').decode('utf8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            fixed = s

        norm = unicodedata.normalize('NFKD', fixed)
        ascii_s = norm.encode('ascii', 'ignore').decode('ascii')
        ascii_s = ascii_s.strip().lower()
        ascii_s = ascii_s.replace(' ', '_').replace(".", "").replace('-', '_').replace("'", "")
        return ascii_s

    df = df.copy()
    df[out_col] = df[col].apply(fix_and_ascii)
    
    english_map = {
        'graubunden': 'grisons',
        'zuerich': 'zurich',
        'neuchatel': 'neuchatel',
        'geneve': 'geneva'
    }

    # normalize english_map keys to the same form produced by fix_and_ascii
    normalized_map = {fix_and_ascii(k): v for k, v in english_map.items()}
    df[out_col] = df[out_col].replace(normalized_map)

    return df
