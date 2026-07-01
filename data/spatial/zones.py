import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.spatial.countries")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.nuts")
    context.stage("data.spatial.postal_codes")
    context.config("include_external_population", default = False)
    context.stage("data.external_population.constants")

def execute(context):
    df_countries = pd.DataFrame(context.stage("data.spatial.countries"), copy = True)
    df_municipalities = pd.DataFrame(context.stage("data.spatial.municipalities")[0], copy = True)
    df_quarters = pd.DataFrame(context.stage("data.spatial.quarters"), copy = True)
    df_nuts = pd.DataFrame(context.stage("data.spatial.nuts"), copy=True)
    df_postal_code = pd.DataFrame(context.stage("data.spatial.postal_codes"), copy=True)
    
    # remove outside municipalities
    if context.config("include_external_population"):
        cst = context.stage("data.external_population.constants")
        df_municipalities = df_municipalities[df_municipalities["municipality_id"] != cst.municipality_id].reset_index(drop=True)

    df_countries["zone_level_id"] = df_countries["country_id"]
    df_municipalities["zone_level_id"] = df_municipalities["municipality_id"]
    df_quarters["zone_level_id"] = df_quarters["quarter_id"]
    df_nuts["zone_level_id"] = df_nuts["nuts_id"]
    df_postal_code["zone_level_id"] = df_postal_code["postal_code"]

    df_countries["zone_name"] = df_countries["country_name"]
    df_municipalities["zone_name"] = df_municipalities["municipality_name"]
    df_quarters["zone_name"] = df_quarters["quarter_name"]
    df_nuts["zone_name"] = df_nuts["nuts_name"]
    df_postal_code["zone_name"] = df_postal_code["postal_code"]

    df_countries["zone_level"] = "country"
    df_municipalities["zone_level"] = "municipality"
    df_quarters["zone_level"] = "quarter"
    df_nuts["zone_level"] = "nuts"
    for level in df_nuts["nuts_level"].unique():
        df_nuts.loc[df_nuts["nuts_level"] == level, "zone_level"] = ("nuts_" + str(level))
    df_postal_code["zone_level"] = "postal_code"

    df_zones = pd.concat([
        df_countries[["zone_level_id", "zone_name", "zone_level"]],
        df_municipalities[["zone_level_id", "zone_name", "zone_level"]],
        df_quarters[["zone_level_id", "zone_name", "zone_level"]],
        df_nuts[["zone_level_id", "zone_name", "zone_level"]],
        df_postal_code[["zone_level_id", "zone_name", "zone_level"]],
    ])

    df_zones.loc[:, "zone_id"] = np.arange(len(df_zones))
    df_zones["zone_level"] = df_zones["zone_level"].astype("category")

    return df_zones[["zone_id", "zone_name", "zone_level", "zone_level_id"]]

def impute(df, df_zones, zone_id_prefix = "",
           zone_fields={"quarter": "quarter_id",
                        "municipality": "municipality_id",
                        "country": "country_id",
                        "nuts": "nuts_id",
                        "postal_code": "postal_code"}):
    logger.info(f"Imputing {list(zone_fields.keys())} zones for {len(df)} points...")
    remaining_mask = np.ones((len(df),), dtype = bool)
    df[zone_id_prefix + "zone_id"] = np.nan
    df[zone_id_prefix + "zone_level"] = np.nan
    
    for zone_level, zone_id_field in zone_fields.items():
        if zone_id_field in df:
            
            # Create filters
            f = ~pd.isnull(df[zone_id_field]) & remaining_mask            
            f_zone = df_zones["zone_level"] == zone_level
            
            # Create temp df to merge
            temp1 = df.loc[f, [zone_id_field]].rename({zone_id_field: "zone_level_id"}, axis=1).reset_index(drop=True)
            temp2 = df_zones.loc[f_zone, ["zone_level_id", "zone_id", "zone_level"]].reset_index(drop=True)

            # Merge
            df_join = pd.merge(temp1, temp2, how = "left", on = "zone_level_id")
            
            # Add to df
            if np.sum(f) == len(df):
                df[zone_id_prefix + "zone_id"] = df_join["zone_id"].values
                df[zone_id_prefix + "zone_level"] = df_join["zone_level"].values
            else:
                df.loc[f, zone_id_prefix + "zone_id"] = df_join["zone_id"].values
                df.loc[f, zone_id_prefix + "zone_level"] = df_join["zone_level"].values
                
            # Compute remaining mask
            remaining_mask &= pd.isnull(df[zone_id_prefix + "zone_id"])
            count = np.count_nonzero(df[zone_id_prefix + "zone_level"] == zone_level)
            logger.info("  Found zone of type %s for %d points", zone_level, count)
            
        else:
            logger.warning("  Id %s for zone type %s not found in dataframe", zone_id_field, zone_level)
        
    unknown_count = np.count_nonzero(pd.isnull(df[zone_id_prefix + "zone_id"]))

    if unknown_count > 0:
        logger.warning("  No information for %d observations", unknown_count)
        
    return df
