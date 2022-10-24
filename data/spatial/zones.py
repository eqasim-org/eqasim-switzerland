import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.spatial.countries")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.nuts")
    context.stage("data.spatial.postal_codes")

def execute(context):
    df_countries = pd.DataFrame(context.stage("data.spatial.countries"), copy = True)
    df_municipalities = pd.DataFrame(context.stage("data.spatial.municipalities")[0], copy = True)
    df_quarters = pd.DataFrame(context.stage("data.spatial.quarters"), copy = True)
    df_nuts = pd.DataFrame(context.stage("data.spatial.nuts"), copy=True)
    df_postal_code = pd.DataFrame(context.stage("data.spatial.postal_codes"), copy=True)

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
           quarter_id_field = "quarter_id", municipality_id_field = "municipality_id", country_id_field = "country_id",
           nuts_id_field = "nuts_id", postal_code_field = "postal_code"):
    print("Imputing %d zones" % len(df))
    remaining_mask = np.ones((len(df),), dtype = np.bool)
    df.loc[:, "zone_id"] = np.nan

    if quarter_id_field in df:
        f = ~pd.isnull(df[quarter_id_field]) & remaining_mask

        df_join = pd.merge(
            df[f][[quarter_id_field]],
            df_zones[df_zones["zone_level"] == "quarter"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = quarter_id_field, right_on = "zone_level_id")

        df.loc[f, zone_id_prefix + "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, zone_id_prefix + "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= pd.isnull(df[zone_id_prefix + "zone_id"])

        print("  Found %d quarters" % np.count_nonzero(df[zone_id_prefix + "zone_level"] == "quarter"))

    if municipality_id_field in df:
        f = ~pd.isnull(df[municipality_id_field]) & remaining_mask

        df_join = pd.merge(
            df[f][[municipality_id_field]],
            df_zones[df_zones["zone_level"] == "municipality"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = municipality_id_field, right_on = "zone_level_id")

        df.loc[f, zone_id_prefix + "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, zone_id_prefix + "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= pd.isnull(df[zone_id_prefix + "zone_id"])

        print("  Found %d municipalities" % np.count_nonzero(df[zone_id_prefix + "zone_level"] == "municipality"))

    if country_id_field in df:
        f = ~pd.isnull(df[country_id_field]) & remaining_mask

        df_join = pd.merge(
            df[f][[country_id_field]],
            df_zones[df_zones["zone_level"] == "country"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = country_id_field, right_on = "zone_level_id")

        df.loc[f, zone_id_prefix + "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, zone_id_prefix + "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= pd.isnull(df[zone_id_prefix + "zone_id"])

        print("  Found %d countries" % np.count_nonzero(df[zone_id_prefix + "zone_level"] == "country"))

    if nuts_id_field in df:
        f = ~pd.isnull(df[nuts_id_field]) & remaining_mask

        df_join = pd.merge(
            df[f][[nuts_id_field]],
            df_zones[df_zones["zone_level"] == "nuts"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = nuts_id_field, right_on = "zone_level_id")

        df.loc[f, zone_id_prefix + "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, zone_id_prefix + "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= pd.isnull(df[zone_id_prefix + "zone_id"])

        print("  Found %d NUTS zones" % np.count_nonzero(df[zone_id_prefix + "zone_level"] == "nuts"))

    if postal_code_field in df:
        f = ~pd.isnull(df[postal_code_field]) & remaining_mask

        df_join = pd.merge(
            df[f][[postal_code_field]],
            df_zones[df_zones["zone_level"] == "postal_code"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = postal_code_field, right_on = "zone_level_id")

        df.loc[f, zone_id_prefix + "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, zone_id_prefix + "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= pd.isnull(df[zone_id_prefix + "zone_id"])

        print("  Found %d postal codes" % np.count_nonzero(df[zone_id_prefix + "zone_level"] == "postal_code"))

    unknown_count = np.count_nonzero(pd.isnull(df[zone_id_prefix + "zone_id"]))

    if unknown_count > 0:
        print("  No information for %d observations" % unknown_count)

    return df
