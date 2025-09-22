import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")


def select_geometry(df, id_col, label):
    # Keep only id + geometry
    out = df[[id_col, "geometry"]].copy()
    out = out.rename(columns={"geometry": f"{label}_geometry"})
    
    # Extract x and y if geometry is a shapely Point
    out[f"{label}X"] = out[f"{label}_geometry"].apply(lambda g: g.x if g is not None else None)
    out[f"{label}Y"] = out[f"{label}_geometry"].apply(lambda g: g.y if g is not None else None)

    out = out.drop(columns=[f"{label}_geometry"])
    
    return out


def classify(row):
    if row["has_work"] and row["has_education"]:
        return "work+education"
    elif row["has_work"]:
        return "work only"
    elif row["has_education"]:
        return "education only"
    else:
        return "none"


def execute(context):
    persons = context.stage("synthesis.population.enriched").copy()
    df_home = context.stage("synthesis.population.spatial.home.locations").copy()
    df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")

    subscription_types = ["ga", "halbtax", "verbund", "strecke", "gleis7", "junior", "other"]
    subscription_cols  = [f"subscriptions_{c}" for c in subscription_types]
    persons = persons[["household_id", "person_id", "age"] + subscription_cols]

    df_home      = select_geometry(df_home, "household_id", "home")
    df_education = select_geometry(df_education, "person_id", "education")
    df_work      = select_geometry(df_work, "person_id", "work")

    merges = [
        (df_home, "household_id"),
        (df_education, "person_id"),
        (df_work, "person_id"),
    ]

    requests = persons.copy()
    for df, key in merges:
        requests = requests.merge(df, on=key, how="left")

    # Boolean column: True if both homeX and homeY are not NA
    requests["has_home"] = requests["homeX"].notna() & requests["homeY"].notna()

    # Quick check
    home_ok = requests["has_home"].all()
    print("All persons have a home location:", home_ok)

    requests["has_work"] = requests["workX"].notna() & requests["workY"].notna()
    requests["has_education"] = requests["educationX"].notna() & requests["educationY"].notna()

    requests["location_type"] = requests.apply(classify, axis=1)

    percentages = (
        requests["location_type"]
        .value_counts(normalize=True) * 100
    ).round(1)

    print(percentages)

    requests.to_csv("/cluster/project/cmdp/asallard/worklocations.csv", index=False)


