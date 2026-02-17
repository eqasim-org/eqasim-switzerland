import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_national_comparison(context, reference_low, reference_high, syn):
    x = 0

    plt.figure(figsize=(8, 6))
    plt.bar(x, syn, color = "#596B8D")
    plt.vlines(x, reference_low, reference_high, colors = "#000000")

    plt.xticks([x], ["Switzerland"])

    plt.ylabel("Number of cars")
    plt.title("Number of cars in Switzerland - BFS data vs. STATPOP + MZ model")

    plt.xlim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(f"{context.path()}/national_comparison_nopxtspt.png")
    plt.close()


def plot_cantonal_comparison(context, df):
    df = df.sort_values("cars_syn", ascending = False)

    x = np.arange(len(df))

    plt.figure(figsize=(12, 6))
    plt.bar(x, df["cars_syn"], color = "#596B8D")

    plt.vlines(x, df["cars_private_ref"], df["cars_all_ref"], colors = "#000000")

    plt.xticks(x, df["canton_name"], rotation=90)

    plt.ylabel("Number of cars")
    plt.title("Number of cars by canton - BFS data vs. STATPOP + MZ model")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/cantonal_comparison_nopxtspt.png")

    # Normalize
    df["syn_ratio"]   = df["cars_syn"] / df["cars_private_ref"]
    df["upper_ratio"] = df["cars_all_ref"] / df["cars_private_ref"]

    x = np.arange(len(df))

    plt.figure(figsize=(12, 6))

    plt.bar(x, df["syn_ratio"], color = "#596B8D")

    plt.vlines(x, 1, df["upper_ratio"], colors = "#000000")

    plt.xticks(x, df["canton_name"], rotation=90)
    plt.ylabel("Ratio (relative to private_ref)")
    plt.title("Normalized car counts by canton")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/cantonal_comparison_normalized_nopxtspt.png")


def plot_municipality_comparison(context, df):
    df = df[df["cars_private_ref"] >= 10000]
    df = df.sort_values("cars_syn", ascending = False)

    x = np.arange(len(df))

    plt.figure(figsize=(12, 6))
    plt.bar(x, df["cars_syn"], color = "#596B8D")

    plt.vlines(x, df["cars_private_ref"], df["cars_all_ref"], colors = "#000000")

    plt.xticks(x, df["municipality_name"], rotation=90)

    plt.ylabel("Number of cars")
    plt.title("Number of cars by municipality - BFS data vs. STATPOP + MZ model\n Municipalities with more than 10'000 private cars according to BFS")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/municipality_comparison_nopxtspt.png")

    # Normalized
    df["syn_ratio"]   = df["cars_syn"] / df["cars_private_ref"]
    df["upper_ratio"] = df["cars_all_ref"] / df["cars_private_ref"]

    x = np.arange(len(df))

    plt.figure(figsize=(12, 6))

    plt.bar(x, df["syn_ratio"], color = "#596B8D")

    plt.vlines(x, 1, df["upper_ratio"], colors = "#000000")

    plt.xticks(x, df["municipality_name"], rotation=90)
    plt.ylabel("Ratio (relative to private_ref)")
    plt.title("Normalized car counts by municipality\n Municipalities with more than 10'000 private cars according to BFS")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/municipality_comparison_normalized_nopxtspt.png")


def configure(context):
    context.stage("data.vehicle_counts.vehicle_counts")
    context.stage("synthesis.population.models.carownership")
    context.stage("data.spatial.municipalities")


def execute(context):

    reference_data = context.stage("data.vehicle_counts.vehicle_counts").copy()
    synthetic_data = context.stage("synthesis.population.models.carownership").copy()
    municipalities = context.stage("data.spatial.municipalities")[0].copy()[["municipality_id", "municipality_name"]]

    synthetic_data = synthetic_data[["number_of_cars_class", "home_municipality_id", "canton_id", "household_id"]].drop_duplicates()
    del synthetic_data["household_id"]
    
    syn_national = synthetic_data["number_of_cars_class"].sum() 
    syn_canton   = synthetic_data.groupby("canton_id", as_index = False)["number_of_cars_class"].sum().rename(columns = {"number_of_cars_class": "cars_syn"})
    syn_mun      = synthetic_data.groupby("home_municipality_id", as_index = False)["number_of_cars_class"].sum().rename(columns = {"number_of_cars_class": "cars_syn"})

    ref_national = [reference_data["cars_like_total"].sum(), reference_data["cars_like_person"].sum()]
    ref_canton   = reference_data.groupby(["canton_id", "canton_name"], as_index = False).agg(
        cars_all_ref = ("cars_like_total", "sum"),
        cars_private_ref = ("cars_like_person", "sum")
    )
    ref_mun      = reference_data.groupby(["municipality_id"], as_index = False).agg(
        cars_all_ref = ("cars_like_total", "sum"),
        cars_private_ref = ("cars_like_person", "sum")
    )

    syn_canton["canton_id"] = syn_canton["canton_id"].astype(int)
    syn_mun["home_municipality_id"] = syn_mun["home_municipality_id"].astype(int)

    cantonal_comparison     = ref_canton.merge(syn_canton, on = "canton_id", how = "outer")
    municipality_comparison = ref_mun.merge(syn_mun, left_on = "municipality_id", right_on = "home_municipality_id", how = "left")
    municipality_comparison = municipality_comparison.merge(municipalities, on = "municipality_id", how = "left")

    # Cantonal comparison
    plot_cantonal_comparison(context, cantonal_comparison.copy())

    # Municipality comparison
    plot_municipality_comparison(context, municipality_comparison.copy())

    # National comparison
    plot_national_comparison(context, ref_national[0], ref_national[1], syn_national)

