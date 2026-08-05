import numpy as np
import matplotlib.pyplot as plt


def plot_national_comparison(context, reference_low, reference_high, syn):
    x = 0

    width = 0.35

    _, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(x + width / 2, syn, width, label = "Synthetic travel demand",
           color = "#1A4390", zorder=3 )
    ax.bar(x - width/2, reference_low, width, label = "BFS private cars",
           color = "#757575", zorder=3 )
    ax.bar(x - width/2, reference_high - reference_low, width, bottom = reference_low, label = "BFS company cars",
           color = "#C0C0C0", zorder=3 )
    
    ax.set_xlabel("")
    ax.set_ylabel("Number of cars")
    ax.set_title("Number of cars in Switzerland - BFS data vs. STATPOP + MZ model")
    ax.set_xticks([x])
    ax.set_xticklabels(["Switzerland"], rotation=90)
    ax.legend()
    ax.grid(True, alpha = 0.3, axis = "y", linestyle = "--")
    
    plt.xlim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(f"{context.path()}/national_comparison_nopxtspt.pdf")
    plt.close()


def plot_cantonal_comparison(context, df):
    df = df.sort_values("cars_syn", ascending = False)
    df["canton_name"] = df["canton_name"].str.split(" / ").str[0]
    df["canton_name"] = df["canton_name"].replace({"Appenzell Ausserrhoden": "Appenzell  A.Rh.",
                                                                           "Appenzell Innerrhoden": "Appenzell I.Rh."})
    x = np.arange(len(df))
    width = 0.35

    df["private_cars_ref"] = df["cars_private_ref"]
    df["company_cars_ref"] = df["cars_all_ref"] - df["cars_private_ref"]

    _, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x + width / 2, df["cars_syn"], width, label = "Synthetic travel demand", 
           color = "#1A4390", zorder=3 )
    ax.bar(x - width / 2, df["private_cars_ref"], width, label = "BFS private cars", 
            color = "#757575", zorder=3 )
    ax.bar(x - width / 2, df["company_cars_ref"], width, bottom = df["private_cars_ref"], 
            label = "BFS company cars", color = "#C0C0C0", zorder=3 )
    
    ax.set_xlabel("Cantons")
    ax.set_ylabel("Number of cars")
    ax.set_title("Number of cars by canton - BFS data vs. STATPOP + MZ model")
    ax.set_xticks(x)
    ax.set_xticklabels(df["canton_name"], rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3, axis = "y", linestyle = "--")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/cantonal_comparison_nopxtspt.pdf")
    plt.close()

    # Normalize
    df["syn_ratio"]   = df["cars_syn"] / df["cars_private_ref"]
    df["upper_ratio"] = df["cars_all_ref"] / df["cars_private_ref"]

    _, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x + width/2, df["syn_ratio"], width, 
                    label = "Synthetic ratio", color = "#1A4390", zorder=3)
    ax.bar(x - width/2, 1.0, width, label = "BFS private cars", 
            color = "#757575", zorder=3 )
    ax.bar(x - width/2, df["upper_ratio"] - 1.0, width, bottom=1.0,
            label = "BFS company cars", color = "#C0C0C0", zorder=3 )

    ax.set_xlabel("Cantons")
    ax.set_ylabel("Ratio (relative to private_ref)")
    ax.set_title("Normalized car counts by canton")
    ax.set_xticks(x)
    ax.set_xticklabels(df["canton_name"], rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3, axis = "y", linestyle = "--")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/cantonal_comparison_normalized_nopxtspt.pdf")
    plt.close()


def plot_municipality_comparison(context, df):
    df = df[df["cars_private_ref"] >= 10000]
    df = df.sort_values("cars_syn", ascending = False)

    df["private_cars_ref"] = df["cars_private_ref"]
    df["company_cars_ref"] = df["cars_all_ref"] - df["cars_private_ref"]

    x = np.arange(len(df))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x + width / 2, df["cars_syn"], width, label = "Synthetic travel demand", 
           color = "#1A4390", zorder=3 )
    ax.bar(x - width / 2, df["private_cars_ref"], width, label = "BFS private cars", 
            color = "#757575", zorder=3 )
    ax.bar(x - width / 2, df["company_cars_ref"], width, bottom = df["private_cars_ref"], 
            label = "BFS company cars", color = "#C0C0C0", zorder=3 )

    ax.set_xlabel("Municipalities")
    ax.set_ylabel("Number of cars")
    ax.set_title("Number of cars by municipality - BFS data vs. STATPOP + MZ model\n Municipalities with more than 10'000 private cars according to BFS")
    ax.set_xticks(x)
    ax.set_xticklabels(df["municipality_name"], rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3, axis = "y", linestyle = "--")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/municipality_comparison_nopxtspt.pdf")
    plt.close()

    # Normalized
    df["syn_ratio"]   = df["cars_syn"] / df["cars_private_ref"]
    df["upper_ratio"] = df["cars_all_ref"] / df["cars_private_ref"]

    x = np.arange(len(df))

    _, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x + width/2, df["syn_ratio"], width, 
                    label = "Synthetic ratio", color = "#1A4390", zorder=3)
    ax.bar(x - width/2, 1.0, width, label = "BFS private cars", 
            color = "#757575", zorder=3 )
    ax.bar(x - width/2, df["upper_ratio"] - 1.0, width, bottom=1.0,
            label = "BFS company cars", color = "#C0C0C0", zorder=3 )
    
    ax.set_xlabel("Municipalities")
    ax.set_ylabel("Ratio (relative to private_ref)")
    ax.set_title("Normalized car counts by municipality")
    ax.set_xticks(x)
    ax.set_xticklabels(df["municipality_name"], rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3, axis = "y", linestyle = "--")

    plt.tight_layout()
    plt.savefig(f"{context.path()}/municipality_comparison_normalized_nopxtspt.pdf")
    plt.close()


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

