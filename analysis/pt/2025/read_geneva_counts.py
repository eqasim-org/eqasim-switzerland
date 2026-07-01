import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import calendar

def configure(context):
    context.config("analysis.pt.tpg_data", default = "TPG_passenger_counts")
    context.config("data_path")


def plot_annual(df, ref_2024, output_path):
    stop_data = df[df["stop_name"] == "Gare Cornavin"].sort_values("bin_idx")

    x    = stop_data["bin_idx"].values
    y    = stop_data["Montees_median"].values
    yerr = np.sqrt(stop_data["Montees_var"].values)

    fig, ax = plt.subplots(figsize=(16, 6))

    ref = ref_2024.sort_values("hour")
    ax.fill_between(ref["hour"], ref["boardings_q10"], ref["boardings_q90"],
                    color="black", alpha=0.12, label="2024 (q10–q90)")
    ax.plot(ref["hour"], ref["boardings_q50"], "-s", color="black",
            linewidth=1.2, markersize=4, label="2024 (q50)")

    ax.fill_between(x, y - yerr, y + yerr, color="steelblue", alpha=0.25, label="2025 (median ± 1 std)")
    ax.plot(x, y, "-o", color="steelblue", linewidth=1.5, markersize=4, label="2025 (median)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:02d}:00" for b in x], ha="right", fontsize=7)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Boardings")
    ax.set_title("Annual boardings at Gare Cornavin (2025 vs 2024)")
    ax.legend(title="Series", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_monthly_variability(df, ref_2024, output_path):
    stop_data = df[df["stop_name"] == "Gare Cornavin"].copy()

    months    = sorted(stop_data["Month_"].unique())
    n_months  = len(months)
    colors    = [cm.tab20(i / max(n_months - 1, 1)) for i in range(n_months)]

    offset_step = 0.06
    offsets     = np.linspace(-(n_months - 1) / 2 * offset_step, (n_months - 1) / 2 * offset_step, n_months)

    fig, ax = plt.subplots(figsize=(16, 6))

    # 2024 reference: shaded q10-q90 band with q50 line
    ref = ref_2024.sort_values("hour")
    ax.fill_between(ref["hour"], ref["boardings_q10"], ref["boardings_q90"],
                    color="black", alpha=0.12, label="2024 (q10–q90)")
    ax.plot(ref["hour"], ref["boardings_q50"], "-s", color="black",
            linewidth=1.2, markersize=4, label="2024 (q50)")

    # 2025 monthly series
    for month, color, offset in zip(months, colors, offsets):
        month_data = stop_data[stop_data["Month_"] == month].sort_values("bin_idx")
        x    = month_data["bin_idx"].values
        y    = month_data["Montees_median"].values
        yerr = np.sqrt(month_data["Montees_var"].values)
        label = calendar.month_abbr[int(month)]

        ax.plot(x + offset, y, "-", color=color, linewidth=0.8, alpha=0.7)
        ax.errorbar(x + offset, y, yerr=yerr, fmt="none", color=color,
                    elinewidth=0.7, capsize=2, alpha=0.6)
        ax.plot(x + offset, y, "o", color=color, markersize=2.5, label=label)

    bins = sorted(stop_data["bin_idx"].unique())
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:02d}:00" for b in bins], ha="right", fontsize=7)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Boardings")
    ax.set_title("Monthly variability in boardings at Gare Cornavin (2025 vs 2024)")
    ax.legend(title="Series", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def execute(context):
    data_path = context.config("data_path")

    # Read TPG lines and stop name descriptions
    tpg_path  = data_path + "/" + context.config("analysis.pt.tpg_data")
    tpg_stops = pd.read_csv(f"{tpg_path}/counts2024/tpg_Arrets.csv", encoding = "latin1", sep = ";")

    tpg_stops.columns = ["stop_code", "lon", "lat", "country", "name", "municipality", "gtfs_code", "date1", "date2"]
    tpg_stops = tpg_stops[["stop_code", "gtfs_code", "name"]]
    tpg_stops = tpg_stops[tpg_stops["gtfs_code"].notna()]
    tpg_stops["gtfs_code"] = tpg_stops["gtfs_code"].astype(int).astype(str)

    stop_info = tpg_stops[["stop_code", "gtfs_code", "name"]]

    tpg_monthly_combined = pd.read_csv(f"{tpg_path}/counts2025/monthly_combined.csv")
    tpg_monthly_combined = tpg_monthly_combined.merge(stop_info, left_on="CodeLong", right_on="stop_code", how="left").drop(columns="stop_code")
    tpg_monthly_combined = tpg_monthly_combined.groupby(["gtfs_code", "Month_", "bin_idx"]).agg(
        stop_name        = ("name",             "first"),
        Montees_median   = ("Montees_median",   "sum"),
        Montees_var      = ("Montees_var",      "sum"),
        Descentes_median = ("Descentes_median", "sum"),
        Descentes_var    = ("Descentes_var",    "sum"),
        n_obs            = ("n_obs",            "sum"),
    ).reset_index()

    tpg_monthly_combined.to_csv("/home/asallard/Documents/Tests/Passenger counts/2025/geneva2025monthlyagg.csv", index = False)

    tpg_annual_combined = pd.read_csv(f"{tpg_path}/counts2025/annual_combined.csv")
    tpg_annual_combined = tpg_annual_combined.merge(stop_info, left_on="CodeLong", right_on="stop_code", how="left").drop(columns="stop_code")
    tpg_annual_combined = tpg_annual_combined.groupby(["gtfs_code", "bin_idx"]).agg(
        stop_name        = ("name",             "first"),
        Montees_median   = ("Montees_median",   "sum"),
        Montees_var      = ("Montees_var",      "sum"),
        Descentes_median = ("Descentes_median", "sum"),
        Descentes_var    = ("Descentes_var",    "sum"),
        n_obs            = ("n_obs",            "sum"),
    ).reset_index()

    tpg2024_ref = pd.read_csv(f"{tpg_path}/counts2024/tpg_counts_agg_workdays.csv")

    cornavin_2024 = tpg2024_ref[tpg2024_ref["gtfs_code"] == 8587057].groupby("hour").agg(
        boardings_q10 = ("boardings_raw_q10", "sum"),
        boardings_q50 = ("boardings_raw_q50", "sum"),
        boardings_q90 = ("boardings_raw_q90", "sum"),
    ).reset_index()

    plot_monthly_variability(tpg_monthly_combined, cornavin_2024, "/home/asallard/Documents/Tests/Passenger counts/2025/cornavin_monthly_variability.png")
    plot_annual(tpg_annual_combined, cornavin_2024, "/home/asallard/Documents/Tests/Passenger counts/2025/cornavin_annual.png")
