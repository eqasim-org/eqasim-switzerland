"""All plotting functions used by the passenger-count comparison stages."""

import base64
import io

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def categorize_error(row):
    sim  = row["boardings_matsim"]
    mean = row["boardings_raw_mean"]
    std  = row["boardings_raw_std"]
    p10  = row["boardings_raw_q10"]
    p90  = row["boardings_raw_q90"]
    p50  = row["boardings_raw_q50"]

    meanminusstd = mean - std
    meanplusstd  = mean + std

    if sim == 0 and p50 == 0:
        return "no data"
    if meanminusstd <= sim <= meanplusstd:
        return "good"
    elif p10 <= sim < meanminusstd:
        return "moderate low"
    elif meanplusstd < sim <= p90:
        return "moderate high"
    elif sim < p10:
        return "bad low"
    else:
        return "bad high"


def plot_comparison_for_stop_and_line(counts, option = "boardings", line = "1_H", stop = "Genève, gare Cornavin", output_path = ""):
    counts2  = counts.copy()
    px_mvmts = counts2[["stop_name", "line_direction", "hour"] + [c for c in counts if option in c]]

    df_stop_line = px_mvmts[(px_mvmts["stop_name"] == stop) & (px_mvmts["line_direction"] == line)]

    _, ax = plt.subplots(figsize = (10, 6))

    x = df_stop_line["hour"]

    ax.vlines(x, df_stop_line[f"{option}_raw_min"], df_stop_line[f"{option}_raw_max"],
               color = "lightgray", linewidth = 3, label = "Min-Max range")

    ax.hlines(df_stop_line[f"{option}_raw_q10"], x - 0.15, x + 0.15, color = "#0f556d", linewidth = 2, label = "10th percentile")
    ax.hlines(df_stop_line[f"{option}_raw_q50"], x - 0.1,  x + 0.1,  color = "#5c3b13", linewidth = 2, label = "Median (50th)")
    ax.hlines(df_stop_line[f"{option}_raw_q90"], x - 0.15, x + 0.15, color = "#0f556d", linewidth = 2, label = "90th percentile")

    ax.scatter(x, df_stop_line[f"{option}_matsim"], s = 10, color = "black", zorder = 5, label = f"MATSim {option}")

    ax.set_xlim(-1, 24)
    ax.grid()
    ax.set_xlabel("hour")
    ax.set_ylabel(f"Number of {option}")
    ax.set_title(f"Boardings at stop {stop} (line {line})")
    ax.legend(loc = "upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_heatmap_for_line(counts, option = "boardings", line = "1_H", output_path = ""):
    counts2  = counts.copy()
    px_mvmts = counts2[["stop_name", "line_direction", "hour"] + [c for c in counts if option in c]]

    px_mvmts = px_mvmts[px_mvmts["line_direction"] == line]

    px_mvmts["error_category"] = px_mvmts.apply(categorize_error, axis = 1)

    error_order  = ["bad low", "moderate low", "good", "moderate high", "bad high", "no data"]
    error_colors = ["#08306b", "#4292c6", "#95d378", "#fb6a4a", "#99000d", "#ffffff"]
    cmap = mcolors.ListedColormap(error_colors)
    norm = mcolors.BoundaryNorm(boundaries = range(len(error_order) + 1), ncolors = len(error_order))

    stops = sorted(px_mvmts["stop_name"].unique())
    hours = sorted(px_mvmts["hour"].unique())

    cat_to_int = {cat: i for i, cat in enumerate(error_order)}
    matrix = np.full((len(hours), len(stops)), error_order.index("no data"))

    for _, row in px_mvmts.iterrows():
        i = hours.index(row["hour"])
        j = stops.index(row["stop_name"])
        matrix[i, j] = cat_to_int.get(row["error_category"], error_order.index("no data"))

    _, ax = plt.subplots(figsize = (10, 6))
    ax.imshow(matrix, cmap = cmap, norm = norm, aspect = "auto")

    ax.set_xticks(range(len(stops)))
    ax.set_xticklabels(stops, rotation = 90, fontsize = 7)
    ax.set_yticks(range(len(hours)))
    ax.set_yticklabels(hours)
    ax.set_xlabel("Stop")
    ax.set_ylabel("Hour")
    ax.set_title(f"Error heatmap for line {line}")

    handles = [plt.Rectangle((0, 0), 1, 1, color = error_colors[i]) for i in range(len(error_order))]
    ax.legend(handles, error_order, loc = "upper right", bbox_to_anchor = (1.25, 1), title = "Error Category")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_comparison_for_line(counts, option = "boardings", line = "1_H", output_path = ""):
    counts2  = counts.copy()
    px_mvmts = counts2[["stop_name", "line_direction", "hour"] + [c for c in counts.columns if option in c]]

    columns_option = [c for c in px_mvmts.columns if option in c]

    df_stop_line = px_mvmts[(px_mvmts["line_direction"] == line)]
    df_stop_line = df_stop_line.groupby("hour", as_index = False)[columns_option].sum()

    _, ax = plt.subplots(figsize = (10, 6))

    x = df_stop_line["hour"]

    ax.vlines(x, df_stop_line[f"{option}_raw_min"], df_stop_line[f"{option}_raw_max"],
               color = "lightgray", linewidth = 3, label = "Min-Max range")

    ax.hlines(df_stop_line[f"{option}_raw_q10"], x - 0.15, x + 0.15, color = "#0f556d", linewidth = 2, label = "10th percentile")
    ax.hlines(df_stop_line[f"{option}_raw_q50"], x - 0.1,  x + 0.1,  color = "#5c3b13", linewidth = 2, label = "Median (50th)")
    ax.hlines(df_stop_line[f"{option}_raw_q90"], x - 0.15, x + 0.15, color = "#0f556d", linewidth = 2, label = "90th percentile")

    ax.scatter(x, df_stop_line[f"{option}_matsim"], s = 10, color = "black", zorder = 5, label = f"MATSim {option}")

    ax.set_xlim(-1, 24)
    ax.grid()
    ax.set_xlabel("hour")
    ax.set_ylabel(f"Number of {option}")
    ax.set_title(f"Boardings for line {line}")
    ax.legend(loc = "upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_comparison_for_stop(counts, option = "boardings", stop = "Genève, gare Cornavin", output_path = ""):
    counts2  = counts.copy()
    px_mvmts = counts2[["stop_name", "line_direction", "hour"] + [c for c in counts.columns if option in c]]

    columns_option = [c for c in px_mvmts.columns if option in c]

    df_stop_line = px_mvmts[(px_mvmts["stop_name"] == stop)]
    df_stop_line = df_stop_line.groupby("hour", as_index = False)[columns_option].sum()

    _, ax = plt.subplots(figsize = (10, 6))

    x = df_stop_line["hour"]

    ax.vlines(x, df_stop_line[f"{option}_raw_min"], df_stop_line[f"{option}_raw_max"],
               color = "lightgray", linewidth = 3, label = "Min-Max range")

    ax.hlines(df_stop_line[f"{option}_raw_q10"], x - 0.15, x + 0.15, color = "#0f556d", linewidth = 2, label = "10th percentile")
    ax.hlines(df_stop_line[f"{option}_raw_q50"], x - 0.1,  x + 0.1,  color = "#5c3b13", linewidth = 2, label = "Median (50th)")
    ax.hlines(df_stop_line[f"{option}_raw_q90"], x - 0.15, x + 0.15, color = "#0f556d", linewidth = 2, label = "90th percentile")

    ax.scatter(x, df_stop_line[f"{option}_matsim"], s = 10, color = "black", zorder = 5, label = f"MATSim {option}")

    ax.set_xlim(-1, 24)
    ax.grid()
    ax.set_xlabel("hour")
    ax.set_ylabel(f"Number of {option}")
    ax.set_title(f"Boardings for stop {stop}")
    ax.legend(loc = "upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_global_hourly_comparison(global_hourly_df, output_path):
    """
    Total passenger events (boardings + alightings) across all stops in the
    perimeter, by hour of day: scaled MATSim total vs the 2024 TPG 95% CI.
    """

    df = global_hourly_df.sort_values("hour")

    _, ax = plt.subplots(figsize = (12, 6))

    ax.fill_between(df["hour"], df["tpg_lo"], df["tpg_hi"], color = "black", alpha = 0.15, label = "TPG 2024 (95% CI)")
    ax.plot(df["hour"], df["tpg_mean"], "-s", color = "black", linewidth = 1.2, markersize = 4, label = "TPG 2024 (mean)")
    ax.plot(df["hour"], df["matsim_total"], "-o", color = "steelblue", linewidth = 1.5, markersize = 5, label = "MATSim (scaled)")

    ax.set_xticks(df["hour"])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Total passenger events (boardings + alightings)")
    ax.set_title("MATSim vs TPG 2024: total passenger events by hour, perimeter-wide")
    ax.legend()
    ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi = 150, bbox_inches = "tight")
    plt.close()


def render_hourly_chart_png(hourly_df, title):
    """
    Same idea as plot_global_hourly_comparison, but for a single stop or
    line and returned as a base64-encoded PNG (no file written) - meant to
    be embedded straight into an HTML popup, e.g. in interactive_map.py's
    full-day stop map and line map. Only the hours present in hourly_df are
    plotted.
    """

    df = hourly_df.sort_values("hour")

    fig, ax = plt.subplots(figsize = (5, 2.6))

    ax.fill_between(df["hour"], df["tpg_lo"], df["tpg_hi"], color = "black", alpha = 0.15, label = "TPG (95% CI)")
    ax.plot(df["hour"], df["tpg_mean"], "-s", color = "black", linewidth = 1, markersize = 3, label = "TPG (mean)")
    ax.plot(df["hour"], df["matsim_total"], "-o", color = "steelblue", linewidth = 1.2, markersize = 3.5, label = "MATSim (scaled)")

    ax.set_xlabel("Hour of day", fontsize = 8)
    ax.set_ylabel("Passenger events", fontsize = 8)
    ax.set_title(title, fontsize = 9)
    ax.tick_params(labelsize = 7)
    ax.legend(fontsize = 6, loc = "upper left")
    ax.grid(True, alpha = 0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format = "png", dpi = 110)
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("ascii")
