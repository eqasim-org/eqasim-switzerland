import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

def plot_scatter(x, y, title, xlabel, ylabel, out_path, figsize=(6,6)):
    # valid paired values
    mask = x.notna() & y.notna()
    xv = x[mask].astype(float).values
    yv = y[mask].astype(float).values

    # stats     
    diff = yv - xv
    mean_x = xv.mean()
    mean_y = yv.mean()
    median_diff = np.median(diff)
    mean_diff = diff.mean()
    rmse = np.sqrt((diff ** 2).mean())
    pearson_r, _ = pearsonr(xv, yv)
    r_squared = r2_score(xv, yv)

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    # scatter
    ax.scatter(xv, yv, alpha=0.5, s=2)
    # identity line and limits
    max_val = max(xv.max(), yv.max())
    lim = np.ceil(max_val / 10.0) * 10.0
    ax.plot([0, lim], [0, lim], color='red', linestyle='--', linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.5)

    stats = {
        'N': len(xv),
        'mean_x': mean_x,
        'mean_y': mean_y,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'RMSE': rmse,
        'pearson_r': pearson_r,
        'r_squared': r_squared
    }

    stats_text = (
        f"N = {len(xv)}\n"
        f"mean x = {mean_x:.1f} min\n"
        f"mean y = {mean_y:.1f} min\n"
        f"mean diff = {mean_diff:.1f} min\n"
        f"median diff = {median_diff:.1f} min\n"
        f"RMSE = {rmse:.1f} min\n"
        f"pearson r = {pearson_r:.2f}\n"
        f"r^2 = {r_squared:.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    return stats

def plot_boxplot_by(df, by, value1, value2, title, out_path, xlabel="", figsize=(8,5), between=None):    
    df = pd.DataFrame({
        "x": df[value1],
        "y": df[value2],
        by: df[by]
    }).copy()
    # valid paired values
    df = df[df["x"].notna() & df["y"].notna()].reset_index(drop=True)

    if between is not None:
        df = df[df[by].between(between[0], between[1])].reset_index(drop=True)

    # compute differences
    df["diff"] = (df["y"] - df["x"])/df["x"] * 100  # percentage difference   

    fig, ax = plt.subplots(figsize=figsize)
    df.boxplot(column="diff", by=by, ax=ax, grid=False, patch_artist=True, showfliers=False, 
               boxprops=dict(facecolor='mediumturquoise', color='navy'), medianprops=dict(color='navy'), 
               whiskerprops=dict(color='navy'), capprops=dict(color='navy'))
    ax.set_ylabel("Difference [%]")
    ax.set_xlabel(xlabel)    
    ax.set_title(title)    
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)        
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

def plot_boxplot(x, y, title, out_path, xlabel="", figsize=(6,6), fig_ax = None, show_stats=True, save_image=True):
    # valid paired values
    mask = x.notna() & y.notna()
    xv = x[mask].astype(float).values
    yv = y[mask].astype(float).values
    n = len(xv)

    # stats
    diff = (yv - xv)/xv * 100  # percentage difference   
    mean_x = xv.mean()
    mean_y = yv.mean()
    median_diff = np.median(diff)
    mean_diff = diff.mean()
    rmse = np.sqrt(( (yv - xv) ** 2).mean())
    pearson_r, _ = pearsonr(xv, yv)
    r_squared = r2_score(xv, yv)

    # plot
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # --- boxplot ---
    ax.boxplot(diff, vert=True, patch_artist=True, showfliers=False)
    ax.set_ylabel("Difference [%]")
    ax.set_xlabel(xlabel)  # optional
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)        
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # --- stats annotation ---
    if show_stats:
        stats = {
            'N': n,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'RMSE': rmse,
            'pearson_r': pearson_r,
            'r_squared': r_squared
        }

        stats_text = (
            f"N = {n}\n"
            f"mean x = {mean_x:.1f} min\n"
            f"mean y = {mean_y:.1f} min\n"
            f"mean diff = {mean_diff:.1f} min\n"
            f"median diff = {median_diff:.1f} min\n"
            f"RMSE = {rmse:.1f} min\n"
            f"pearson r = {pearson_r:.2f}\n"
            f"r^2 = {r_squared:.2f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    if save_image:
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

    return stats

def plot_average_by_distance_bin(tt1, tt2, dist, bin_km, title, source1, source2, out_path, xlabel="Distance bin mid (km)", ylabel="Travel time (min)", figsize=(8,5)):
    df = pd.DataFrame({
        "tt1": tt1,
        "tt2": tt2,
        "dist": dist
    })
    # Bin up to 50 km, and put all values >= 50 km into one final bin.
    bin_edges = np.arange(0, 50, bin_km)
    bin_edges = np.append(bin_edges, np.inf)
    df["distance_bin"] = pd.cut(df["dist"], bins=bin_edges, include_lowest=True, right=False)
    
    binned = df.groupby("distance_bin", observed=True).agg(
        x_mid = ("dist", lambda v: v.mean()),
        tt1_mean = ("tt1", "mean"),
        tt1_p10 = ("tt1", lambda v: v.quantile(0.1)),
        tt1_p90 = ("tt1", lambda v: v.quantile(0.9)),
        tt2_mean = ("tt2", "mean"),
        tt2_p10 = ("tt2", lambda v: v.quantile(0.1)),
        tt2_p90 = ("tt2", lambda v: v.quantile(0.9)),
        n = ("tt1", "size")
    ).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(binned["x_mid"], binned["tt1_mean"], label=source1)
    ax.fill_between(binned["x_mid"], binned["tt1_p10"], binned["tt1_p90"], alpha=0.2, label=source1+" (10%–90%)")

    ax.plot(binned["x_mid"], binned["tt2_mean"], label=source2)
    ax.fill_between(binned["x_mid"], binned["tt2_p10"], binned["tt2_p90"], alpha=0.2, label=source2+" (10%–90%)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

def plot_by(df, by, value1, value2, title, source1, source2, out_path, between=None, xlabel="", ylabel="Travel time (min)", figsize=(8,5)):
    df = df.copy()
    if between is not None:
        df = df[df[by].between(between[0], between[1])].reset_index(drop=True)

    binned = df.groupby(by, observed=True).agg(        
        value1_mean = (value1, "mean"),
        value1_p10 = (value1, lambda v: v.quantile(0.1)),
        value1_p90 = (value1, lambda v: v.quantile(0.9)),
        value2_mean = (value2, "mean"),
        value2_p10 = (value2, lambda v: v.quantile(0.1)),
        value2_p90 = (value2, lambda v: v.quantile(0.9)),
        n = (value1, "size")
    ).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(binned[by], binned["value1_mean"], label=source1)
    ax.fill_between(binned[by], binned["value1_p10"], binned["value1_p90"], alpha=0.2, label=source1+" (10%–90%)")
    ax.plot(binned[by], binned["value2_mean"], label=source2)
    ax.fill_between(binned[by], binned["value2_p10"], binned["value2_p90"], alpha=0.2, label=source2+" (10%–90%)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    

def plot_distribution(
    df,
    value,
    by=None,
    title="",
    out_path="",
    xlabel="",
    ylabel="Density",
    figsize=(8, 5),
    bins=30
):
    from scipy.stats import gaussian_kde

    data = df.copy()
    if value not in data.columns:
        raise KeyError(f"Column '{value}' not found in dataframe.")

    # Ensure numeric plotting column
    data[value] = pd.to_numeric(data[value], errors="coerce")

    fig, ax = plt.subplots(figsize=figsize)

    if by is not None:
        data = data.sort_values(by)  # Ensure consistent group order
        
        if by not in data.columns:
            raise KeyError(f"Column '{by}' not found in dataframe.")

        plot_df = data[[value, by]].dropna()
        if plot_df.empty:
            raise ValueError(f"No valid data to plot for '{value}' grouped by '{by}'.")

        groups = sorted(plot_df[by].unique())
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(groups))

        for i, group in enumerate(groups):
            vals = plot_df.loc[plot_df[by] == group, value].to_numpy()
            if vals.size == 0:
                continue

            color = cmap(i)

            # Light histogram for shape
            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.18,
                color=color,
                edgecolor="none"
            )

            # Smooth KDE line when possible
            label = f"{group} (n={vals.size})"
            if np.unique(vals).size > 1:
                x_grid = np.linspace(vals.min(), vals.max(), 400)
                kde = gaussian_kde(vals)
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)
            else:
                ax.axvline(vals[0], color=color, linewidth=2, label=label)

        ax.set_xlabel(xlabel if xlabel else value)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, ncol=2)
    else:
        vals = data[value].dropna().to_numpy()
        if vals.size == 0:
            raise ValueError(f"No valid data to plot for '{value}'.")

        ax.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.28,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
            label="Histogram"
        )

        if np.unique(vals).size > 1:
            x_grid = np.linspace(vals.min(), vals.max(), 400)
            kde = gaussian_kde(vals)
            ax.plot(x_grid, kde(x_grid), color="#1F3B73", linewidth=2.2, label="KDE")

        ax.axvline(np.mean(vals), color="#D62728", linestyle="--", linewidth=1.5, label="Mean")
        ax.axvline(np.median(vals), color="#2CA02C", linestyle=":", linewidth=1.8, label="Median")

        ax.set_xlabel(xlabel if xlabel else value)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)

    # Visual polish
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_link_error_on_network(
    network_gdf,
    routed_df,
    swiss_border,
    out_path,
    title="Network map of MATSim travel-time error",
    figsize=(22, 22)
):
    """Plot link-level MATSim over/underestimation on the network.

    Parameters
    ----------
    network_gdf : GeoDataFrame
        Network links with at least columns ['link_id', 'geometry'].
    routed_df : DataFrame
        Trip dataframe with columns ['links', 'travel_time_min_api', 'travel_time_min_matsim'].
        'links' must contain '-' separated link ids.
    out_path : str
        Output PNG path.
    """
    required_network_cols = {"link_id", "geometry"}
    required_routed_cols = {"links", "travel_time_min_api", "travel_time_min_matsim"}

    if not required_network_cols.issubset(set(network_gdf.columns)):
        raise KeyError(f"network_gdf must contain columns {required_network_cols}")
    if not required_routed_cols.issubset(set(routed_df.columns)):
        raise KeyError(f"routed_df must contain columns {required_routed_cols}")

    df = routed_df.copy()
    df = df[df["links"].notna()].copy()

    # Vote per trip: +1 for overestimation, -1 for underestimation, 0 for equal.
    diff = df["travel_time_min_matsim"] - df["travel_time_min_api"]
    df["trip_vote"] = np.sign(diff).astype(float)
    df = df[df["trip_vote"].notna()].copy()
    if df.empty:
        raise ValueError("No valid routed trips with travel-time error to plot.")

    df["link_id"] = df["links"].astype(str).str.split("-")
    df = df.explode("link_id").dropna(subset=["link_id"]).copy()
    df["link_id"] = df["link_id"].astype(str).str.strip()
    df = df[df["link_id"] != ""].copy()

    # Aggregate votes per traversed link.
    link_stats = df.groupby("link_id", as_index=False).agg(
        vote_sum=("trip_vote", "sum"),
        n_trips=("trip_vote", "size")
    )
    link_stats["signed_error_pct"] = 100.0 * link_stats["vote_sum"] / link_stats["n_trips"].replace(0, np.nan)

    net = network_gdf[["link_id", "geometry"]].copy()
    net["link_id"] = net["link_id"].astype(str)

    # Keep only links inside Swiss border.
    if isinstance(swiss_border, gpd.GeoSeries):
        border_geom = swiss_border.geometry.union_all()
        border_gdf = gpd.GeoDataFrame(geometry=swiss_border)
    elif isinstance(swiss_border, gpd.GeoDataFrame):
        border_geom = swiss_border.geometry.union_all()
        border_gdf = swiss_border[["geometry"]].copy()
    else:
        border_geom = swiss_border
        border_gdf = gpd.GeoDataFrame(geometry=[swiss_border], crs=getattr(network_gdf, "crs", None))

    net = net[net.geometry.within(border_geom)].copy()

    plot_gdf = net.merge(link_stats, on="link_id", how="left")

    fig, ax = plt.subplots(figsize=figsize)

    # Draw inside-border network in light gray and border in black.
    plot_gdf.plot(ax=ax, color="#D9D9D9", linewidth=0.35, alpha=0.8)
    border_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

    used = plot_gdf[plot_gdf["signed_error_pct"].notna()].copy()
    if used.empty:
        raise ValueError("No routed links matched network link ids for plotting.")

    vmax = float(np.nanpercentile(np.abs(used["signed_error_pct"].values), 95))
    vmax = max(vmax, 10.0)
    vmin = -vmax

    used.plot(
        ax=ax,
        column="signed_error_pct",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        linewidth=0.9,
        legend=True,
        legend_kwds={
            "label": "Signed trip error share (%)",
            "shrink": 0.75,
            "orientation": "horizontal",
            "pad": 0.02
        }
    )

    cbar_ax = fig.axes[-1]
    cbar_ax.set_title(
        "Negative: underestimation | Positive: overestimation",
        fontsize=16,
        pad=10
    )
    cbar_ax.tick_params(labelsize=12)

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
