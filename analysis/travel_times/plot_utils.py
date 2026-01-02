import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
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
        "by": df[by]
    }).copy()
    # valid paired values
    df = df[df["x"].notna() & df["y"].notna()].reset_index(drop=True)

    if between is not None:
        df = df[df["by"].between(between[0], between[1])].reset_index(drop=True)

    # compute differences
    df["diff"] = (df["y"] - df["x"])/df["x"] * 100  # percentage difference   

    fig, ax = plt.subplots(figsize=figsize)
    df.boxplot(column="diff", by="by", ax=ax, grid=False, patch_artist=True)
    ax.set_ylabel("Difference [%]")
    ax.set_xlabel(xlabel)  # optional
    ax.set_title(title)
    ax.set_title("")  # remove automatic 'Boxplot grouped by ...' title
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
    ax.boxplot(diff, vert=True, patch_artist=True)
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
    

