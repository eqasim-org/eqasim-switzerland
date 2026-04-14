import os
from .matching.network import RoadNetwork
import glob
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path
import seaborn as sns
from scipy import stats
import warnings
import sys
import logging
import json

logger = logging.getLogger("synpp")

def filter_data(df, network, require_simulated=True):
    # Work on a copy
    df = df.copy()

    # Select link info and attempt to identify lanes column
    net_links = network.links.copy()
    net_links['link_id'] = net_links['link_id'].astype(str)

    # Keep only relevant columns    
    keep_cols = ['link_id', 'highway', 'permlanes']
    net_links = net_links[keep_cols]

    # Merge with the input df
    unique_links = df.link_id.explode().unique()
    link_feat = (net_links[net_links.link_id.isin(unique_links)]
                 .set_index('link_id')
                 .to_dict(orient="index"))
    
    df["highway"] = df.link_id.apply(lambda x: link_feat.get(x[0], {}).get('highway', None))
    df["permlanes"] = df.link_id.apply(lambda x: np.sum([link_feat.get(lk, {}).get('permlanes', -1e5) for lk in (x if isinstance(x, (list, tuple, np.ndarray)) else [x])]))
    df = df[(df.permlanes>0) & (df.highway.notna())]    
    
    # Compute vehicles per HOUR per LANE
    # divide by 24 to get per hour, then divide by number of lanes
    df['obs_vphpl'] = df['flow'] / 24.0 / df['permlanes']
    if require_simulated:
        if 'simulated_flow' not in df.columns:
            logger.info("\t Missing simulated_flow column while require_simulated=True.")
            return None
        df['sim_vphpl'] = df['simulated_flow'] / 24.0 / df['permlanes']

    # Drop rows with NaN vphpl
    if require_simulated:
        df = df[df['obs_vphpl'].notna() | df['sim_vphpl'].notna()].copy()
    else:
        df = df[df['obs_vphpl'].notna()].copy()
    if df.empty:
        logger.info("\t No valid vphpl values after computation.")
        return None
    
    ### FILTERS
    len_before = len(df)  
    # We filter out the data matched to links that are far
    df = df[df.distance.apply(lambda x: max(x)<30)]
    # filter out the data where the distance to links is significantly differente
    # maybe these are one way, but matched with two links
    df = df[df.distance.apply(lambda x: (max(x)-min(x))<15)]
    # filter out those with different highway types
    highways = df.link_id.apply(lambda x: [link_feat.get(xi, {}).get('highway', None) for xi in x])
    df = df[highways.apply(np.unique).apply(len)==1]
    # if the difference in umber of lanes is more than one, then we suspect it, to be removed
    permlanes = df.link_id.apply(lambda x: [link_feat.get(lk, {}).get('permlanes', -1e5) for lk in x])
    df = df[permlanes.apply(lambda x: (max(x)-min(x))<=1)]
    # filter outliers (comes from matching)
    def remove_outliers(group, column):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
    
    df = df.groupby('highway').apply(lambda group: remove_outliers(group, 'obs_vphpl')).reset_index(drop=True)
    if require_simulated:
        df = df.groupby('highway').apply(lambda group: remove_outliers(group, 'sim_vphpl')).reset_index(drop=True)
    len_after = len(df)
    logger.info(f"\t Filtered dataset: {len_after} records remaining (removed {len_before - len_after} records)")
    return df

def compute_statistics(df, flow_col='flow', simulated_flow_col='simulated_flow',
                       output_path=None):
    """Calculate comprehensive statistics for flow comparison."""
    # Remove any rows with missing values
    df_clean = df.dropna(subset=[flow_col, simulated_flow_col])
    
    flow_obs = df_clean[flow_col].values
    flow_sim = df_clean[simulated_flow_col].values

    # Basic statistics
    n_points = len(flow_obs)
    mean_obs = np.mean(flow_obs)
    mean_sim = np.mean(flow_sim)
    
    # R-squared
    r2 = r2_score(flow_obs, flow_sim)    
    # Correlation coefficient
    correlation = np.corrcoef(flow_obs, flow_sim)[0, 1]
    
    # Linear regression for fitting line (no intercept)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(flow_obs.reshape(-1, 1), flow_sim)
    slope = lr.coef_[0]
    intercept = 0.0  # No intercept
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((flow_obs - flow_sim) ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(flow_obs - flow_sim))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((flow_obs - flow_sim) / flow_obs)) * 100
    
    # Percentage underestimation
    underestimation = ((mean_obs - mean_sim) / mean_obs) * 100
    
    # Bias (Mean Error)
    bias = np.mean(flow_sim - flow_obs)
    
    # Nash-Sutcliffe Efficiency
    nse = 1 - (np.sum((flow_obs - flow_sim) ** 2) / np.sum((flow_obs - mean_obs) ** 2))

    # compute goodness-of-fit: GEH Statistic (Geoffrey E. Havers)
    obs_veh_h = (flow_obs / 24.0)/2
    sim_veh_h = (flow_sim / 24.0)/2
    geh_values = np.sqrt(2 * (sim_veh_h - obs_veh_h) ** 2 / (sim_veh_h + obs_veh_h + 1e-6))
    geh_within_5 = int(np.sum(geh_values <= 5))
    geh_within_10 = int(np.sum(geh_values <= 10))
    geh_within_15 = int(np.sum(geh_values <= 15))
    geh_within_25 = int(np.sum(geh_values <= 25))
    geh_within_5_pct = (geh_within_5 / n_points) * 100
    geh_within_10_pct = (geh_within_10 / n_points) * 100
    geh_within_15_pct = (geh_within_15 / n_points) * 100
    geh_within_25_pct = (geh_within_25 / n_points) * 100
    
        
    # scalable quality value (SQV)
    f = 10_000
    sqv = 1/( 1 + np.sqrt(  (flow_sim/2-flow_obs/2)**2/(f*flow_obs/2)  ) )
    sqv_09_pct = round(np.sum(sqv >= 0.9) / n_points * 100, 1)
    sqv_085_pct = round(np.sum(sqv >= 0.85) / n_points * 100, 1)
    sqv_08_pct = round(np.sum(sqv >= 0.8) / n_points * 100, 1)
    sqv_07_pct = round(np.sum(sqv >= 0.7) / n_points * 100, 1)


    stats_dict = {
        'n_points': n_points,
        'r2': r2,
        'correlation': correlation,
        'slope': slope,
        'intercept': intercept,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'bias': bias,
        'nse': nse,
        'mean_observed': mean_obs,
        'mean_simulated': mean_sim,
        'underestimation_percent': underestimation,
        'geh': {
            'within_5': geh_within_5,
            'within_5_pct': geh_within_5_pct,
            'within_10': geh_within_10,
            'within_10_pct': geh_within_10_pct,
            'within_15': geh_within_15,
            'within_15_pct': geh_within_15_pct,
            'within_25': geh_within_25,
            'within_25_pct': geh_within_25_pct,
        },
        'sqv': {
            '0.9': sqv_09_pct,
            '0.85': sqv_085_pct,
            '0.8': sqv_08_pct,
            '0.7': sqv_07_pct,
        }
    }
    # Save statistics to JSON if output path is provided
    if output_path is not None:
        if os.path.exists(output_path):
            logger.info(f"\t Saving the statistics file to {output_path}")
            output_file = os.path.join(output_path, "flow_comparison_stats.json")
            with open(output_file, "w") as f:
                json.dump(stats_dict, f, indent=4)
            logger.info(f"\t Statistics saved to: {output_file}")

    return stats_dict

def print_detailed_statistics(stats_dict):
    """Print detailed statistics to console."""
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED STATISTICS")
    logger.info("=" * 60)
    
    logger.info(f"📊 Dataset Overview:")
    logger.info(f"   • Total data points: {stats_dict['n_points']:,}")
    logger.info(f"   • Mean observed flow: {stats_dict['mean_observed']:,.0f} vehicles/day")
    logger.info(f"   • Mean simulated flow: {stats_dict['mean_simulated']:,.0f} vehicles/day")
    
    logger.info(f"\n🎯 Model Performance:")
    logger.info(f"   • R² (coefficient of determination): {stats_dict['r2']:.4f}")
    logger.info(f"   • Correlation coefficient: {stats_dict['correlation']:.4f}")
    logger.info(f"   • Nash-Sutcliffe Efficiency: {stats_dict['nse']:.4f}")
    
    logger.info(f"\n📏 Fitting Line:")
    logger.info(f"   • Equation: y = {stats_dict['slope']:.4f}x")
    logger.info(f"   • Slope: {stats_dict['slope']:.4f}")
    
    logger.info(f"\n❌ Error Metrics:")
    logger.info(f"   • RMSE (Root Mean Square Error): {stats_dict['rmse']:.1f}")
    logger.info(f"   • MAE (Mean Absolute Error): {stats_dict['mae']:.1f}")
    logger.info(f"   • MAPE (Mean Absolute Percentage Error): {stats_dict['mape']:.1f}%")
    logger.info(f"   • Bias (Mean Error): {stats_dict['bias']:.1f}")
    
    logger.info(f"\n🔻 Flow Estimation:")
    if stats_dict['underestimation_percent'] > 0:
        logger.info(f"   • Model UNDERESTIMATES flow by {stats_dict['underestimation_percent']:.1f}%")
    else:
        logger.info(f"   • Model OVERESTIMATES flow by {abs(stats_dict['underestimation_percent']):.1f}%")
    
    logger.info(f"\n🚦 GEH Statistic:")
    logger.info(f"   • GEH ≤ 5: {stats_dict['geh']['within_5']:,} points ({stats_dict['geh']['within_5_pct']:.1f}%)")
    logger.info(f"   • GEH ≤ 10: {stats_dict['geh']['within_10']:,} points ({stats_dict['geh']['within_10_pct']:.1f}%)")
    logger.info(f"   • GEH ≤ 15: {stats_dict['geh']['within_15']:,} points ({stats_dict['geh']['within_15_pct']:.1f}%)")
    logger.info(f"   • GEH ≤ 25: {stats_dict['geh']['within_25']:,} points ({stats_dict['geh']['within_25_pct']:.1f}%)")
    logger.info("=" * 60 + "\n")

def create_comprehensive_plot(df, stats_dict, output_path=None):
    """Create a comprehensive scatter plot with statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("CREATING COMPREHENSIVE PLOT")
    logger.info("=" * 60)
    
    # Clean data
    df_clean = df.dropna(subset=['flow', 'simulated_flow'])
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[:2, :2])
    
    # Color by city
    cities = df_clean['city'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(cities)))
    
    for i, city in enumerate(cities):
        city_data = df_clean[df_clean['city'] == city]
        ax_main.scatter(city_data['flow'], city_data['simulated_flow'],
                       alpha=0.6, s=50, label=city.capitalize(),
                       color=colors[i], edgecolors="white")
    
    # Add perfect correlation line (1:1)
    min_val = min(df_clean['flow'].min(), df_clean['simulated_flow'].min())
    max_val = max(df_clean['flow'].max(), df_clean['simulated_flow'].max())
    ax_main.plot([min_val, max_val], [min_val, max_val], 
                'k--', alpha=0.8, linewidth=2, label='Perfect correlation (1:1)')
    
    # Add regression line (no intercept)
    x_range = np.linspace(min_val, max_val, 100)
    y_fit = stats_dict['slope'] * x_range
    ax_main.plot(x_range, y_fit, 'r-', linewidth=2, alpha=0.5,
                label=f'Fitted line (y = {stats_dict["slope"]:.3f}x)')
    
    ax_main.set_xlabel('Observed Flow (vehicles/day)', fontsize=12)
    ax_main.set_ylabel('Simulated Flow (vehicles/day)', fontsize=12)
    ax_main.set_title('Observed vs Simulated Traffic Flows', fontsize=14, fontweight='bold')
    ax_main.legend(bbox_to_anchor=(1.02, 1.1), loc='upper left', ncols = 2)
    ax_main.grid(True, alpha=0.3)
    
    # Statistics text box
    ax_stats = fig.add_subplot(gs[:2, 2])
    ax_stats.axis('off')
    
    city_names = [c.capitalize() for c in cities]
    mid = (len(city_names) + 1) // 2
    first_line = ', '.join(city_names[:mid])
    second_line = ', '.join(city_names[mid:]) if city_names[mid:] else ''
    cities_lines = first_line + (f"\n  {second_line}" if second_line else "")

    stats_text = f"""
FLOW COMPARISON STATISTICS

Data Points: {stats_dict['n_points']:,}

Correlation & Fit:
R² = {stats_dict['r2']:.4f}
Correlation = {stats_dict['correlation']:.4f}
Slope = {stats_dict['slope']:.4f}

Error Metrics:
RMSE = {stats_dict['rmse']:.1f}
MAE = {stats_dict['mae']:.1f}
MAPE = {stats_dict['mape']:.1f}%
Bias = {stats_dict['bias']:.1f}
NSE = {stats_dict['nse']:.4f}

Flow Statistics:
Mean Observed = {stats_dict['mean_observed']:,.0f}
Mean Simulated = {stats_dict['mean_simulated']:,.0f}
Percentage difference = {stats_dict['underestimation_percent']:.1f}%

GEH:
GEH<=5 : {stats_dict['geh']['within_5_pct']:.1f} % | GEH<=10: {stats_dict['geh']['within_10_pct']:.1f} %
GEH<=15: {stats_dict['geh']['within_15_pct']:.1f} % | GEH<=25: {stats_dict['geh']['within_25_pct']:.1f} %

SQV:
SQV>=0.9: {stats_dict['sqv']['0.9']} % | SQV>=0.85: {stats_dict['sqv']['0.85']} %
SQV>=0.8: {stats_dict['sqv']['0.8']} % | SQV>=0.7 : {stats_dict['sqv']['0.7']} %

Cities:
{cities_lines}
            """
    
    ax_stats.text(0.0, 0.92, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Residuals plot
    ax_residuals = fig.add_subplot(gs[2, :2])
    residuals = df_clean['simulated_flow'] - df_clean['flow']
    ax_residuals.scatter(df_clean['flow'], residuals, alpha=0.6, s=30, edgecolors="white")
    ax_residuals.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax_residuals.set_xlabel('Observed Flow (vehicles/day)', fontsize=12)
    ax_residuals.set_ylabel('Residuals (Sim - Obs)', fontsize=12)
    ax_residuals.set_title('Residuals Plot', fontsize=12, fontweight='bold')
    ax_residuals.grid(True, alpha=0.3)
    
    # Distribution comparison
    ax_dist = fig.add_subplot(gs[2, 2])
    ax_dist.hist(df_clean['flow'], bins=30, alpha=0.7, label='Observed', density=True)
    ax_dist.hist(df_clean['simulated_flow'], bins=30, alpha=0.7, label='Simulated', density=True)
    ax_dist.set_xlabel('Flow (vehicles/day)', fontsize=12)
    ax_dist.set_ylabel('Density', fontsize=12)
    ax_dist.set_title('Flow Distributions', fontsize=12, fontweight='bold')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    plt.suptitle('Traffic Flow Analysis: Observed vs Simulated', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    if output_path is not None:
        output_file = os.path.join(output_path, "flow_analysis_comprehensive.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Plot saved as: {output_file}")
        
    # Display plot
    plt.close()

def plot_by_road_cat(df, output_path=None, title=None):
    """
    Plot average observed and simulated flow by highway (road) category.

    - df: DataFrame with at least 'flow', 'simulated_flow' and 'link_id' columns.
          'link_id' can be a scalar/string or an iterable (list/tuple/ndarray)
          in which case the first element is used as the actual link id.
    - output_path: directory to save the figure (if None, figure won't be saved).
    - title: optional plot title (defaults to a sensible title).
    Returns the matplotlib Axes object or None if plotting could not be produced.
    """

    # Create local copy to avoid mutating caller's DataFrame
    df = df.copy()

    if 'highway' not in df.columns or df['highway'].isna().all():
        logger.info("❌ No highway information available after merge. Cannot produce highway-type plot.")
        return None

    # Filter out rows without highway
    df = df[df['highway'].notna()].reset_index(drop=True)

    # Compute counts and averages
    highway_counts = df['highway'].value_counts().to_dict()
    df_avg = df.groupby('highway', as_index=False)[['flow', 'simulated_flow']].mean()
    df_melted = df_avg.melt(id_vars='highway', var_name='Flow Type', value_name='Average Flow')

    # Prepare plot order (preserve df_avg order)
    order = df_avg['highway'].tolist()

    # Create plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_melted,
        x='highway',
        y='Average Flow',
        hue='Flow Type',
        order=order,
        ci=None,
        palette='Set2'
    )

    # Annotate group counts above the bars (use max average per highway for offset)
    max_per_highway = df_melted.groupby('highway')['Average Flow'].max()
    for i, lab in enumerate(order):
        # x position is i; y is the max average + small offset
        y = float(max_per_highway.get(lab, 0.0))
        offset = max(1.0, y * 0.03)  # small absolute or relative offset
        ax.text(i, y + offset, f"n={highway_counts.get(lab, 0)}", ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Labels and layout
    plot_title = title if title is not None else "Average Observed and Simulated Flow by Highway Type"
    ax.set_title(plot_title, fontsize=16, pad=12)
    ax.set_xlabel("Highway Type", fontsize=13)
    ax.set_ylabel("Average Flow (vehicles/day)", fontsize=13)
    ax.tick_params(axis='x', rotation=30, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(title="", fontsize=13)
    plt.ylim([0, df_melted["Average Flow"].max()*1.2])
    plt.tight_layout()
    # Ensure output directory exists and save if requested
    if output_path is not None:        
        output_file = os.path.join(output_path, "flow_by_road_type.png")
        plt.savefig(output_file, dpi=150, bbox_inches="tight")

    plt.close()

def get_average_flow_veh_h_by_category(df, output_path=None):
    """
    For each highway (road) type compute the average flow in vehicles per HOUR per LANE.
    - Observed: df['flow']
    - Simulated: df['simulated_flow']
    Result is both plotted (if output_file provided) and returned as a DataFrame.
    Returned DataFrame columns: ['highway', 'avg_obs_vphpl', 'avg_sim_vphpl', 'n_links']
    """
    # Work on a copy
    merged = df.copy()
    
    # only keep categories with more than 20 unique counts
    merged = merged[merged.groupby('highway')['id'].transform('nunique')>15]        
        
    # Group by highway and compute averages and counts
    df_avg = merged.groupby('highway', as_index=False).agg(
        avg_obs_vphpl = ('obs_vphpl', 'mean'),
        avg_sim_vphpl = ('sim_vphpl', 'mean'),
        n_links = ('id', 'nunique')
    )    
    
    # Sort by observed vphpl descending for nicer plotting
    df_avg = df_avg.sort_values('avg_obs_vphpl', ascending=False).reset_index(drop=True)

    # Plot results
    df_plot = df_avg.melt(id_vars='highway', value_vars=['avg_obs_vphpl', 'avg_sim_vphpl'],
                            var_name='Flow Type', value_name='Avg VPHPL')
    # make nicer labels
    df_plot['Flow Type'] = df_plot['Flow Type'].map({
        'avg_obs_vphpl': 'Observed (vphpl)',
        'avg_sim_vphpl': 'Simulated (vphpl)'
    })

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_plot, x='highway', y='Avg VPHPL', hue='Flow Type',
                        order=df_avg['highway'].tolist(), ci=None, palette='Set2')

    # annotate counts
    max_per_highway = df_plot.groupby('highway')['Avg VPHPL'].max()
    for i, hw in enumerate(df_avg['highway'].tolist()):
        y = float(max_per_highway.get(hw, 0.0))
        offset = max(500, y)  # Increased offset to place text higher above the bars
        ax.text(i, y + offset, f"n={int(df_avg.loc[df_avg['highway'] == hw, 'n_links'].iloc[0])}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overlay individual data points as vertical dots (strip plots)
    df_strip = pd.concat([
        merged[["highway", "obs_vphpl"]].rename(columns={"obs_vphpl": "vphpl"}).assign(type="Observed"),
        merged[["highway", "sim_vphpl"]].rename(columns={"sim_vphpl": "vphpl"}).assign(type="Simulated")
    ])
    # Observed points
    sns.stripplot(data=df_strip, x='highway', y='vphpl', hue='type',
                  dodge=True, palette={'Observed': 'darkblue', 'Simulated': 'darkred'}, 
                  alpha=0.3, jitter=0.25, size=3, ax=ax, 
                  order=df_avg['highway'].tolist())

    ax.set_title("Average Flow per HOUR per LANE by Highway Type", fontsize=14, pad=12)
    ax.set_xlabel("Highway Type", fontsize=12, labelpad=20)
    ax.set_ylabel("Average (vehicles / hour / lane)", fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(title="")    
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_path, "flow_by_road_type_veh_h.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')


    plt.close()

def save_as_target(network, df, path_to_output):
    df = df[['link_id','obs_vphpl']]
    df = df.explode("link_id")
    df = df.rename(columns={"link_id":"linkId",
                            "obs_vphpl":"count"})

    # links that are duplicates of others, we give them the original link id
    replicated_links = network.links[network.links.replicate_of.notna()][["link_id","replicate_of"]]
    df = df.merge(replicated_links.rename(columns={"link_id":"linkId", "replicate_of":"original_linkId"}), on="linkId", how="left")
    df["linkId"] = df["original_linkId"].fillna(df["linkId"])
    df = df.drop(columns=["original_linkId"])
    
    # If multiple link ids, we average the counts
    df = df.groupby("linkId", as_index=False)["count"].mean()

    # save it as csv file
    file_path = os.path.join(path_to_output, "target_flow.csv")
    df.to_csv(file_path, index=False, sep=",")
    logger.info(f"✅ Target flow saved to: {file_path}")






