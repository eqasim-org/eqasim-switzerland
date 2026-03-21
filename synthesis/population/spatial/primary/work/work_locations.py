import numpy as np
import pandas as pd
import data.spatial.utils as spatial_utils
import logging

logger = logging.getLogger("synpp")

def configure(context):

    context.stage("data.structural_survey.structural_survey")
    context.stage("synthesis.population.spatial.primary.work.work_remotly")
    context.stage("synthesis.population.spatial.primary.work.fixed_work_locations")
    context.stage("synthesis.population.spatial.primary.work.moving_work_locations")
    context.stage("data.microcensus.commute")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.21.persons")

    context.config("random_seed")
    context.config("input_downsampling")    

def execute(context):
    fixed_work_locations = context.stage("synthesis.population.spatial.primary.work.fixed_work_locations")
    moving_work_locations = context.stage("synthesis.population.spatial.primary.work.moving_work_locations")
    remote_agents = context.stage("synthesis.population.spatial.primary.work.work_remotly")

    # Concatenate all work locations
    fixed_work_locations["work_location_type"] = "fixed"
    moving_work_locations["work_location_type"] = "moving"
    remote_agents["work_location_type"] = "remote"
    cols = ["person_id", "destination_id", "commute_distance", "x", "y"] + ["work_location_type"]
    out = pd.concat([fixed_work_locations[cols], moving_work_locations[cols], remote_agents[cols]], ignore_index=True)

    # Ensure no missing coordinates
    assert np.isfinite(out["x"]).all() and np.isfinite(out["y"]).all()

    out = spatial_utils.to_gpd(context, out, coord_type="work")
    
    # try:
    #     plot_analysis(context, out, df_statent, pdf_matrices, zone_ids, persons_per_agent)
    # except Exception as e:
    #     logger.warning(f"Plotting analysis failed: {e}")
    
    return out[["person_id", "destination_id", "work_location_type", "commute_distance", "geometry"]] 





def plot_analysis(context, work_locations, df_statent, pdf_matrices, zone_ids, persons_per_agent):    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    work_locations = work_locations.copy()
    
    # path to figures
    path = context.path()

    # load structural survey data to compare commute distances
    df_ss = context.stage("data.structural_survey.structural_survey")[["home_zone_id", "work_zone_id", "start_work","crowfly_distance_to_work","weight",
                                                                       "mode", "home_zone_level", "work_zone_level"]]
    # working home
    df_ss["crowfly_distance"] = df_ss["crowfly_distance_to_work"] * 1e3
    df_ss.loc[df_ss["start_work"]==1,"crowfly_distance"] = 0
    df_ss_home = df_ss.loc[df_ss["start_work"]==1].copy()
    
    # filter df_ss as in matrix
    df_ss = df_ss[~np.isnan(df_ss["home_zone_id"])]
    df_ss = df_ss[~np.isnan(df_ss["work_zone_id"])]
    df_ss = df_ss[(df_ss["crowfly_distance_to_work"] > 0.0) | (df_ss["start_work"] > 1)]
    df_ss = df_ss[~(df_ss["work_zone_level"] == "country")]
    df_ss = df_ss[~(df_ss["home_zone_level"] == "country")]
    # df_ss = df_ss[~((df_ss["mode"] == "unknown") | (df_ss["mode"] == "other"))]

    # bring back working home for comparison
    df_ss = pd.concat([df_ss, df_ss_home], ignore_index=True)

    # load Mz commute
    mz = context.stage("data.microcensus.21.persons")
    mz = mz[mz["employed"]]

    # Plot commute distance distribution
    fig, ax = plt.subplots(figsize=(12,4))

    # Define colors
    color1 = 'blue' 
    color2 = 'red'
    color3 = 'c'
    bins = np.linspace(0,80000,41)

    # Plot filled histograms with low alpha
    mz.work_commute_distance.plot.hist(ax=ax, bins=bins, alpha=0.3, density=True, color=color1, label='Commute Distance (Microcensus21)', weights=mz["person_weight"])
    df_ss.crowfly_distance.plot.hist(ax=ax, bins=bins, alpha=0.2, density=True, color=color2, label='Commute Distance (SSurvey)', weights=df_ss["weight"])
    work_locations.commute_distance.plot.hist(ax=ax, bins=bins, alpha=0.2, density=True, color=color3, label='Commute Distance (Assigned)')

    # Plot step histograms with higher alpha and same colors
    mz.work_commute_distance.plot.hist(ax=ax, bins=bins, alpha=0.8, histtype='step', linewidth=1, color=color1, density=True, label="", weights=mz["person_weight"])
    df_ss.crowfly_distance.plot.hist(ax=ax, bins=bins, alpha=0.8, histtype='step', linewidth=1, color=color2, density=True, label="", weights=df_ss["weight"])
    work_locations.commute_distance.plot.hist(ax=ax, bins=bins, alpha=0.8, histtype='step', linewidth=1, color=color3, density=True, label="")

    # Mz commute stats
    sel = mz.work_commute_distance > -1e3
    commute_mean = np.average(mz.work_commute_distance[sel], weights=mz["person_weight"][sel])
    commute_median = weighted_median(mz.work_commute_distance[sel], mz["person_weight"][sel])
    ax.axvline(commute_mean, color=color1, linestyle='-', alpha=0.8, linewidth=1., label=f'Microcensus Mean: {commute_mean:.1f}')
    ax.axvline(commute_median, color=color1, linestyle=':', alpha=0.8, linewidth=1., label=f'Microcensus Median: {commute_median:.1f}')

    # Survey commute stats
    sel = df_ss.crowfly_distance >-1e3
    commute_mean = np.average(df_ss.crowfly_distance[sel], weights=df_ss["weight"][sel])
    commute_median = weighted_median(df_ss.crowfly_distance[sel], df_ss["weight"][sel])
    ax.axvline(commute_mean, color=color2, linestyle='-', alpha=0.8, linewidth=1., label=f'SSurvey Mean: {commute_mean:.1f}')
    ax.axvline(commute_median, color=color2, linestyle=':', alpha=0.8, linewidth=1., label=f'SSurvey Median: {commute_median:.1f}')

    # Assigned commute stats
    sel = work_locations.commute_distance >-1e3
    commute_mean = work_locations.commute_distance[sel].mean()
    commute_median = work_locations.commute_distance[sel].median()
    ax.axvline(commute_mean, color=color3, linestyle='-', alpha=0.8, linewidth=1., label=f'Assigned Mean: {commute_mean:.1f}')
    ax.axvline(commute_median, color=color3, linestyle=':', alpha=0.8, linewidth=1., label=f'Assigned Median: {commute_median:.1f}')

    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Density')
    plt.xlabel('Distance')    
    plt.savefig(f"{path}/commute_distance_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()

    # from this point we only keep agents not working from home
    work_locations = work_locations[~work_locations.work_remotly].reset_index(drop=True)
    # companies number of employees
    out = work_locations.merge(df_statent[["enterprise_id", "zone_id"]], left_on="destination_id", right_on="enterprise_id")
    employees_by_zone_statent = df_statent.groupby("zone_id").number_employees.sum().reset_index()
    employees_by_zone_assigned = out.groupby("zone_id").size().mul(persons_per_agent).reset_index(name="number_employees_assigned")
    employees = employees_by_zone_statent.merge(employees_by_zone_assigned, on="zone_id", how="left")
    employees["number_employees_assigned"] = employees["number_employees_assigned"].fillna(0.0)
    
    fig, ax = plt.subplots(figsize=(7,7))
    plt.scatter(employees.number_employees, employees.number_employees_assigned, label = "Employees per zone")
    
    max_employment = max(employees.number_employees.max(), employees.number_employees_assigned.max())    
    plt.plot([0, max_employment], [0, max_employment], "k--", alpha=0.5, label = f"Line: y = x")
    # regression    
    model = LinearRegression(fit_intercept=False)
    model.fit(employees.number_employees.values.reshape(-1, 1), employees.number_employees_assigned.values)
    slope = model.coef_[0]
    x_range = np.linspace(0, employees.number_employees.max(), 100)
    plt.plot(x_range, slope * x_range, "r-", alpha=0.7, label=f"Fitted: y = {slope:.3f}x")
    # plot labels
    plt.grid(linestyle='--',alpha=0.3)
    plt.legend()
    plt.xlabel("Sum of employees (from statent)")
    plt.ylabel("Assigned represented employees per zone")
    plt.savefig(f"{path}/number_of_employees_per_zone.png", dpi=200, bbox_inches='tight')
    plt.close()


    # plot ODs
    matrix = pdf_matrices
    matrix = pd.DataFrame(matrix, columns = zone_ids, index = zone_ids)
    matrix = matrix.reset_index().melt(id_vars='index', var_name='destination_zone_id', value_name='flow').rename(columns={"index":"origin_zone_id"})
    matrix = matrix[(matrix.flow.notna())&(matrix.flow>0)].reset_index(drop=True)

    out = out.merge(context.stage("synthesis.population.sampled")[["person_id","home_zone_id","car_availability"]], on="person_id", how="left")
    origin_counts = out.groupby("home_zone_id").size().reset_index(name="count")

    flows = out.groupby(["home_zone_id", "zone_id"]).size().reset_index(name="assigned_count")
    flows = flows.merge(origin_counts, on="home_zone_id", how="left")
    flows["flow_assigned"] = flows["assigned_count"] / flows["count"]
    flows = flows.rename(columns={"home_zone_id": "origin_zone_id", "zone_id": "destination_zone_id"})

    flows = matrix.merge(
        flows[["origin_zone_id", "destination_zone_id", "flow_assigned", "count"]],
        on=["origin_zone_id", "destination_zone_id"],
        how="left"
    ).rename(columns={"flow": "flow_od"})
    flows["flow_assigned"] = flows["flow_assigned"].fillna(0.0)
    flows["count"] = flows["count"].fillna(0)

    threshold = 50
    mask = flows["count"]>threshold

    fig, ax = plt.subplots(figsize=(7,7))
    plt.scatter(flows.flow_od[mask], flows.flow_assigned[mask], label = "OD Probabilities")
    plt.plot([0,1],[0,1], "k--", alpha=0.5, label = "Line : y = x")
    # plot labels
    plt.grid(linestyle='--',alpha=0.3)
    plt.legend()
    plt.xlabel("OD Probabilities (survey)")
    plt.ylabel("OD Probabilities (Assigned work location)")
    _=plt.title(f"Only ODs with more than {threshold} trips are considered")
    plt.savefig(f"{path}/od_probabilities.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Plot ovgk distribution for assigned work locations (distinction by car availability)
    out = out.merge(df_statent[["enterprise_id", "ovgk"]], left_on="destination_id", right_on="enterprise_id", how="left")
    out["ovgk"] = out["ovgk"].fillna("Other")
    fig, ax = plt.subplots(figsize=(5,5))
    colors = ["navy","darkorange"]
    
    ovgk_data = []
    for car_status, group in out.groupby("car_availability"):
        ovgk_data.append(group["ovgk"].value_counts(normalize=True).sort_index())
    
    pd.DataFrame(ovgk_data, index=["Without Car", "With Car"]).T.plot(kind='bar', ax=ax, color=colors, width=0.8)
    
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend()
    plt.xlabel("OVGK Category")
    plt.ylabel("Proportion of Assigned Work Locations")
    plt.title("Distribution of OVGK Categories for Assigned Work Locations by Car Availability")
    plt.savefig(f"{path}/ovgk_distribution_by_car_availability.png", dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"\n Work Assignement: \t Plots saved to {path} \n")

def weighted_median(values, weights):
    values = np.array(values)
    weights = np.array(weights)
    sorted_idx = np.argsort(values)
    values_sorted = values[sorted_idx]
    weights_sorted = weights[sorted_idx]
    cum_weights = np.cumsum(weights_sorted)
    cutoff = weights_sorted.sum() / 2.0
    return values_sorted[cum_weights >= cutoff][0]