import numpy as np
import pandas as pd
import data.spatial.utils as spatial_utils
import logging
from data.structural_survey.structural_survey import get_filtered_data
from data.od.matrix import (DEFAULT_SEGMENT_KEY, AGE_BIN_EDGES)

logger = logging.getLogger("synpp")


def get_segment_key(sex_value, age_value):
    try:
        sex = int(sex_value)
    except (TypeError, ValueError):
        return DEFAULT_SEGMENT_KEY

    if sex not in (0, 1):
        return DEFAULT_SEGMENT_KEY

    if not np.isfinite(age_value):
        return DEFAULT_SEGMENT_KEY

    age_bin = int(np.digitize(float(age_value), AGE_BIN_EDGES, right=False))
    if age_bin not in (0, 1, 2, 3):
        return DEFAULT_SEGMENT_KEY

    return (sex, age_bin)


def age_bin_label(age_bin):
    labels = {
        0: "<30",
        1: "30-45",
        2: "45-65",
        3: "65+",
    }
    return labels.get(age_bin, str(age_bin))

def configure(context):

    context.stage("data.structural_survey.structural_survey")
    context.stage("synthesis.population.spatial.primary.work.work_remotly")
    context.stage("synthesis.population.spatial.primary.work.fixed_work_locations")
    context.stage("synthesis.population.spatial.primary.work.moving_work_locations")
    context.stage("data.spatial.zones")
    context.stage("data.statent.statent")
    context.stage("synthesis.population.sampled")
    context.stage("data.od.matrix")
    context.stage("data.od.matrix_moving")
    
    if context.config("include_cross_border"):
        context.stage("data.cross_border.destinations")

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

    try:
        plot_analysis(context, fixed_work_locations, moving_work_locations)
    except Exception as e:
        logger.warning(f"Work location plotting analysis failed: {e}")

    out = spatial_utils.to_gpd(context, out, coord_type="work")

    return out[["person_id", "destination_id", "work_location_type", "commute_distance", "geometry"]] 





def plot_analysis(context, fixed_work_locations, moving_work_locations):
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    path = context.path()
    persons_per_agent = 1 / context.config("input_downsampling")

    zone_ids = context.stage("data.spatial.zones")["zone_id"].values
    pdf_fixed, _ = context.stage("data.od.matrix")
    pdf_moving, _ = context.stage("data.od.matrix_moving")

    def get_reference_matrix(pdf_by_segment):
        if isinstance(pdf_by_segment, dict):
            if DEFAULT_SEGMENT_KEY in pdf_by_segment:
                return pdf_by_segment[DEFAULT_SEGMENT_KEY]
            return next(iter(pdf_by_segment.values()))
        return pdf_by_segment

    pdf_fixed_reference = get_reference_matrix(pdf_fixed)
    pdf_moving_reference = get_reference_matrix(pdf_moving)

    persons = context.stage("synthesis.population.sampled")[["person_id", "home_zone_id", "sex", "age"]].copy()
    persons["age"] = pd.to_numeric(persons["age"], errors="coerce")
    df_statent = context.stage("data.statent.statent")[["enterprise_id", "zone_id", "number_employees"]].copy()
    df_statent = df_statent.dropna(subset=["enterprise_id", "zone_id", "number_employees"])

    def matrix_to_long(pdf_matrix):
        matrix = pd.DataFrame(pdf_matrix, columns=zone_ids, index=zone_ids)
        matrix = matrix.reset_index().melt(
            id_vars="index", var_name="destination_zone_id", value_name="flow_od"
        ).rename(columns={"index": "origin_zone_id"})
        return matrix[(matrix["flow_od"].notna()) & (matrix["flow_od"] > 0)].reset_index(drop=True)

    def prepare_assigned(work_df):
        assigned = work_df[["person_id", "destination_id", "commute_distance"]].copy()
        assigned = assigned.merge(persons, on="person_id", how="left")
        assigned = assigned.merge(
            df_statent[["enterprise_id", "zone_id"]],
            left_on="destination_id",
            right_on="enterprise_id",
            how="left"
        )
        assigned = assigned.rename(columns={"zone_id": "destination_zone_id", "home_zone_id": "origin_zone_id"})
        return assigned.dropna(subset=["origin_zone_id", "destination_zone_id"]).reset_index(drop=True)

    def plot_distance_comparison(kind_label, survey_df, assigned_df, filename):
        fig, ax = plt.subplots(figsize=(11, 4))
        bins = np.linspace(0, 80000, 41)

        survey_dist = survey_df["crowfly_distance_to_work"].to_numpy(dtype=float) * 1e3
        survey_w = survey_df["weight"].to_numpy(dtype=float)
        assigned_dist = assigned_df["commute_distance"].to_numpy(dtype=float)

        ax.hist(survey_dist, bins=bins, alpha=0.25, density=True, color="tab:red", weights=survey_w,
                label=f"Survey ({kind_label})")
        ax.hist(assigned_dist, bins=bins, alpha=0.25, density=True, color="tab:blue",
                label=f"Assigned ({kind_label})")

        ax.hist(survey_dist, bins=bins, alpha=0.9, density=True, histtype="step", linewidth=1.2,
                color="tab:red", weights=survey_w, label="")
        ax.hist(assigned_dist, bins=bins, alpha=0.9, density=True, histtype="step", linewidth=1.2,
                color="tab:blue", label="")

        survey_mean = np.average(survey_dist, weights=survey_w)
        survey_median = weighted_median(survey_dist, survey_w)
        assigned_mean = np.mean(assigned_dist)
        assigned_median = np.median(assigned_dist)

        ax.axvline(survey_mean, color="tab:red", linestyle="-", alpha=0.8, linewidth=1.0,
                   label=f"Survey mean: {survey_mean:.0f}")
        ax.axvline(survey_median, color="tab:red", linestyle=":", alpha=0.8, linewidth=1.0,
                   label=f"Survey median: {survey_median:.0f}")
        ax.axvline(assigned_mean, color="tab:blue", linestyle="-", alpha=0.8, linewidth=1.0,
                   label=f"Assigned mean: {assigned_mean:.0f}")
        ax.axvline(assigned_median, color="tab:blue", linestyle=":", alpha=0.8, linewidth=1.0,
                   label=f"Assigned median: {assigned_median:.0f}")

        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylabel("Density")
        ax.set_xlabel("Distance [m]")
        ax.set_title(f"Commute distance comparison ({kind_label})")
        plt.savefig(f"{path}/{filename}", dpi=200, bbox_inches="tight")
        plt.close()

    def plot_od_comparison(kind_label, matrix_long, assigned_df, filename):
        origin_counts = assigned_df.groupby("origin_zone_id").size().reset_index(name="count")
        flows = assigned_df.groupby(["origin_zone_id", "destination_zone_id"]).size().reset_index(name="assigned_count")
        flows = flows.merge(origin_counts, on="origin_zone_id", how="left")
        flows["flow_assigned"] = flows["assigned_count"] / flows["count"]

        flows = matrix_long.merge(
            flows[["origin_zone_id", "destination_zone_id", "flow_assigned", "count"]],
            on=["origin_zone_id", "destination_zone_id"],
            how="left"
        )
        flows["flow_assigned"] = flows["flow_assigned"].fillna(0.0)
        flows["count"] = flows["count"].fillna(0)

        threshold = 50
        mask = flows["count"] > threshold

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(flows.loc[mask, "flow_od"], flows.loc[mask, "flow_assigned"], alpha=0.7,
                   label=f"ODs with origin count > {threshold}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Line: y = x")
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend()
        ax.set_xlabel("OD probabilities (reference matrix)")
        ax.set_ylabel("OD probabilities (assigned)")
        ax.set_title(f"OD comparison ({kind_label})")
        plt.savefig(f"{path}/{filename}", dpi=200, bbox_inches="tight")
        plt.close()

    survey_fixed = get_filtered_data(context, "fixed")[["crowfly_distance_to_work", "weight"]].copy()
    survey_moving = get_filtered_data(context, "moving")[["crowfly_distance_to_work", "weight"]].copy()
    survey_fixed = survey_fixed.dropna(subset=["crowfly_distance_to_work", "weight"])
    survey_moving = survey_moving.dropna(subset=["crowfly_distance_to_work", "weight"])

    assigned_fixed = prepare_assigned(fixed_work_locations)
    assigned_moving = prepare_assigned(moving_work_locations)

    if len(assigned_fixed) > 0:
        assigned_fixed["segment_key"] = [
            get_segment_key(sex, age)
            for sex, age in zip(assigned_fixed["sex"].values, assigned_fixed["age"].values)
        ]

    if len(survey_fixed) > 0 and len(assigned_fixed) > 0:
        plot_distance_comparison("fixed", survey_fixed, assigned_fixed, "commute_distance_distribution_fixed.png")
    if len(survey_moving) > 0 and len(assigned_moving) > 0:
        plot_distance_comparison("moving", survey_moving, assigned_moving, "commute_distance_distribution_moving.png")

    if len(assigned_fixed) > 0:
        plot_od_comparison("fixed", matrix_to_long(pdf_fixed_reference), assigned_fixed, "od_probabilities_fixed.png")

        # Additional OD comparisons for each fixed-work sex/age segment.
        if isinstance(pdf_fixed, dict):
            segment_keys = [
                key for key in pdf_fixed.keys()
                if key != DEFAULT_SEGMENT_KEY and isinstance(key, tuple) and len(key) == 2
            ]

            for key in sorted(segment_keys):
                seg_assigned = assigned_fixed[assigned_fixed["segment_key"] == key]
                if len(seg_assigned) == 0:
                    continue

                seg_matrix = pdf_fixed.get(key, pdf_fixed_reference)
                sex, age_bin = key
                segment_label = f"fixed sex={sex} age={age_bin_label(age_bin)}"
                segment_filename = f"od_probabilities_fixed_sex{sex}_agebin{age_bin}.png"
                plot_od_comparison(
                    segment_label,
                    matrix_to_long(seg_matrix),
                    seg_assigned,
                    segment_filename,
                )

    if len(assigned_moving) > 0:
        plot_od_comparison("moving", matrix_to_long(pdf_moving_reference), assigned_moving, "od_probabilities_moving.png")

    assigned_all = pd.concat([assigned_fixed, assigned_moving], ignore_index=True)
    if context.config("include_cross_border"):
        df_cross_border = context.stage("data.cross_border.destinations")[[
            "cross_border_person_id", "trip_purpose", "destination_id"
        ]]        
        df_cross_border = df_cross_border[df_cross_border["trip_purpose"] == "work"]
        df_cross_border = df_cross_border.drop_duplicates(subset=["cross_border_person_id"]) 

        assigned_cross_border = df_cross_border.merge(
            df_statent[["enterprise_id", "zone_id"]],
            left_on="destination_id",
            right_on="enterprise_id",
            how="left"
        ).rename(columns={"zone_id": "destination_zone_id"})
        assigned_cross_border = assigned_cross_border.dropna(subset=["destination_zone_id"])
        assigned_cross_border = assigned_cross_border[["destination_zone_id"]].copy()

        assigned_all = pd.concat([
            assigned_all[["destination_zone_id"]],
            assigned_cross_border
        ], ignore_index=True)

    assigned_by_zone = assigned_all.groupby("destination_zone_id").size().mul(persons_per_agent).reset_index(
        name="number_employees_assigned"
    )

    employees_by_zone_statent = df_statent.groupby("zone_id")["number_employees"].sum().reset_index()
    employees = employees_by_zone_statent.merge(
        assigned_by_zone,
        left_on="zone_id",
        right_on="destination_zone_id",
        how="left"
    )
    employees["number_employees_assigned"] = employees["number_employees_assigned"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(employees["number_employees"], employees["number_employees_assigned"],
               alpha=0.7, label="Assigned fixed + moving employees per zone")
    max_employment = max(employees["number_employees"].max(), employees["number_employees_assigned"].max())
    ax.plot([0, max_employment], [0, max_employment], "k--", alpha=0.5, label="Line: y = x")

    model = LinearRegression(fit_intercept=False)
    model.fit(employees["number_employees"].values.reshape(-1, 1), employees["number_employees_assigned"].values)
    slope = model.coef_[0]
    x_range = np.linspace(0, employees["number_employees"].max(), 100)
    ax.plot(x_range, slope * x_range, "r-", alpha=0.7, label=f"Fitted: y = {slope:.3f}x")

    ax.grid(linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_xlabel("Sum of employees (STATENT)")
    ax.set_ylabel("Assigned represented employees per zone")
    ax.set_title("Employees per zone (fixed + moving assignments)")
    plt.savefig(f"{path}/number_of_employees_per_zone.png", dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"\n Work assignment analysis: plots saved to {path} \n")

def weighted_median(values, weights):
    values = np.array(values)
    weights = np.array(weights)
    sorted_idx = np.argsort(values)
    values_sorted = values[sorted_idx]
    weights_sorted = weights[sorted_idx]
    cum_weights = np.cumsum(weights_sorted)
    cutoff = weights_sorted.sum() / 2.0
    return values_sorted[cum_weights >= cutoff][0]