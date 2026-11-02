"""Compare person-level home/education distances across surveys and population."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


AGE_BOUNDS = [-1, 5, 14, 17, 23, 29, 39, 49, 59, np.inf]
AGE_LABELS = ["0–5", "6–14", "15–17", "18–23", "24–29", "30–39", "40–49", "50–59", "60+"]


def configure(context):
    for stage in ["data.microcensus.persons", "data.microcensus.commute",
                  "data.structural_survey.structural_survey", "data.spatial.municipalities",
                  "synthesis.population.enriched", "synthesis.population.activities",
                  "synthesis.population.spatial.primary.education.locations"]:
        context.stage(stage)
    context.config("analysis_path")
    context.config("specific_day_scenario", "workday")
    # User confirmed that DISTTOSCHOOL_AIR should be interpreted in kilometres.
    # Keep the unit explicit for extracts using a different convention.
    context.config("education_distance_ss_unit", "km")


def day_selector(persons, scenario):
    """Use the same reference-day selection as donor matching."""
    if scenario == "workday":
        return persons["workday"]
    if scenario == "weekend":
        return persons["weekend"]
    if scenario in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        return persons["day"].eq(scenario)
    raise ValueError("Unknown education distance reference-day scenario")


def municipality_ids(points, municipalities):
    """Locate observed coordinates without treating unknown places as distance 0."""
    valid = np.isfinite(points[["x", "y"]].to_numpy(dtype=float)).all(axis=1)
    result = pd.Series(pd.NA, index=points.index, dtype="Int64")
    if not valid.any():
        return result
    frame = gpd.GeoDataFrame(index=points.index[valid],
        geometry=gpd.points_from_xy(points.loc[valid, "x"], points.loc[valid, "y"]), crs="EPSG:2056")
    matches = gpd.sjoin(frame, municipalities[["municipality_id", "geometry"]],
                       how="inner", predicate="intersects")
    # Points exactly on a municipal boundary can match twice. Choose one stable
    # ID so one person is never duplicated in a distance distribution.
    ids = matches.groupby(level=0)["municipality_id"].min()
    result.loc[ids.index] = pd.array(ids, dtype="Int64")
    return result


def build_observations(mz_persons, mz_education, survey, persons, locations, activities,
                       municipalities, scenario="workday", ss_unit="km"):
    """One primary education distance per person, with a common spatial check.

    Native distances use MZ/population coordinates and SS DISTTOSCHOOL_AIR.
    Municipality-centroid distances coarsen all three sources to the same scale.
    Survey weights are retained; synthetic people have equal weight. SS is a
    usual education commute, whereas MZ/population describe the selected day.
    """
    if ss_unit not in ["km", "m"]:
        raise ValueError("Structural-survey education distance unit must be km or m")
    municipalities = municipalities.to_crs("EPSG:2056")
    centroids = municipalities.geometry.centroid
    centre_x = pd.Series(centroids.x.to_numpy(), index=municipalities.municipality_id.to_numpy())
    centre_y = pd.Series(centroids.y.to_numpy(), index=municipalities.municipality_id.to_numpy())
    frames, coverage = [], []

    mz = mz_persons.loc[day_selector(mz_persons, scenario)].merge(mz_education, on="person_id", validate="one_to_one")
    mz["education_x"], mz["education_y"] = mz["commute_x"], mz["commute_y"]
    mz["weight"] = mz["person_weight"]
    mz["is_student"] = mz["employment_status"].isin([2, 3])
    mz["distance_native_km"] = mz["commute_home_distance"] / 1000.
    mz["registered_home_x"], mz["registered_home_y"] = mz["home_x"], mz["home_y"]
    if {"commute_home_x", "commute_home_y"}.issubset(mz.columns):
        # Use the same observed diary home as the commute-distance stage for
        # domestic-endpoint filtering and municipality-centroid comparisons.
        mz["home_x"], mz["home_y"] = mz["commute_home_x"], mz["commute_home_y"]
    mz["reference_year"] = mz["date"].dt.year.astype(str)

    active_ids = activities.loc[activities["purpose"].eq("education"), "person_id"].unique()
    population = persons[persons["person_id"].isin(active_ids)].merge(locations, on="person_id", validate="one_to_one")
    if "x" in population and "y" in population:
        population["education_x"], population["education_y"] = population["x"], population["y"]
    else:
        # A pandas merge can retain geometry values while losing GeoDataFrame
        # methods. Restore the GeoSeries explicitly before reading coordinates.
        geometry = gpd.GeoSeries(population["geometry"], crs=locations.crs).to_crs("EPSG:2056")
        population["education_x"], population["education_y"] = geometry.x, geometry.y
    population["weight"] = 1.
    population["distance_native_km"] = np.hypot(population["home_x"] - population["education_x"],
                                               population["home_y"] - population["education_y"]) / 1000.
    population["reference_year"] = "synthetic population"

    for source, frame in [("MZ", mz), ("Population", population)]:
        frame = frame.reset_index(drop=True)
        home = municipality_ids(frame[["home_x", "home_y"]].rename(columns={"home_x":"x", "home_y":"y"}), municipalities)
        education = municipality_ids(frame[["education_x", "education_y"]].rename(
            columns={"education_x":"x", "education_y":"y"}), municipalities)
        frame["home_municipality_id"], frame["education_municipality_id"] = home, education
        if "registered_home_x" in frame:
            frame["registered_home_municipality_id"] = municipality_ids(
                frame[["registered_home_x", "registered_home_y"]].rename(
                    columns={"registered_home_x": "x", "registered_home_y": "y"}), municipalities)
        # Restrict both endpoints to Switzerland for a comparable domestic view.
        frame["eligible"] = home.notna() & education.notna()
        frame["source"] = source
        coverage.append(dict(source=source, reference_records=len(frame), domestic_endpoints=int(frame.eligible.sum())))
        frames.append(frame)

    ss = survey[survey["is_student"].eq(1)].copy().reset_index(drop=True)
    country = pd.to_numeric(ss["education_country_code"], errors="coerce")
    foreign = country.gt(0) & country.ne(8100)
    # DISTTOSCHOOL_AIR starts at DEPFORSCHOOL*, which can differ from RES_*.
    # Use the same origin for the centroid comparison and retain residence
    # distance separately so the change in definition remains inspectable.
    ss["registered_home_municipality_id"] = ss["home_municipality_id"]
    ss["home_municipality_id"] = ss["education_departure_municipality_id"]
    ss["eligible"] = (~foreign & ss["education_departure_status"].eq("domestic")
                      & ss["home_municipality_id"].isin(centre_x.index)
                      & ss["education_municipality_id"].isin(centre_x.index))
    ss["distance_native_km"] = pd.to_numeric(ss["crowfly_distance_to_school"], errors="coerce")
    if ss_unit == "m":
        ss["distance_native_km"] /= 1000.
    # Negative values are BFS missing/not-applicable codes, not short commutes.
    ss.loc[ss["distance_native_km"].lt(0), "distance_native_km"] = np.nan
    ss["source"], ss["reference_year"] = "Structural survey", ss["survey_year"].astype(str)
    coverage.append(dict(source="Structural survey", reference_records=len(ss), domestic_endpoints=int(ss.eligible.sum())))
    frames.append(ss)

    observations = []
    for frame in frames:
        frame = frame.loc[frame["eligible"]].copy()
        # Uniform flags also make the exported observation CSV round-trip safely.
        frame["is_student"] = frame["is_student"].astype("Int64")
        frame["distance_municipality_km"] = np.hypot(
            frame["home_municipality_id"].map(centre_x) - frame["education_municipality_id"].map(centre_x),
            frame["home_municipality_id"].map(centre_y) - frame["education_municipality_id"].map(centre_y)) / 1000.
        residence = frame.get("registered_home_municipality_id", frame["home_municipality_id"])
        frame["distance_registered_residence_municipality_km"] = np.hypot(
            residence.map(centre_x) - frame["education_municipality_id"].map(centre_x),
            residence.map(centre_y) - frame["education_municipality_id"].map(centre_y)) / 1000.
        frame["age_group"] = pd.cut(frame["age"], AGE_BOUNDS, labels=AGE_LABELS)
        observations.append(frame[["source", "reference_year", "age", "age_group", "is_student", "weight",
                                   "distance_native_km", "distance_municipality_km",
                                   "distance_registered_residence_municipality_km"]])
    return pd.concat(observations, ignore_index=True), pd.DataFrame(coverage)


def weighted_distribution(distance, weight):
    """Return the exact weighted ECDF, collapsing equal distances."""
    valid = np.isfinite(distance) & (distance >= 0) & np.isfinite(weight) & (weight > 0)
    frame = pd.DataFrame({"distance": distance[valid], "weight": weight[valid]})
    if frame.empty:
        return np.array([]), np.array([])
    totals = frame.groupby("distance", sort=True)["weight"].sum()
    return totals.index.to_numpy(), totals.cumsum().to_numpy() / totals.sum()


def write_analysis(observations, coverage, output_path, scenario="workday", ss_unit="km"):
    """Export weighted age distributions and plots, keeping missing coverage visible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(path / "coverage.csv", index=False)
    observations.to_csv(path / "distance_observations.csv", index=False)
    summary, cdfs = [], []
    sources = ["MZ", "Structural survey", "Population"]
    colours = {"MZ": "#2279b9", "Structural survey": "#d86a12", "Population": "#278c32"}
    for scope in ["all_education", "students_only"]:
        selected = observations if scope == "all_education" else observations[observations["is_student"].eq(1)]
        for metric in ["native", "municipality"]:
            column = "distance_%s_km" % metric
            plot_max = max(250., float(selected[column].max()) * 1.05)
            fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
            for age, ax in zip(AGE_LABELS, axes.flat):
                for source in sources:
                    group = selected[selected["source"].eq(source) & selected["age_group"].eq(age)]
                    distances = group[column].to_numpy(dtype=float)
                    weights = group["weight"].to_numpy(dtype=float)
                    valid = np.isfinite(distances) & (distances >= 0) & np.isfinite(weights) & (weights > 0)
                    x, cdf = weighted_distribution(distances, weights)
                    row = dict(scope=scope, metric=metric, age_group=age, source=source,
                               records=len(group), valid_records=int(valid.sum()),
                               missing_records=int((~valid).sum()), weight_sum=float(weights[valid].sum()),
                               mean_km=np.average(distances[valid], weights=weights[valid]) if valid.any() else np.nan)
                    for q in [.05, .25, .5, .75, .95]:
                        row["p%d_km" % round(q*100)] = x[np.searchsorted(cdf, q)] if len(x) else np.nan
                    summary.append(row)
                    if not len(x):
                        continue
                    # Thin only the drawing/exported CDF; summaries use all observations.
                    indices = np.unique(np.linspace(0, len(x)-1, min(1000, len(x))).astype(int))
                    # Missing age coverage must not change another source's colour.
                    ax.step(x[indices], cdf[indices], where="post", label=source, color=colours[source])
                    cdfs.extend(dict(scope=scope, metric=metric, age_group=age, source=source,
                                     distance_km=float(x[i]), cumulative_share=float(cdf[i])) for i in indices)
                ax.set_title("Age %s" % age)
                if not ax.lines:
                    ax.text(.5, .5, "No eligible education observations", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="grey")
                ax.set_xscale("symlog", linthresh=1)
                ax.set_xlim(0, plot_max)
                ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
            handles, labels = [], []
            for ax in axes.flat:
                h, l = ax.get_legend_handles_labels()
                for item, label in zip(h, l):
                    if label not in labels:
                        handles.append(item); labels.append(label)
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3)
            fig.suptitle("Home–education distance · %s · %s · %s" % (scenario, scope, metric))
            fig.supxlabel("Straight-line distance (km; logarithmic scale above 1 km)", y=.055)
            fig.supylabel("Weighted cumulative share")
            fig.tight_layout(rect=(.02, .10, 1, .96))
            fig.savefig(path / ("distance_cdf_%s_%s.png" % (scope, metric)), dpi=160)
            fig.savefig(path / ("distance_cdf_%s_%s.pdf" % (scope, metric)))
            plt.close(fig)
    pd.DataFrame(summary).to_csv(path / "distance_summary.csv", index=False)
    pd.DataFrame(cdfs).to_csv(path / "distance_cdf.csv", index=False)
    (path / "README.md").write_text(
        "# Home–education distance comparison\n\n"
        "One primary education destination per person; repeated education activities do not duplicate people. "
        "MZ uses its longest primary education activity and person weights. Population uses the assigned primary location "
        "of people with an actual education activity and equal weights. Both endpoints must be in Switzerland. "
        "SS uses domestic students with valid municipal locations and survey weights pooled across survey years.\n\n"
        "Native: MZ and population coordinate distances in EPSG:2056; SS DISTTOSCHOOL_AIR, interpreted as %s. "
        "That SS unit is a user-confirmed modelling assumption. Negative missing codes are excluded. "
        "Municipality: all sources use distances between the same municipality centroids. Within-municipality "
        "distances are therefore zero; this coarsened comparison checks geographic-resolution effects. "
        "SS municipality distances use DEPFORSCHOOLMUN, the actual school-departure origin. "
        "Registered RES_MUN distance is also retained in the observation CSV for comparison. "
        "MZ uses the preceding observed diary home, or the following diary home if none precedes education, "
        "with household coordinates as a fallback. Population still uses its assigned home.\n\n"
        "MZ/population reference-day filter: %s. SS describes a usual education commute, not attendance on a specific "
        "weekday; its frequency is not used to reweight observations. MZ 2015 and SS 2021–2023 also describe different "
        "years. These differences matter when interpreting the curves. SS has no observations under age 15; "
        "MZ has no observations under age 6. Empty groups remain missing, not zero-distance distributions.\n\n"
        "Both all-education and student-only views are exported. All-education includes nonstudent MZ/population "
        "education activities, while SS is limited to recorded students. PDF/PNG figures are standalone artifacts; "
        "summary quantiles and means use all valid observations and source weights.\n" % (ss_unit, scenario))
    return {"observations": len(observations), "output_path": str(path)}


def execute(context):
    observations, coverage = build_observations(
        context.stage("data.microcensus.persons"), context.stage("data.microcensus.commute")["education"],
        context.stage("data.structural_survey.structural_survey"), context.stage("synthesis.population.enriched"),
        context.stage("synthesis.population.spatial.primary.education.locations"),
        context.stage("synthesis.population.activities"), context.stage("data.spatial.municipalities")[0],
        context.config("specific_day_scenario"), context.config("education_distance_ss_unit"))
    return write_analysis(observations, coverage, Path(context.config("analysis_path")) / "education_distances",
                          context.config("specific_day_scenario"), context.config("education_distance_ss_unit"))
