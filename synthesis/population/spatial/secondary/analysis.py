import logging
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.spatial.ovgk import impute_parallel as impute_ovgk_parallel
from data.statent.density import impute_parallel as impute_statent_density

logger = logging.getLogger("synpp")

PRIMARY = {"home", "work", "education"}
OVGK_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1, "None": 0}
MAX_DISTANCE_M = 50000.0


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.spatial.ovgk")
    context.stage("data.statent.density")

    context.config("threads")
    context.config("locations_v2_analysis_max_points", 250000)
    context.config("locations_v2_analysis_density_radius", 250.0)


def _to_num(s, fallback=np.nan):
    out = pd.to_numeric(s, errors="coerce")
    if np.isnan(fallback):
        return out
    return out.fillna(fallback)


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _weighted_share(mask, weights):
    return _weighted_mean(np.asarray(mask, dtype=float), weights)


def _distance_from_xy(df, ox, oy, dx, dy):
    return np.sqrt((df[dx] - df[ox]) ** 2 + (df[dy] - df[oy]) ** 2)


def _clip_distance(series, max_distance=MAX_DISTANCE_M):
    s = _to_num(series)
    s = s.where((s >= 0.0) & (s <= max_distance), np.nan)
    return s


def _normalize_ovgk(series):
    s = pd.Series(series).astype(str).str.upper().fillna("NONE")
    s = s.where(s.isin(["A", "B", "C", "D"]), "None")
    return s


def _recode_car_availability(series):
    x = pd.to_numeric(series, errors="coerce")
    # User rule: 0 or 1 means available, 2 means not available.
    out = np.where(x.isin([0, 1]), 1, np.where(x == 2, 0, np.nan))
    return pd.Series(out, index=series.index)


def _attach_trip_context(df, pid_col="person_id", trip_col="trip_id"):
    df = df.sort_values([pid_col, trip_col]).copy()
    df["prev_purpose"] = df.groupby(pid_col)["origin_purpose"].shift(0)
    df["next_purpose"] = df.groupby(pid_col)["purpose"].shift(-1)
    return df


def _impute_ovgk_for_points(context, df, x_col, y_col, out_col):
    if x_col not in df.columns or y_col not in df.columns:
        df[out_col] = "None"
        return df

    sample_max = int(context.config("locations_v2_analysis_max_points"))
    if len(df) > sample_max:
        keep = np.random.RandomState(123).choice(np.arange(len(df)), size=sample_max, replace=False)
        base = df.iloc[keep].copy()
    else:
        base = df.copy()

    g = gpd.GeoDataFrame(base[[x_col, y_col]].copy(), geometry=gpd.points_from_xy(base[x_col], base[y_col]), crs="EPSG:2056")
    g = impute_ovgk_parallel(
        context,
        g,
        x=x_col,
        y=y_col,
        geometry="geometry",
        output_column=out_col,
        point_type="microcensus destination",
        chunk_size=5000,
        n_jobs=max(1, min(int(context.config("threads")), 8)),
    )

    base[out_col] = g[out_col].astype(str)
    merged = df.reset_index().merge(base.reset_index()[["index", out_col]], on="index", how="left").drop(columns=["index"])
    merged[out_col] = merged[out_col].fillna("None")
    return merged


def _impute_statent_density_for_points(context, df, x_col, y_col):
    if x_col not in df.columns or y_col not in df.columns:
        df["companies_density"] = np.nan
        df["employees_density"] = np.nan
        return df

    sample_max = int(context.config("locations_v2_analysis_max_points"))
    if len(df) > sample_max:
        keep = np.random.RandomState(123).choice(np.arange(len(df)), size=sample_max, replace=False)
        base = df.iloc[keep].copy()
    else:
        base = df.copy()

    radius = float(context.config("locations_v2_analysis_density_radius"))
    n_jobs = max(1, min(int(context.config("threads")), 8))

    base = impute_statent_density(
        context,
        base,
        x=x_col,
        y=y_col,
        radius=radius,
        point_type="microcensus destination",
        chunk_size=5000,
        n_jobs=n_jobs,
        measure="companies",
        output_column="companies_density",
    )

    base = impute_statent_density(
        context,
        base,
        x=x_col,
        y=y_col,
        radius=radius,
        point_type="microcensus destination",
        chunk_size=5000,
        n_jobs=n_jobs,
        measure="employees",
        output_column="employees_density",
    )

    merged = df.reset_index().merge(
        base.reset_index()[["index", "companies_density", "employees_density"]], on="index", how="left"
    ).drop(columns=["index"])

    return merged


def _prepare_microcensus(context):
    persons = context.stage("data.microcensus.persons").copy()
    trips = context.stage("data.microcensus.trips")[0].copy()

    trips = _attach_trip_context(trips)

    if "crowfly_distance" in trips.columns:
        trips["distance_prev"] = _to_num(trips["crowfly_distance"])
    elif all(c in trips.columns for c in ["origin_x", "origin_y", "destination_x", "destination_y"]):
        trips["distance_prev"] = _distance_from_xy(trips, "origin_x", "origin_y", "destination_x", "destination_y")
    else:
        trips["distance_prev"] = np.nan

    home_from_origin = trips[trips["origin_purpose"] == "home"][["person_id", "origin_x", "origin_y"]].dropna()
    home_from_origin = home_from_origin.groupby("person_id", as_index=False).first().rename(columns={"origin_x": "home_x", "origin_y": "home_y"})
    home_from_dest = trips[trips["purpose"] == "home"][["person_id", "destination_x", "destination_y"]].dropna()
    home_from_dest = home_from_dest.groupby("person_id", as_index=False).first().rename(columns={"destination_x": "home_x", "destination_y": "home_y"})
    homes = pd.concat([home_from_origin, home_from_dest], ignore_index=True).drop_duplicates(subset=["person_id"], keep="first")

    work_from_origin = trips[trips["origin_purpose"] == "work"][["person_id", "origin_x", "origin_y"]].dropna()
    work_from_origin = work_from_origin.groupby("person_id", as_index=False).first().rename(columns={"origin_x": "work_x", "origin_y": "work_y"})
    work_from_dest = trips[trips["purpose"] == "work"][["person_id", "destination_x", "destination_y"]].dropna()
    work_from_dest = work_from_dest.groupby("person_id", as_index=False).first().rename(columns={"destination_x": "work_x", "destination_y": "work_y"})
    works = pd.concat([work_from_origin, work_from_dest], ignore_index=True).drop_duplicates(subset=["person_id"], keep="first")

    sec = trips[~trips["purpose"].isin(PRIMARY)].copy()
    sec = sec.merge(homes, on="person_id", how="left")
    sec = sec.merge(works, on="person_id", how="left")

    if all(c in sec.columns for c in ["destination_x", "destination_y", "home_x", "home_y"]):
        sec["distance_home"] = _distance_from_xy(sec, "home_x", "home_y", "destination_x", "destination_y")
    else:
        sec["distance_home"] = np.nan

    if all(c in sec.columns for c in ["destination_x", "destination_y", "work_x", "work_y"]):
        sec["distance_work"] = _distance_from_xy(sec, "work_x", "work_y", "destination_x", "destination_y")
    else:
        sec["distance_work"] = np.nan

    keep_attrs = [
        c
        for c in ["person_id", "person_weight", "sex", "age", "age_class", "income_class", "car_availability", "ovgk", "municipality_type"]
        if c in persons.columns
    ]
    sec = sec.merge(persons[keep_attrs], on="person_id", how="left")
    if "car_availability" in sec.columns:
        sec["car_availability"] = _recode_car_availability(sec["car_availability"])

    if "ovgk" in sec.columns:
        sec["ovgk"] = _normalize_ovgk(sec["ovgk"])

    sec = _impute_ovgk_for_points(context, sec, "destination_x", "destination_y", "destination_ovgk")
    sec["destination_ovgk"] = _normalize_ovgk(sec["destination_ovgk"])
    sec = _impute_statent_density_for_points(context, sec, "destination_x", "destination_y")
    sec["destination_ovgk_score"] = sec["destination_ovgk"].map(OVGK_SCORE).fillna(0).astype(float)
    sec["log_companies_density"] = np.log1p(_to_num(sec.get("companies_density", np.nan)))
    sec["log_employees_density"] = np.log1p(_to_num(sec.get("employees_density", np.nan)))

    sec["distance_prev"] = _clip_distance(sec.get("distance_prev", np.nan))
    sec["distance_home"] = _clip_distance(sec.get("distance_home", np.nan))
    sec["distance_work"] = _clip_distance(sec.get("distance_work", np.nan))
    sec["distance"] = sec["distance_prev"]

    sec = sec[
        sec[["distance_prev", "distance_home", "distance_work"]].notna().any(axis=1)
    ].copy()

    if "person_weight" not in sec.columns:
        sec["person_weight"] = 1.0

    return sec


def _overall_metrics(df):
    w = _to_num(df.get("person_weight", 1.0), 1.0)
    d_prev = _to_num(df.get("distance_prev", np.nan))
    d_home = _to_num(df.get("distance_home", np.nan))
    d_work = _to_num(df.get("distance_work", np.nan))
    ovgk = _to_num(df.get("destination_ovgk_score", np.nan))
    comp = _to_num(df.get("companies_density", np.nan))
    emp = _to_num(df.get("employees_density", np.nan))

    return pd.DataFrame(
        {
            "n_records": [int(len(df))],
            "mean_distance_prev_m": [_weighted_mean(d_prev, w)],
            "mean_distance_home_m": [_weighted_mean(d_home, w)],
            "mean_distance_work_m": [_weighted_mean(d_work, w)],
            "median_distance_prev_m": [float(np.nanmedian(d_prev)) if np.isfinite(d_prev).any() else np.nan],
            "median_distance_home_m": [float(np.nanmedian(d_home)) if np.isfinite(d_home).any() else np.nan],
            "median_distance_work_m": [float(np.nanmedian(d_work)) if np.isfinite(d_work).any() else np.nan],
            "mean_destination_ovgk_score": [_weighted_mean(ovgk, w)],
            "share_high_pt_access_ovgk_ab": [_weighted_share(df.get("destination_ovgk", "None").isin(["A", "B"]), w)],
            "mean_companies_density": [_weighted_mean(comp, w)],
            "mean_employees_density": [_weighted_mean(emp, w)],
        }
    )


def _by_attribute(df, attr):
    if attr not in df.columns:
        return pd.DataFrame(
            columns=[
                "attribute",
                "group",
                "n",
                "mean_distance_m",
                "mean_ovgk_score",
                "share_high_pt_ab",
                "mean_companies_density",
                "mean_employees_density",
            ]
        )

    x = df[[attr, "distance", "destination_ovgk_score", "destination_ovgk", "person_weight", "companies_density", "employees_density"]].copy()
    x["distance_prev"] = df.get("distance_prev", np.nan)
    x["distance_home"] = df.get("distance_home", np.nan)
    x["distance_work"] = df.get("distance_work", np.nan)
    x[attr] = x[attr].astype("string").fillna("nan").astype(str)

    rows = []
    for g, part in x.groupby(attr):
        w = _to_num(part["person_weight"], 1.0)
        rows.append(
            {
                "attribute": attr,
                "group": g,
                "n": int(len(part)),
                "mean_distance_prev_m": _weighted_mean(part["distance_prev"], w),
                "mean_distance_home_m": _weighted_mean(part["distance_home"], w),
                "mean_distance_work_m": _weighted_mean(part["distance_work"], w),
                "mean_ovgk_score": _weighted_mean(part["destination_ovgk_score"], w),
                "share_high_pt_ab": _weighted_share(part["destination_ovgk"].isin(["A", "B"]), w),
                "mean_companies_density": _weighted_mean(part["companies_density"], w),
                "mean_employees_density": _weighted_mean(part["employees_density"], w),
            }
        )

    return pd.DataFrame(rows)


def _spearman_like_corr(x, y):
    xv = pd.Series(x).astype(float)
    yv = pd.Series(y).astype(float)
    m = xv.notna() & yv.notna()
    if m.sum() < 25:
        return np.nan
    return float(xv[m].rank().corr(yv[m].rank()))


def _correlation_table(df):
    attrs = [c for c in ["age", "age_class", "income_class", "car_availability"] if c in df.columns]
    targets = [
        c
        for c in [
            "distance_prev",
            "distance_home",
            "distance_work",
            "destination_ovgk_score",
            "companies_density",
            "employees_density",
            "log_companies_density",
            "log_employees_density",
        ]
        if c in df.columns
    ]

    rows = []
    for a in attrs:
        for t in targets:
            r = _spearman_like_corr(df[a], df[t])
            if pd.notna(r):
                rows.append({"attribute": a, "target": t, "spearman_r": r, "abs_r": abs(r)})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return out


def _purpose_profiles(df, purpose_col):
    if purpose_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "purpose",
                "n",
                "mean_distance_m",
                "mean_ovgk_score",
                "mean_companies_density",
                "mean_employees_density",
            ]
        )

    rows = []
    for p, part in df.groupby(purpose_col):
        w = _to_num(part.get("person_weight", 1.0), 1.0)
        rows.append(
            {
                "purpose": p,
                "n": int(len(part)),
                "mean_distance_prev_m": _weighted_mean(part.get("distance_prev", np.nan), w),
                "mean_distance_home_m": _weighted_mean(part.get("distance_home", np.nan), w),
                "mean_distance_work_m": _weighted_mean(part.get("distance_work", np.nan), w),
                "mean_ovgk_score": _weighted_mean(part.get("destination_ovgk_score", np.nan), w),
                "mean_companies_density": _weighted_mean(part.get("companies_density", np.nan), w),
                "mean_employees_density": _weighted_mean(part.get("employees_density", np.nan), w),
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def _correlation_ratio(categories, values):
    c = pd.Series(categories).astype("string")
    v = pd.to_numeric(values, errors="coerce")
    mask = c.notna() & v.notna()
    c = c[mask]
    v = v[mask]

    if len(v) < 50:
        return np.nan

    grand_mean = v.mean()
    denom = ((v - grand_mean) ** 2).sum()
    if denom <= 0:
        return np.nan

    num = 0.0
    for _, grp in v.groupby(c):
        if len(grp) == 0:
            continue
        num += len(grp) * (grp.mean() - grand_mean) ** 2

    return float(num / denom)


def _purpose_correlation_table(df):
    if "purpose" not in df.columns:
        return pd.DataFrame(columns=["predictor", "target", "eta2"])

    targets = [
        c
        for c in [
            "distance_prev",
            "distance_home",
            "distance_work",
            "destination_ovgk_score",
            "companies_density",
            "employees_density",
            "log_companies_density",
            "log_employees_density",
        ]
        if c in df.columns
    ]

    rows = []
    for target in targets:
        eta2 = _correlation_ratio(df["purpose"], df[target])
        if pd.notna(eta2):
            rows.append({"predictor": "purpose", "target": target, "eta2": eta2})

    if "next_purpose" in df.columns:
        for target in targets:
            eta2 = _correlation_ratio(df["next_purpose"], df[target])
            if pd.notna(eta2):
                rows.append({"predictor": "next_purpose", "target": target, "eta2": eta2})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("eta2", ascending=False).reset_index(drop=True)
    return out


def _sex_distance_test(df):
    if "sex" not in df.columns:
        return pd.DataFrame(columns=["group_a", "group_b", "distance_type", "mean_a", "mean_b", "ratio_b_over_a"])

    x = df[["sex", "person_weight", "distance_prev", "distance_home", "distance_work"]].copy()
    x["sex"] = x["sex"].astype(str).str.lower()

    female = x[x["sex"].isin(["f", "female", "2", "woman"])]
    male = x[x["sex"].isin(["m", "male", "1", "man"])]

    if len(female) == 0 or len(male) == 0:
        return pd.DataFrame(columns=["group_a", "group_b", "distance_type", "mean_a", "mean_b", "ratio_b_over_a"])

    rows = []
    for d in ["distance_prev", "distance_home", "distance_work"]:
        mean_f = _weighted_mean(female[d], _to_num(female.get("person_weight", 1.0), 1.0))
        mean_m = _weighted_mean(male[d], _to_num(male.get("person_weight", 1.0), 1.0))
        rows.append(
            {
                "group_a": "female",
                "group_b": "male",
                "distance_type": d,
                "mean_a": mean_f,
                "mean_b": mean_m,
                "ratio_b_over_a": mean_m / mean_f if pd.notna(mean_f) and mean_f > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _save_figure(context, fig, filename):
    out = os.path.join(context.path(), filename)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_distance_distribution(context, mc):
    distances = {
        "From Previous": _to_num(mc.get("distance_prev", np.nan)).replace([np.inf, -np.inf], np.nan).dropna(),
        "From Home": _to_num(mc.get("distance_home", np.nan)).replace([np.inf, -np.inf], np.nan).dropna(),
        "From Work": _to_num(mc.get("distance_work", np.nan)).replace([np.inf, -np.inf], np.nan).dropna(),
    }
    if all(len(v) == 0 for v in distances.values()):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (label, x) in zip(axes, distances.items()):
        if len(x) > 0:
            q99 = float(np.nanpercentile(x, 99))
            x_clip = x[x <= q99] if q99 > 0 else x
            ax.hist(x_clip, bins=40, color="#1f77b4", alpha=0.9)
        ax.set_title(label)
        ax.set_xlabel("Distance (m)")
    axes[0].set_ylabel("Count")
    fig.suptitle("Secondary Distance Distributions (<= 50 km)")
    return _save_figure(context, fig, "analysis_distance_distribution.png")


def _plot_ovgk_by_purpose(context, purpose_profiles):
    if len(purpose_profiles) == 0:
        return None
    df = purpose_profiles.copy()
    df["purpose"] = df["purpose"].astype(str)
    df = df.sort_values("purpose").reset_index(drop=True)
    dcols = ["mean_distance_prev_m", "mean_distance_home_m", "mean_distance_work_m"]
    labels = ["From Previous", "From Home", "From Work"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, col, label in zip(axes, dcols, labels):
        ax.scatter(df[col], df["mean_ovgk_score"], color="#2ca02c", alpha=0.85)
        for _, r in df.iterrows():
            ax.text(r[col], r["mean_ovgk_score"], str(r["purpose"]), fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Mean distance (m)")
    axes[0].set_ylabel("Mean OVGK score")
    fig.suptitle("Purpose-Level OVGK vs Distance")
    return _save_figure(context, fig, "analysis_ovgk_by_purpose.png")


def _plot_density_by_purpose(context, purpose_profiles):
    if len(purpose_profiles) == 0:
        return None
    df = purpose_profiles.copy()
    df["purpose"] = df["purpose"].astype(str)
    df = df.sort_values("purpose").reset_index(drop=True)
    dcols = ["mean_distance_prev_m", "mean_distance_home_m", "mean_distance_work_m"]
    labels = ["From Previous", "From Home", "From Work"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, col, label in zip(axes, dcols, labels):
        ax.scatter(df[col], df["mean_companies_density"], color="#ff7f0e", alpha=0.85, label="Companies")
        ax.scatter(df[col], df["mean_employees_density"], color="#d62728", alpha=0.65, label="Employees")
        for _, r in df.iterrows():
            # Label each purpose at the companies-density point to keep the plot readable.
            ax.text(r[col], r["mean_companies_density"], str(r["purpose"]), fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Mean distance (m)")
    axes[0].set_ylabel("Mean density within radius")
    axes[-1].legend()
    fig.suptitle("Purpose-Level STATENT Density vs Distance")
    return _save_figure(context, fig, "analysis_statent_density_by_purpose.png")


def _plot_attribute_distance(context, by_attr, attribute):
    part = by_attr[by_attr["attribute"] == attribute].copy() if len(by_attr) else pd.DataFrame()
    if len(part) == 0:
        return None

    part["group"] = part["group"].astype(str)
    part = part.sort_values("group").reset_index(drop=True)

    groups = part["group"].astype(str).values
    x = np.arange(len(groups))
    width = 0.28

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, part["mean_distance_prev_m"], width=width, label="From previous", color="#9467bd", alpha=0.9)
    ax.bar(x, part["mean_distance_home_m"], width=width, label="From home", color="#1f77b4", alpha=0.8)
    ax.bar(x + width, part["mean_distance_work_m"], width=width, label="From work", color="#2ca02c", alpha=0.8)
    ax.set_title(f"Mean Secondary Distances by {attribute}")
    ax.set_xlabel(attribute)
    ax.set_ylabel("Mean distance (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    return _save_figure(context, fig, f"analysis_distance_by_{attribute}.png")


def _plot_transition_heatmap(context, transitions):
    if len(transitions) == 0:
        return None

    top = transitions.copy()
    top_rows = np.sort(top["purpose"].astype(str).unique())[:8]
    top_cols = np.sort(top["next_purpose"].astype(str).unique())[:8]

    mat = (
        top[top["purpose"].isin(top_rows) & top["next_purpose"].isin(top_cols)]
        .pivot_table(index="purpose", columns="next_purpose", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=top_rows, columns=top_cols, fill_value=0)
    )
    if mat.size == 0:
        return None

    positive = mat.values[mat.values > 0]
    if len(positive) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    values = np.ma.masked_less_equal(mat.values, 0)
    norm = matplotlib.colors.LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    im = ax.imshow(values, cmap="Blues", aspect="auto", norm=norm)
    ax.set_title("Top Secondary Purpose Transitions (Distance context in other figures)")
    ax.set_xlabel("Next purpose")
    ax.set_ylabel("Current purpose")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    fig.colorbar(im, ax=ax, label="Count (log scale)")
    return _save_figure(context, fig, "analysis_transition_heatmap.png")


def _plot_density_distance_relationship(context, mc):
    max_plot_distance = 20000.0
    dmap = {
        "From Previous": _to_num(mc.get("distance_prev", np.nan)).replace([np.inf, -np.inf], np.nan),
        "From Home": _to_num(mc.get("distance_home", np.nan)).replace([np.inf, -np.inf], np.nan),
        "From Work": _to_num(mc.get("distance_work", np.nan)).replace([np.inf, -np.inf], np.nan),
    }
    y = _to_num(mc.get("log_companies_density", np.nan)).replace([np.inf, -np.inf], np.nan)
    if all((x.notna() & y.notna()).sum() < 50 for x in dmap.values()):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, (label, x) in zip(axes, dmap.items()):
        m = x.notna() & y.notna() & (x <= max_plot_distance)
        if m.sum() >= 50:
            hb = ax.hexbin(x[m], y[m], gridsize=35, cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Distance (m)")
    axes[0].set_ylabel("log(1 + companies density)")
    fig.suptitle("Distance vs Companies Density")
    return _save_figure(context, fig, "analysis_distance_vs_companies_density.png")


def _plot_correlation_bars(context, corr):
    if len(corr) == 0:
        return None

    target_groups = ["distance_prev", "distance_home", "distance_work"]
    titles = ["From Previous", "From Home", "From Work"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

    for ax, tgt, ttl in zip(axes, target_groups, titles):
        top = corr[corr["target"] == tgt].copy()
        top["attribute"] = top["attribute"].astype(str)
        top = top.sort_values("attribute").head(8)
        if len(top) == 0:
            ax.set_title(ttl)
            ax.set_xlabel("Correlation")
            continue
        labels = top["attribute"].astype(str) + " -> " + top["target"].astype(str)
        ax.barh(labels[::-1], top["spearman_r"].values[::-1], color="#17becf", alpha=0.9)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(ttl)
        ax.set_xlabel("Correlation")

    fig.suptitle("Top Numeric Correlations by Distance Definition")
    return _save_figure(context, fig, "analysis_top_numeric_correlations.png")


def _create_figures(context, mc, by_attr, corr, purpose, transitions):
    files = []
    plot_jobs = [
        lambda: _plot_distance_distribution(context, mc),
        lambda: _plot_ovgk_by_purpose(context, purpose),
        lambda: _plot_density_by_purpose(context, purpose),
        lambda: _plot_transition_heatmap(context, transitions),
        lambda: _plot_density_distance_relationship(context, mc),
        lambda: _plot_correlation_bars(context, corr),
        lambda: _plot_attribute_distance(context, by_attr, "sex"),
        lambda: _plot_attribute_distance(context, by_attr, "age_class"),
        lambda: _plot_attribute_distance(context, by_attr, "car_availability"),
        lambda: _plot_attribute_distance(context, by_attr, "income_class"),
    ]

    for job in plot_jobs:
        try:
            output = job()
            if output is not None:
                files.append(output)
        except Exception as error:
            logger.warning("Skipping one analysis figure due to plotting error: %s", error)

    return files


def execute(context):
    mc = _prepare_microcensus(context)

    overall = _overall_metrics(mc)

    attr_cols = ["sex", "age_class", "car_availability", "income_class", "municipality_type"]
    by_attr = pd.concat([_by_attribute(mc, a) for a in attr_cols], ignore_index=True)

    corr = _correlation_table(mc)

    purpose = _purpose_profiles(mc, "purpose")
    purpose_corr = _purpose_correlation_table(mc)
    sex_distance = _sex_distance_test(mc)

    # Transition patterns from microcensus secondary chains.
    transitions = mc.dropna(subset=["purpose", "next_purpose"]).groupby(["purpose", "next_purpose"]).size().reset_index(name="count")
    transitions = transitions.sort_values("count", ascending=False)

    figure_files = _create_figures(context, mc, by_attr, corr, purpose, transitions)

    logger.info("locations_v2 analysis prepared (microcensus only): %d rows, %d figures", len(mc), len(figure_files))

    return {
        "overall": overall,
        "by_attribute": by_attr,
        "numeric_correlations": corr,
        "purpose_correlations": purpose_corr,
        "sex_distance_summary": sex_distance,
        "purpose_profiles": purpose,
        "microcensus_transitions": transitions,
        "microcensus_secondary_trips": mc,
        "figure_files": figure_files,
    }
