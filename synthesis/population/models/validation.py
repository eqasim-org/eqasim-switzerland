import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("synthesis.population.models.students")


def _find_survey_weight_column(df):
    for col in ["weight", "person_weight"]:
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find a survey weight column. Expected one of: 'weight', 'person_weight'."
    )


def _prepare_binary_column(df, col):
    out = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return (out > 0).astype(int)


def _prepare_group_column(df, col):
    return df[col].astype("Int64").astype(str)


def _aggregate_positive_counts(df, group_col, value_col, weight_col=None):
    tmp = df[[group_col, value_col]].copy()

    if weight_col is not None:
        tmp["_weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    else:
        tmp["_weight"] = 1.0

    tmp = tmp.dropna(subset=[group_col])
    tmp["_positive_weight"] = tmp[value_col] * tmp["_weight"]

    result = (
        tmp.groupby(group_col, dropna=False)["_positive_weight"]
        .sum()
        .reset_index()
        .rename(columns={"_positive_weight": "count"})
    )

    return result


def _build_comparison_table(survey_df, pop_df, geography_col, variable_col, survey_weight_col):
    survey_counts = _aggregate_positive_counts(
        survey_df,
        group_col=geography_col,
        value_col=variable_col,
        weight_col=survey_weight_col,
    ).rename(columns={"count": "survey_weighted"})

    pop_counts = _aggregate_positive_counts(
        pop_df,
        group_col=geography_col,
        value_col=variable_col,
        weight_col=None,
    ).rename(columns={"count": "synthetic_count"})

    comparison = pd.merge(
        survey_counts,
        pop_counts,
        on=geography_col,
        how="outer"
    ).fillna(0.0)

    # Annualize survey because it contains 3 years
    comparison["survey_weighted_annual"] = comparison["survey_weighted"] / 3.0

    comparison["abs_diff"] = comparison["synthetic_count"] - comparison["survey_weighted_annual"]
    comparison["rel_diff"] = np.where(
        comparison["survey_weighted_annual"] > 0,
        comparison["abs_diff"] / comparison["survey_weighted_annual"],
        np.nan,
    )

    return comparison.sort_values(geography_col).reset_index(drop=True)


def _thousands_formatter(x, pos):
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)


def _compute_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.full_like(x, np.nan)

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return slope, intercept, r2, y_hat


def _make_scatterplot(comparison_df, geography_col, variable_col, output_path, annotate=False):
    fig, ax = plt.subplots(figsize=(8, 8))

    x = comparison_df["survey_weighted_annual"].to_numpy(dtype=float)
    y = comparison_df["synthetic_count"].to_numpy(dtype=float)
    labels = comparison_df[geography_col].astype(str).tolist()

    ax.scatter(x, y, s=35, alpha=0.8)

    slope, intercept, r2, y_hat = _compute_r2(x, y)
    if len(x) >= 2 and not np.isnan(slope):
        x_line = np.linspace(np.min(x), np.max(x), 200)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            linewidth=2,
            label=f"Fit: y = {slope:.3f}x + {intercept:.1f}\n$R^2$ = {r2:.3f}"
        )

    vmin = min(np.min(x), np.min(y))
    vmax = max(np.max(x), np.max(y))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1, alpha=0.6, label="y = x")

    if annotate:
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)

    geo_label = "canton" if geography_col == "canton_id" else "district"
    var_label = "employed" if variable_col == "employed" else "is_student"

    ax.set_title(f"Validation of '{var_label}' by {geo_label} (age 15+)")
    ax.set_xlabel("Structural survey (weighted annualized)")
    ax.set_ylabel("Synthetic population")

    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands_formatter))

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def execute(context):
    output_dir = context.path()
    os.makedirs(output_dir, exist_ok=True)

    survey_df = context.stage("data.structural_survey.structural_survey").copy()
    survey_df["canton_id"] = pd.to_numeric(survey_df["canton_id"], errors="coerce").astype("Int64")
    survey_df["district_id"] = pd.to_numeric(survey_df["district_id"], errors="coerce").astype("Int64")
    pop_df = context.stage("synthesis.population.models.students").copy()
    pop_df["canton_id"] = pd.to_numeric(pop_df["canton_id"], errors="coerce").astype("Int64")
    pop_df["district_id"] = pd.to_numeric(pop_df["district_id"], errors="coerce").astype("Int64")

    survey_weight_col = _find_survey_weight_column(survey_df)

    required_cols = ["age", "canton_id", "district_id", "employed", "is_student"]
    for col in required_cols:
        if col not in survey_df.columns:
            raise KeyError(f"Missing column '{col}' in structural survey dataframe.")
        if col not in pop_df.columns:
            raise KeyError(f"Missing column '{col}' in subscriptions output dataframe.")

    survey_df = survey_df[pd.to_numeric(survey_df["age"], errors="coerce") >= 15].copy()
    pop_df = pop_df[pd.to_numeric(pop_df["age"], errors="coerce") >= 15].copy()

    for col in ["employed", "is_student"]:
        survey_df[col] = _prepare_binary_column(survey_df, col)
        pop_df[col] = _prepare_binary_column(pop_df, col)

    for col in ["canton_id", "district_id"]:
        survey_df[col] = _prepare_group_column(survey_df, col)
        pop_df[col] = _prepare_group_column(pop_df, col)

    results = {}

    tasks = [
        ("canton_id", "employed"),
        ("canton_id", "is_student"),
        ("district_id", "employed"),
        ("district_id", "is_student"),
    ]

    for geography_col, variable_col in tasks:
        comparison = _build_comparison_table(
            survey_df=survey_df,
            pop_df=pop_df,
            geography_col=geography_col,
            variable_col=variable_col,
            survey_weight_col=survey_weight_col,
        )

        geography_suffix = "canton" if geography_col == "canton_id" else "district"
        pdf_name = f"validation_{variable_col}_by_{geography_suffix}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        _make_scatterplot(
            comparison_df=comparison,
            geography_col=geography_col,
            variable_col=variable_col,
            output_path=pdf_path,
            annotate=(geography_col == "canton_id")
        )

        csv_name = f"validation_{variable_col}_by_{geography_suffix}.csv"
        comparison.to_csv(os.path.join(output_dir, csv_name), index=False)

        results[f"{variable_col}_{geography_suffix}"] = comparison
        print(f"Saved {pdf_path}")

    return results