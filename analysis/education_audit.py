"""Audit source education codes without changing student or location assignments."""

import argparse
from pathlib import Path

import pandas as pd

import data.structural_survey.raw
from data.structural_survey.structural_survey import classify_education


def configure(context):
    context.stage("data.structural_survey.raw")
    context.stage("data.statent.statent")
    context.config("analysis_path")


# NOGA 2008 institution types; these do not identify the student's study level.
TERTIARY_NOGA = {
    "854201": "UH",
    "854202": "FH_PH",
    "854203": "higher_vocational",
}


def school_type(codes):
    result = codes.map(TERTIARY_NOGA).fillna("other_noga")
    missing = codes.isna() | codes.eq("") | codes.str.startswith("-", na=False)
    result.loc[missing] = "missing_code"
    return result


def summarize(df, groups):
    counts = df.groupby(groups, dropna=False).size().rename("respondents")
    weights = df.groupby(groups, dropna=False)["weight"].sum().rename("weighted_persons")
    return pd.concat([counts, weights], axis=1).reset_index()


def audit(df_survey, df_statent, output_path):
    """Describe survey codes and STATENT evidence before choosing student groups."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df = df_survey.copy()
    df["school_type"] = school_type(df["education_noga"])
    df["school_enterprise_type"] = school_type(df["education_enterprise_noga"])

    # Audit raw employment status, before the pipeline's apprenticeship correction.
    flags = {
        "in_education": df["in_education"].eq(1),
        "student": df["is_student"].eq(1),
        "employed": df["employed"].eq(1),
        "employed_student": df["employed"].eq(1) & df["is_student"].eq(1),
        "valid_school_municipality": df["education_municipality"].gt(0),
        "positive_school_frequency": df["education_freq_per_week"].gt(0),
        "local_tertiary_noga": df["education_noga"].isin(TERTIARY_NOGA),
        "enterprise_tertiary_noga": df["education_enterprise_noga"].isin(TERTIARY_NOGA),
        "tertiary_type_conflict": (
            df["education_noga"].isin(TERTIARY_NOGA)
            & df["education_enterprise_noga"].isin(TERTIARY_NOGA)
            & df["school_type"].ne(df["school_enterprise_type"])
        ),
        "student_flag_disagreement": df["in_education"].ne(df["is_student"]),
    }
    groups = ["survey_year", "current_education"]
    summary = summarize(df, groups).set_index(groups)
    for name, flag in flags.items():
        summary[name + "_respondents"] = flag.astype(int).groupby(
            [df[column] for column in groups]).sum()
        summary[name + "_weighted"] = (flag.astype(int) * df["weight"]).groupby(
            [df[column] for column in groups]).sum()
    summary["median_age"] = df.groupby(groups)["age"].median()
    summary.reset_index().to_csv(output_path / "education_by_code.csv", index=False)

    tables = {
        "school_type_by_code": ["survey_year", "current_education", "school_type", "school_enterprise_type"],
        "school_noga_by_code": ["survey_year", "current_education", "education_noga", "education_enterprise_noga"],
        "employment_by_code": ["survey_year", "current_education", "employed", "is_student", "in_education"],
        "commuting_by_code": ["survey_year", "current_education", "commute_to_education"],
        "school_frequency_by_code": ["survey_year", "current_education", "education_freq_per_week"],
        "employer_type_by_code": ["survey_year", "current_education", "employed", "work_noga", "work_enterprise_noga"],
    }
    for name, columns in tables.items():
        summarize(df, columns).to_csv(output_path / (name + ".csv"), index=False)

    classified = classify_education(df)
    classified["age_group"] = pd.cut(
        classified["age"], bins=[0, 15, 20, 25, 30, 40, 65, float("inf")],
        labels=["under_15", "15_19", "20_24", "25_29", "30_39", "40_64", "65_plus"],
        right=False,
    ).astype(str)
    classification_tables = {
        "classification": ["survey_year", "study_level", "institution_type", "education_od_group",
                           "institution_type_source", "institution_type_conflict", "study_institution_conflict",
                           "education_participation_conflict", "education_codebook_assumed"],
        "classification_by_age": ["survey_year", "study_level", "education_od_group", "age_group"],
        "classification_by_employment": ["survey_year", "study_level", "education_od_group", "employed"],
    }
    for name, columns in classification_tables.items():
        summarize(classified, columns).to_csv(output_path / (name + ".csv"), index=False)

    unresolved = classified[classified["education_od_group"].eq("unresolved")].copy()
    unresolved["reason"] = "missing_institution"
    unresolved.loc[unresolved["study_level"].eq("unresolved"), "reason"] = "missing_study_level"
    for flag, reason in [("study_institution_conflict", "study_institution_conflict"),
                         ("institution_type_conflict", "institution_type_conflict"),
                         ("education_participation_conflict", "participation_conflict")]:
        unresolved.loc[unresolved[flag], "reason"] = reason
    summarize(unresolved, ["survey_year", "current_education", "reason"]).to_csv(
        output_path / "unresolved_reasons.csv", index=False)
    unresolved["destination_status"] = "unknown_country_and_municipality"
    unresolved.loc[unresolved["education_municipality"].gt(0),
                   "destination_status"] = "valid_school_municipality"
    foreign = unresolved["education_country_code"].gt(0) & unresolved["education_country_code"].ne(8100)
    unresolved.loc[foreign, "destination_status"] = "reported_foreign_country"
    summarize(unresolved, ["survey_year", "current_education", "destination_status"]).to_csv(
        output_path / "unresolved_destinations.csv", index=False)

    # Enterprise counts are grouping diagnostics, not counts of named universities.
    tertiary = df_statent[df_statent["noga"].isin(TERTIARY_NOGA)].copy()
    tertiary.groupby("noga").agg(
        establishments=("statent_local_id", "size"),
        enterprises=("statent_enterprise_id", "nunique"),
        employees=("number_employees", "sum"),
        employees_fte=("number_employees_fte", "sum"),
        median_establishment_employees=("number_employees", "median"),
    ).reset_index().to_csv(output_path / "statent_tertiary_summary.csv", index=False)
    tertiary.groupby(["noga", "statent_enterprise_id"]).agg(
        establishments=("statent_local_id", "size"),
        employees=("number_employees", "sum"),
    ).reset_index().to_csv(output_path / "statent_enterprise_groups.csv", index=False)

    coverage = summarize(df, ["survey_year"])
    coverage.to_csv(output_path / "survey_coverage.csv", index=False)
    (output_path / "README.md").write_text(
        "# Education input audit\n\n"
        "Source: structural survey 2021–2023 and STATENT 2022. "
        "Each annual weighted total is reported separately; summing the three "
        "years does not estimate one year's population.\n\n"
        "Current education and commuting variables remain raw codes. The supplied "
        "2023 codebook do-d-40-se_user-code-2023 (2).xlsx confirms education "
        "codes 11 = bachelor, 12 = master, 13 = doctorate/habilitation (sheet 25), "
        "and detailed school commuting codes (sheet 51). Historical applicability "
        "to 2021–2022 still needs release-specific confirmation. Frequency "
        "categories describe commuting movements per week (sheet 46), not attendance "
        "days; the raw frequency's precise journey-counting convention is not defined "
        "by this code list.\n\n"
        "Employment is audited before the existing apprenticeship correction. "
        "Median ages are unweighted. Negative NOGA codes are grouped as missing "
        "only for coverage analysis; their individual meanings are not assumed. "
        "Other NOGA codes and disagreements are retained for review.\n\n"
        "Classification tables use the same classify_education helper as the "
        "processed survey. Study level, institution type, OD group, evidence source "
        "and review flags remain separate. FH/PH and higher vocational education "
        "share the other_tertiary OD group. Unresolved or conflicting tertiary "
        "observations are retained as unresolved rather than dropped.\n\n"
        "UH = 854201; FH/PH = 854202; higher vocational = 854203. "
        "These are institution types, not personal study levels. Source: "
        "https://www.kubb-tool.bfs.admin.ch/fr/noga/2008/8542\n\n"
        "All CSV outputs contain aggregates, not respondent records. "
        "Enterprise groups are anonymous and have not been matched to named institutions.\n",
        encoding="utf-8",
    )
    return coverage


def execute(context):
    return audit(
        context.stage("data.structural_survey.raw"),
        context.stage("data.statent.statent"),
        Path(context.config("analysis_path")) / "education_audit",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    # Run directly on the inputs without rebuilding spatial stages.
    class Context:
        def config(self, name):
            if name == "data_path":
                return args.data_path
            raise KeyError(name)

        def progress(self, items, **kwargs):
            return items

    survey = data.structural_survey.raw.execute(Context())
    statent = pd.read_csv(
        Path(args.data_path) / "statent/250221_STATENT_2022_LOC_17042025.csv",
        encoding="latin1", sep=",",
        usecols=["NOGA08_CD", "ANONYM_ENT_ID", "ANONYM_LOCAL_ID", "EMPTOT", "EMPFTE"],
        dtype={"NOGA08_CD": str, "ANONYM_ENT_ID": str, "ANONYM_LOCAL_ID": str},
    ).rename(columns={
        "NOGA08_CD": "noga", "ANONYM_ENT_ID": "statent_enterprise_id",
        "ANONYM_LOCAL_ID": "statent_local_id", "EMPTOT": "number_employees",
        "EMPFTE": "number_employees_fte",
    })
    print(audit(survey, statent, args.output_path).to_string(index=False))


if __name__ == "__main__":
    main()
