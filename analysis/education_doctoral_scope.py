"""Compare doctoral reference coverage without inventing a survey study field."""
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.education_student_stock import survey_stock


# BFS's employment publication excludes these four fields, but includes pharmacy.
# These are programme fields. Occupation/employer evidence below is NOT the same
# classification and must not be used to delete or relabel doctoral respondents.
MEDICAL_FIELD_PREFIXES = ("5.1 ", "5.2 ", "5.3 ", "5.5 ")
CLINICAL_OCCUPATIONS = {2210, 2211, 2212, 2250, 2261}


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.config("data_path")
    context.config("education_enrolment_path", None)
    context.config("analysis_path")


def snapshot_path(context):
    root = context.config("education_enrolment_path")
    root = Path(root) if root is not None else Path(context.config("data_path"))/"education_enrolment"
    return root/"px-x-1502040100_108.json"


def validate(context):
    return hashlib.sha256(snapshot_path(context).read_bytes()).hexdigest()


def doctoral_field_totals(dataset):
    """Decode the frozen BFS cube with its declared axis order and missing cells."""
    dimensions = dataset["id"]
    if set(dimensions) != {"Jahr","Hochschule","Studienstufe","Fachbereich","Bildungsherkunft"}:
        raise ValueError("Unexpected BFS doctoral enrolment dimensions")
    categories = []
    for dimension, length in zip(dimensions, dataset["size"]):
        index = dataset["dimension"][dimension]["category"]["index"]
        codes = sorted(index,key=index.get) if isinstance(index,dict) else index
        if len(codes) != length:
            raise ValueError("BFS doctoral cube category count is inconsistent")
        categories.append(codes)
    values = dataset["value"]
    size = int(np.prod(dataset["size"]))
    if isinstance(values,dict):
        values = [values.get(str(i)) for i in range(size)]
    if len(values) != size:
        raise ValueError("BFS doctoral cube value count is inconsistent")
    frame = pd.DataFrame(product(*categories),columns=dimensions)
    frame["students"] = pd.to_numeric(pd.Series(values),errors="raise")
    if frame.students.dropna().lt(0).any():
        raise ValueError("Negative BFS doctoral enrolment")
    for dimension in dimensions:
        frame[dimension+"_label"] = frame[dimension].map(dataset["dimension"][dimension]["category"]["label"])
    frame = frame.loc[frame.Studienstufe_label.eq("Doktorat")].copy()
    if frame.empty:
        raise ValueError("No Doktorat category in BFS snapshot")
    frame["field_scope"] = np.where(frame.Fachbereich_label.str.startswith(MEDICAL_FIELD_PREFIXES),
        "medical_fields_excluded_in_employment_publication","other_fields_including_pharmacy")
    rows = []
    for (year,scope), group in frame.groupby(["Jahr_label","field_scope"],sort=True):
        missing = int(group.students.isna().sum())
        rows.append(dict(academic_year=year,field_scope=scope,
                         students=float(group.students.sum()) if not missing else np.nan,
                         missing_cells=missing))
    return pd.DataFrame(rows)


def doctoral_evidence(survey):
    """Keep medical job/sector evidence distinct from an observed study field.

    A physician may pursue a doctorate outside medicine, and a hospital may
    employ a researcher in another discipline. Missing/aggregate occupation
    codes and university-sector codes do not establish a non-medical programme.
    """
    reports = survey.loc[survey.current_education.eq(13)].copy()
    reports, _ = survey_stock(reports)
    occupation = pd.to_numeric(reports.isco_code,errors="coerce")
    reports["clinical_occupation_evidence"] = occupation.isin(CLINICAL_OCCUPATIONS)
    for prefix, columns in [("work",["work_noga","work_enterprise_noga"]),
                            ("school",["education_noga","education_enterprise_noga"])]:
        evidence = pd.Series(False,index=reports.index)
        for column in columns:
            code = reports[column].astype("string")
            # Veterinary services are outside NOGA 86 but relevant to the
            # medical fields omitted in the BFS employment publication.
            evidence |= code.str.startswith("86").fillna(False) | code.eq("750000").fillna(False)
        reports[prefix+"_medical_sector_evidence"] = evidence
    reports["medical_job_or_sector_evidence"] = (reports.clinical_occupation_evidence
        | reports.work_medical_sector_evidence | reports.school_medical_sector_evidence)
    reports["evidence_group"] = np.select(
        [reports.clinical_occupation_evidence, reports.medical_job_or_sector_evidence],
        ["clinical_occupation_evidence","medical_sector_evidence_only"],
        default="no_specific_medical_evidence")
    reports["doctoral_study_field"] = "unavailable_in_survey_extract"
    reports["already_completed_doctorate"] = reports.highest_completed_education.eq(13)
    return reports


def write_audit(survey,dataset,output_path):
    path = Path(output_path)
    path.mkdir(parents=True,exist_ok=True)
    fields = doctoral_field_totals(dataset)
    fields.to_csv(path/"bfs_doctoral_field_scope.csv",index=False)
    reports = doctoral_evidence(survey)
    # Do not sum annual survey weights across years. Unknown destinations are
    # explicit stock evidence, not automatically domestic campus travellers.
    totals = reports.groupby(["survey_year","education_destination_status","is_student"],dropna=False).agg(
        observations=("weight","size"),weighted_persons=("weight","sum")).reset_index()
    totals.to_csv(path/"survey_doctoral_stock_scope.csv",index=False)
    reports.groupby(["survey_year","education_destination_status","is_student","evidence_group","employed"],
                    dropna=False).agg(observations=("weight","size"),weighted_persons=("weight","sum")).reset_index().to_csv(
        path/"survey_doctoral_medical_evidence.csv",index=False)
    reports.groupby(["survey_year","education_destination_status","is_student","evidence_group",
                     "already_completed_doctorate","commute_to_education","education_freq_per_week","freq_per_week"],
                    dropna=False).agg(observations=("weight","size"),weighted_persons=("weight","sum")).reset_index().to_csv(
        path/"survey_doctoral_commuting_reports.csv",index=False)
    (path/"README.md").write_text(
        "# Doctoral reference scope and medical evidence\n\n"
        "No student status, study level, employment or campus assignment is changed. BFS doctoral "
        "programme fields are split using the exclusions in its 2022 university-employment publication: "
        "human medicine, dentistry, veterinary medicine and interdisciplinary medical fields. Pharmacy "
        "remains included. All fields remain in the national enrolment reference.\n\n"
        "The supplied structural-survey extract has no current doctoral subject-field variable. "
        "CH-ISCO-19 physician/dentist/veterinarian occupations and NOGA healthcare/veterinary workplaces "
        "or school establishments provide inspection evidence, not programme classifications. "
        "No specific medical evidence does not mean non-medical doctoral study. Respondents with "
        "unknown work codes, or those not employed, remain visible. All study fields are therefore "
        "labelled unavailable in the survey; no medical programme is imputed.\n\n"
        "The structural survey covers permanent residents aged 15+ in private households. BFS counts "
        "enrolments at Swiss institutions. Current-residence coverage cannot be inferred from educational "
        "origin. Annual source weights are reported per survey year. Foreign school destinations and "
        "unknown destinations are kept separate. Explicit ongoing doctorates with a zero student flag "
        "are retained for review. A missing school commute is not grounds to delete an enrolled student. "
        "Commuting fields are exported as reported codes/frequencies, including missing codes; no new "
        "daily attendance rate is inferred. Already holding a doctorate is an inspection flag, not "
        "proof that the current programme is a habilitation or an invalid second doctorate.\n\n"
        "Recommended interpretation: preserve both source references and model predictions while "
        "reconciling population frame and programme definitions. Do not remove medical fields from "
        "the BFS target solely because a national total then resembles the survey total. Do not promote "
        "healthcare workers to doctoral students solely from their occupation.\n\n"
        "Sources: https://www.bfs.admin.ch/bfs/en/home.assetdetail.31186755.html ; "
        "https://www.swissstats.bfs.admin.ch/data/webviewer/appId/ch.admin.bfs.swissstat/article/issue241523292200-01/package ; "
        "https://www.census1850.bfs.admin.ch/de/methodologie-daten.html ; supplied 2023 codebook sheets 25,30,57.\n")
    return {"doctoral_reports":len(reports),"bfs_scope_rows":len(fields),"output_path":str(path)}


def execute(context):
    with snapshot_path(context).open() as stream:
        dataset = json.load(stream)
    return write_audit(context.stage("data.structural_survey.structural_survey"),dataset,
                      Path(context.config("analysis_path"))/"education_doctoral_scope")
