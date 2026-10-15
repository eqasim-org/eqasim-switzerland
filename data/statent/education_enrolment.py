"""BFS institution enrolment and reviewed links to anonymous STATENT groups."""

import json
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


TABLES = {"UH": "px-x-1502040100_108", "FH_PH": "px-x-1502040400_136"}
# Relative input paths are resolved under data_path; references stay outside Git.
DEFAULT_MAPPING = Path("education_enrolment/education_institution_links.csv")
DEFAULT_HIGHER_VOCATIONAL = Path("education_enrolment/education_higher_vocational_enrolment.csv")
LEVELS = {
    "Bachelor": "bachelor", "Master": "master", "Doktorat": "doctorate_habilitation",
    "Lizenziat/Diplom": "legacy_degree", "Diplom": "legacy_degree",
    "Weiterbildung, Vertiefung und Andere": "continuing_education_and_other",
    "Weiterbildung": "continuing_education_and_other",
}
LINK_FIELDS = ["institution_type", "bfs_institution_code", "statent_enterprise_id",
               "match_status", "institution_name", "evidence_url", "evidence_note"]


def configure(context):
    context.config("data_path")
    context.config("education_enrolment_path", None)
    context.config("education_enrolment_year", "2023/24")
    context.config("education_institution_links", str(Path(context.config("data_path")) / DEFAULT_MAPPING))
    context.config("education_higher_vocational_reference", str(Path(context.config("data_path")) / DEFAULT_HIGHER_VOCATIONAL))
    context.stage("data.statent.education_institutions")


def validate(context):
    """Invalidate cached enrolment links when a reviewed CSV or snapshot changes.

    A mapping's path can remain unchanged while ETH/UZH links are added. Synpp
    must then rebuild the dependent campus targets instead of reusing old links.
    """
    path = context.config("education_enrolment_path")
    path = Path(path) if path is not None else Path(context.config("data_path")) / "education_enrolment"
    files = [Path(context.config("education_institution_links")), *sorted(path.glob("*.json"))]
    files.append(Path(context.config("education_higher_vocational_reference")))
    # Synpp stores tokens in JSON: a dict survives that round-trip unchanged,
    # whereas tuples would become lists and invalidate the cache on every run.
    return {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def table_url(institution_type, api=False):
    table = TABLES[institution_type]
    root = "api/v1" if api else "pxweb"
    return "https://www.pxweb.bfs.admin.ch/%s/de/%s/%s.px" % (root, table, table)


def build_query(metadata, academic_years):
    """Request complete category coverage so missing totals are not mistaken for zero."""
    query = []
    for variable in metadata["variables"]:
        values = variable["values"]
        if variable["code"] == "Jahr":
            labels = dict(zip(variable["valueTexts"], values))
            if not set(academic_years).issubset(labels):
                raise ValueError("Requested academic year is absent from BFS metadata")
            values = [labels[year] for year in academic_years]
        query.append({"code": variable["code"],
                      "selection": {"filter": "item", "values": values}})
    return {"query": query, "response": {"format": "json-stat2"}}


def parse_enrolment(dataset, institution_type):
    """Sum exhaustive, mutually exclusive source categories; keep unknowns unknown."""
    dimensions = dataset["id"]
    size = dataset["size"]
    expected = {"Jahr", "Hochschule", "Studienstufe", "Fachbereich",
                "Bildungsherkunft" if institution_type == "UH" else "Geschlecht"}
    if set(dimensions) != expected:
        raise ValueError("Unexpected BFS enrolment dimensions")
    categories = []
    for dimension, length in zip(dimensions, size):
        category = dataset["dimension"][dimension]["category"]
        index = category["index"]
        codes = sorted(index, key=index.get) if isinstance(index, dict) else index
        if len(codes) != length:
            raise ValueError("BFS category count does not match cube dimensions")
        categories.append(codes)
    count = int(np.prod(size))
    values = dataset["value"]
    if isinstance(values, dict):
        values = [values.get(str(i)) for i in range(count)]
    if len(values) != count:
        raise ValueError("BFS value count does not match cube dimensions")
    frame = pd.DataFrame(product(*categories), columns=dimensions)
    frame["students"] = pd.to_numeric(pd.Series(values), errors="raise")
    if (frame["students"].dropna() < 0).any():
        raise ValueError("BFS enrolment counts must be non-negative")
    for dimension in ["Jahr", "Hochschule", "Studienstufe"]:
        frame[dimension + "_label"] = frame[dimension].map(
            dataset["dimension"][dimension]["category"]["label"])
    keys = ["Jahr_label", "Hochschule", "Hochschule_label", "Studienstufe", "Studienstufe_label"]
    grouped = frame.groupby(keys, dropna=False)["students"]
    totals = grouped.sum(min_count=1).rename("students").to_frame()
    totals["source_cells"] = grouped.size()
    totals["missing_source_cells"] = grouped.apply(lambda x: x.isna().sum())
    totals.loc[totals["missing_source_cells"].gt(0), "students"] = np.nan
    totals = totals.reset_index().rename(columns={
        "Jahr_label": "academic_year", "Hochschule": "bfs_institution_code",
        "Hochschule_label": "bfs_institution_label", "Studienstufe": "bfs_study_level_code",
        "Studienstufe_label": "bfs_study_level_label"})
    totals["study_level"] = totals["bfs_study_level_label"].map(LEVELS)
    if totals["study_level"].isna().any():
        raise ValueError("Unrecognised BFS study level; review its definition before importing")
    totals["institution_type"] = institution_type
    totals["bfs_table_id"] = TABLES[institution_type]
    totals["source_url"] = table_url(institution_type)
    totals["aggregate_provider"] = totals["bfs_institution_label"].str.startswith("Andere ")
    totals["model_study_level"] = totals["study_level"].isin(
        ["bachelor", "master", "doctorate_habilitation"])
    # Educational origin is not current residence; no domestic-population filtering here.
    return totals


def read_enrolment(input_path):
    """Read frozen BFS responses with metadata, preserving the source's missing values."""
    tables = []
    for institution_type, table in TABLES.items():
        with (Path(input_path) / (table + ".json")).open() as stream:
            dataset = json.load(stream)
        with (Path(input_path) / (table + ".metadata.json")).open() as stream:
            metadata = json.load(stream)
        # A subset of subjects, sex or educational origin would silently undercount.
        for variable in metadata["variables"]:
            if variable["code"] == "Jahr":
                continue
            category = dataset["dimension"][variable["code"]]["category"]
            if set(category["index"]) != set(variable["values"]):
                raise ValueError("BFS snapshot omits categories needed for complete totals")
        tables.append(parse_enrolment(dataset, institution_type))
    return pd.concat(tables, ignore_index=True)


def model_enrolment_reference(enrolment):
    """Align BFS degrees with structural-survey ongoing-education codes 11–13.

    The supplied 2023 codebook includes FH/PH diplomas in Bachelor (11), and
    university/ETH licentiates and diplomas in Master (12). Preserve the raw
    categories separately and aggregate before linking or splitting campuses,
    so an institution with both degree categories receives its total only once.
    Continuing education has no established correspondence and stays excluded.
    """
    source = enrolment.copy()
    keys = ["academic_year", "institution_type", "bfs_institution_code"]
    if source.duplicated(keys + ["study_level"]).any():
        raise ValueError("Duplicate raw institution/study-level enrolment")
    legacy = source["study_level"].eq("legacy_degree")
    if not source.loc[legacy, "institution_type"].isin(["UH", "FH_PH"]).all():
        raise ValueError("Legacy diploma reference requires a known institution type")
    source["strict_degree_students"] = source["students"].where(~legacy, 0.)
    source["legacy_degree_students"] = source["students"].where(legacy, 0.)
    source.loc[legacy & source["institution_type"].eq("FH_PH"), "study_level"] = "bachelor"
    source.loc[legacy & source["institution_type"].eq("UH"), "study_level"] = "master"
    source = source.loc[source["study_level"].isin(["bachelor", "master", "doctorate_habilitation"])]
    keys += ["bfs_institution_label", "aggregate_provider", "study_level", "bfs_table_id", "source_url"]
    # Unknown source cells must not become a smaller, seemingly complete total.
    complete_sum = lambda values: values.sum() if values.notna().all() else np.nan
    result = source.groupby(keys, dropna=False, as_index=False).agg(
        students=("students", complete_sum),
        strict_degree_students=("strict_degree_students", complete_sum),
        legacy_degree_students=("legacy_degree_students", complete_sum),
        source_cells=("source_cells", "sum"),
        missing_source_cells=("missing_source_cells", "sum"),
        bfs_source_study_level_codes=("bfs_study_level_code", lambda values: ";".join(values.astype(str))),
        bfs_source_study_level_labels=("bfs_study_level_label", lambda values: ";".join(values.astype(str))),
    )
    result["model_study_level"] = True
    result["reference_definition"] = "structural_survey_2023_codes_11_12_13"
    return result


def link_institutions(enrolment, institutions, links, academic_year):
    """Expose proposed links, but only verified links can carry model-level totals."""
    selected = enrolment[enrolment["academic_year"].eq(academic_year)].copy()
    if selected.empty:
        raise ValueError("Requested academic year is not present in the enrolment snapshots")
    if not set(LINK_FIELDS).issubset(links.columns):
        raise ValueError("Institution mapping is missing required fields")
    links = links[LINK_FIELDS].copy()
    keys = ["institution_type", "bfs_institution_code"]
    if links.duplicated(keys + ["statent_enterprise_id"]).any():
        raise ValueError("Duplicate BFS/STATENT institution link")
    if not links["match_status"].isin(["proposed", "verified"]).all():
        raise ValueError("Institution links must be proposed or verified")
    source_keys = enrolment[keys].drop_duplicates()
    if len(links.merge(source_keys, on=keys, how="inner")) != len(links):
        raise ValueError("Institution link references an unknown BFS institution")
    if not links["statent_enterprise_id"].isin(institutions["statent_enterprise_id"]).all():
        raise ValueError("Institution link references an unknown STATENT enterprise")
    source_types = links.merge(institutions[["statent_enterprise_id", "institution_types"]],
                               on="statent_enterprise_id", validate="many_to_one")
    if any(row.institution_type not in row.institution_types.split(";")
           for row in source_types.itertuples()):
        raise ValueError("BFS/STATENT institution types are incompatible")
    verified = links[links["match_status"].eq("verified")]
    if verified.duplicated("statent_enterprise_id").any():
        raise ValueError("A STATENT group cannot have multiple verified BFS institutions")
    if verified.duplicated(keys).any():
        raise ValueError("Multiple STATENT groups require an explicit allocation rule before verification")
    if (verified[["institution_name", "evidence_url", "evidence_note"]].isna().any().any()
            or verified[["institution_name", "evidence_url", "evidence_note"]].eq("").any().any()):
        raise ValueError("Verified institution links require a name and evidence")
    aggregates = selected.loc[selected["aggregate_provider"], keys].drop_duplicates()
    if not verified.merge(aggregates, on=keys).empty:
        raise ValueError("An aggregate BFS provider cannot be verified as one institution")

    linked = selected.merge(links, on=keys, how="left", validate="many_to_many")
    linked = linked.merge(institutions, on="statent_enterprise_id", how="left",
                          suffixes=("", "_statent"), validate="many_to_one")
    linked["match_status"] = linked["match_status"].fillna("unmatched")
    linked["verified_enrolled_students"] = linked["students"].where(
        linked["match_status"].eq("verified") & linked["model_study_level"])
    model_reference = model_enrolment_reference(enrolment)
    selected_model = model_reference.loc[model_reference["academic_year"].eq(academic_year)]
    linked_model = selected_model.merge(links, on=keys, how="left", validate="many_to_many")
    linked_model = linked_model.merge(institutions, on="statent_enterprise_id", how="left",
                                      suffixes=("", "_statent"), validate="many_to_one")
    linked_model["match_status"] = linked_model["match_status"].fillna("unmatched")
    linked_model["verified_enrolled_students"] = linked_model["students"].where(
        linked_model["match_status"].eq("verified"))
    # Enrolment remains distinct from commuter targets and campus capacity.
    institution_totals = selected.groupby(keys + ["bfs_institution_label", "aggregate_provider"]).agg(
        enrolled_students=("students", lambda x: x.sum() if x.notna().all() else np.nan),
        missing_source_cells=("missing_source_cells", "sum"),
    ).reset_index()
    model_totals = selected[selected["model_study_level"]].groupby(keys)["students"].agg(
        lambda x: x.sum() if x.notna().all() else np.nan).rename("strict_degree_enrolled_students")
    aligned_totals = selected_model.groupby(keys)["students"].agg(
        lambda x: x.sum() if x.notna().all() else np.nan).rename("model_level_enrolled_students")
    institution_totals = institution_totals.merge(model_totals, on=keys, how="left").merge(
        aligned_totals, on=keys, how="left")
    status = links.groupby(keys).agg(candidate_groups=("statent_enterprise_id", "size"),
                                    verified_groups=("match_status", lambda x: x.eq("verified").sum()))
    institution_totals = institution_totals.merge(status, on=keys, how="left")
    institution_totals[["candidate_groups", "verified_groups"]] = institution_totals[
        ["candidate_groups", "verified_groups"]].fillna(0).astype(int)
    institution_totals["academic_year"] = academic_year
    groups = institutions.merge(links, on="statent_enterprise_id", how="left", suffixes=("", "_link"))
    groups["match_status"] = groups["match_status"].fillna("unmatched")
    return {"enrolment": enrolment, "linked_enrolment": linked,
            "model_enrolment": model_reference, "linked_model_enrolment": linked_model,
            "institution_totals": institution_totals, "statent_links": groups}


def pooled_enrolment_reference(enrolment, higher_vocational):
    """Pool disjoint BFS programme counts once per institution, retaining all types.

    Includes the published continuing/other university category. Its coverage is
    not identical to the SS student flag. Higher vocational evidence supplies a
    national type share only: it cannot identify individual schools or campuses.
    """
    keys = ["academic_year", "institution_type", "bfs_institution_code"]
    if enrolment.duplicated(keys + ["study_level"]).any():
        raise ValueError("Duplicate programme counts would inflate pooled enrolment")
    pooled = enrolment.groupby(keys, dropna=False, as_index=False).agg(
        students=("students", lambda x: x.sum() if x.notna().all() else np.nan),
        bfs_institution_label=("bfs_institution_label", "first"),
        source_url=("source_url", "first"))
    if higher_vocational.duplicated(["academic_year", "study_level"]).any():
        raise ValueError("Duplicate national higher-vocational reference")
    higher = higher_vocational.groupby("academic_year", as_index=False).agg(
        students=("students", lambda x: x.sum() if x.notna().all() else np.nan),
        source_url=("source_url", "first"))
    higher["institution_type"] = "higher_vocational"
    higher["bfs_institution_code"] = "national_higher_vocational"
    higher["bfs_institution_label"] = "Higher vocational education — national total"
    pooled = pd.concat([pooled, higher], ignore_index=True)
    if not np.isfinite(pooled.students).all() or pooled.students.lt(0).any():
        raise ValueError("Pooled BFS reference contains missing or invalid counts")
    pooled["study_level"] = "tertiary"
    pooled["reference_definition"] = "all_published_programmes_including_continuing_education"
    return pooled


def execute(context):
    input_path = context.config("education_enrolment_path")
    if input_path is None:
        input_path = Path(context.config("data_path")) / "education_enrolment"
    links = pd.read_csv(context.config("education_institution_links"), dtype=str)
    registry = context.stage("data.statent.education_institutions")
    result = link_institutions(read_enrolment(input_path), registry["institutions"], links,
                               context.config("education_enrolment_year"))
    higher = pd.read_csv(context.config("education_higher_vocational_reference"))
    result["pooled_enrolment"] = pooled_enrolment_reference(result["enrolment"], higher)
    return result
