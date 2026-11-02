"""Convert a frozen official BFS cube to the national reference used by education.

This helper has no pipeline stage: reference updates are an explicit maintenance
operation. Totals on the other axes avoid adding totals to their own components.
"""

from itertools import product

import numpy as np
import pandas as pd


TABLE = "px-x-1502030000_112"
API_URL = f"https://www.pxweb.bfs.admin.ch/api/v1/de/{TABLE}/{TABLE}.px"
SOURCE_URL = f"https://www.pxweb.bfs.admin.ch/pxweb/de/{TABLE}/-/{TABLE}.px/"
TYPE_LABELS = {
    "0": "Bildungstyp - Total",
    "310": "Höhere Fachschulen (Vorgänger Fachhochschulen)",
    "311": "Höhere Fachschulen",
    "312": "Vorbereitung auf Berufsprüfung",
    "313": "Vorbereitung auf höhere Fachprüfung",
    "314": "Unterrichtsberufe",
    "315": "Nicht BBG-reglementierte höhere Berufsbildung",
}
TOTAL_LABELS = {"Ausbildungsfeld": "Ausbildungsfeld - Total",
                "Geschlecht": "Geschlecht - Total",
                "Staatsangehörigkeit (Kategorie)": "Staatsangehörigkeit - Total"}
GROUPS = {"higher_vocational_college": ["311"],
          "higher_professional_examination": ["312", "313"],
          "higher_vocational_non_bbg_regulated": ["315"]}


def build_query(metadata, academic_years):
    """Fail on changed definitions instead of silently changing reference coverage."""
    variables = {v["code"]: v for v in metadata["variables"]}
    expected = set(TOTAL_LABELS) | {"Bildungstyp", "Jahr"}
    if len(variables) != len(metadata["variables"]) or set(variables) != expected:
        raise ValueError("Unexpected higher-vocational BFS dimensions; review the table")
    types = dict(zip(variables["Bildungstyp"]["values"], variables["Bildungstyp"]["valueTexts"]))
    if types != TYPE_LABELS:
        raise ValueError("Higher-vocational education types changed; review the mapping")
    years = dict(zip(variables["Jahr"]["valueTexts"], variables["Jahr"]["values"]))
    if not academic_years or len(set(academic_years)) != len(academic_years) or not set(academic_years) <= set(years):
        raise ValueError("Requested academic years must be unique and available in BFS metadata")
    query = []
    for name, variable in variables.items():
        if name == "Jahr":
            values = [years[y] for y in academic_years]
        elif name == "Bildungstyp":
            values = list(TYPE_LABELS)
        else:
            labels = dict(zip(variable["values"], variable["valueTexts"]))
            if labels.get("0") != TOTAL_LABELS[name]:
                raise ValueError("Higher-vocational total category changed: " + name)
            values = ["0"]
        query.append({"code": name, "selection": {"filter": "item", "values": values}})
    return {"query": query, "response": {"format": "json-stat2"}}


def parse_reference(dataset, metadata, academic_years):
    """Reconcile three model groups to the published total for every requested year.

    Historic types 310/314 have no current model mapping. They must be explicitly
    zero; a new positive or suppressed count requires a coverage decision.
    Missing/suppressed cells never become zero. Source-axis order is immaterial.
    """
    query = build_query(metadata, academic_years)
    wanted = {v["code"]: v["selection"]["values"] for v in query["query"]}
    dimensions, sizes = dataset["id"], dataset["size"]
    if dataset.get("class") != "dataset" or len(set(dimensions)) != len(dimensions) or set(dimensions) != set(wanted) or len(sizes) != len(dimensions):
        raise ValueError("Unexpected higher-vocational response dimensions")
    axes = []
    labels = {v["code"]: dict(zip(v["values"], v["valueTexts"])) for v in metadata["variables"]}
    for name, size in zip(dimensions, sizes):
        category = dataset["dimension"][name]["category"]
        index = category["index"]
        codes = sorted(index, key=index.get) if isinstance(index, dict) else list(index)
        if (len(codes) != size or len(set(codes)) != size or set(codes) != set(wanted[name])
                or any(category["label"].get(c) != labels[name][c] for c in codes)):
            raise ValueError("Higher-vocational response omits or changes requested categories: " + name)
        if isinstance(index, dict) and set(index.values()) != set(range(size)):
            raise ValueError("Invalid BFS category index")
        axes.append(codes)
    count = int(np.prod(sizes))
    values = dataset["value"]
    if isinstance(values, dict):
        if not set(values) <= {str(i) for i in range(count)}:
            raise ValueError("Invalid sparse BFS value indices")
        values = [values.get(str(i)) for i in range(count)]
    if len(values) != count:
        raise ValueError("BFS cube size does not match its values")
    frame = pd.DataFrame(product(*axes), columns=dimensions)
    frame["students"] = pd.to_numeric(pd.Series(values), errors="raise")
    if (not np.isfinite(frame.students).all() or frame.students.lt(0).any()
            or frame.students.ne(frame.students.round()).any()):
        raise ValueError("Higher-vocational reference requires complete nonnegative integer counts")
    frame["academic_year"] = frame.Jahr.map(labels["Jahr"])
    rows, checks = [], []
    for year in academic_years:
        source = frame.loc[frame.academic_year.eq(year)].set_index("Bildungstyp").students
        if source.loc[["310", "314"]].sum() != 0:
            raise ValueError("Positive historic education types need an explicit model mapping: " + year)
        if source.drop("0").sum() != source.loc["0"]:
            raise ValueError("Higher-vocational component counts do not conserve the BFS total")
        for level, codes in GROUPS.items():
            note = "National enrolment; not institution-level enrolment"
            if level == "higher_professional_examination":
                note += "; preparation-course coverage incomplete; not examination awards"
            rows.append(dict(academic_year=year, study_level=level,
                             students=int(source.loc[codes].sum()), source_url=SOURCE_URL,
                             coverage_note=note))
        checks.append(dict(academic_year=year, published_students=int(source.loc["0"]),
                           mapped_students=int(source.loc[[c for codes in GROUPS.values() for c in codes]].sum()),
                           unmapped_historic_students=0))
    return pd.DataFrame(rows), pd.DataFrame(checks)
