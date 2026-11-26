from pathlib import Path

import pandas as pd


def configure(context):
    context.config("data_path")
    context.stage("synthesis.population.sampled")


def execute(context):

    # Identify private-household residents in the sampled STATPOP population.

    persons = context.stage("synthesis.population.sampled")
    cols = ["person_id", "statpop_person_id", "home_municipality_id", "is_synthetic_household"]
    if not set(cols).issubset(persons.columns):
        raise ValueError("Workplace survey comparisons require a STATPOP population")
    sample = persons[cols].copy()
    sample["home_municipality_id"] = pd.to_numeric(sample["home_municipality_id"])
    ids = pd.Index(sample["statpop_person_id"].unique())

    # Read residence and permit flags for the source register records.

    directory = Path(context.config("data_path")) / "statpop"
    fields = ["personPseudoID", "TYPEOFRESIDENCE", "RESIDENTPERMIT", "POPULATIONTYPE",
              "REPORTINGMUNICIPALITYID"]
    matches = []
    for part in (1, 2):
        path = directory / f"STATPOP_PP_2023_TEIL_{part}.csv"
        for rows in pd.read_csv(path, sep=";", encoding="latin1", usecols=fields, chunksize=400_000):
            matches.append(rows[rows.personPseudoID.isin(ids) & rows.POPULATIONTYPE.eq(1)])
    register = pd.concat(matches, ignore_index=True)
    joined = sample.merge(register, left_on="statpop_person_id", right_on="personPseudoID", how="left")
    if joined.TYPEOFRESIDENCE.isna().any():
        raise ValueError("Missing STATPOP residence record for workplace survey comparison")
    joined["municipality_match"] = joined.REPORTINGMUNICIPALITYID.eq(joined.home_municipality_id)
    joined["main"] = joined.TYPEOFRESIDENCE.eq(1)
    joined = joined.sort_values(["municipality_match", "main"], ascending=False, kind="stable")
    joined = joined.drop_duplicates("person_id").set_index("person_id")
    return joined.main & ~joined.is_synthetic_household & ~joined.RESIDENTPERMIT.isin([11, 12])
