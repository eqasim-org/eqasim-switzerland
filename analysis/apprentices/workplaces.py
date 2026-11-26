"""Project the 2023 apprentice stock and check reported 2024 workplace geography."""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.apprentices import projection
from data.structural_survey import apprentice_references
from data.structural_survey.apprenticeship import DEFAULT_REPAIR_FRACTION
from data.structural_survey.structural_survey import resolve_work_start

CITIES = [261, 6621, 5586]
SCOPE = "synthesis.population.spatial.primary.work.survey_scope"


def configure(context):
    context.config("data_path")
    context.config("workplace_validation_survey_file", default=None)
    context.config("vocational_apprentice_repair_fraction", default=DEFAULT_REPAIR_FRACTION)
    for name in ("data.spatial.municipalities", "data.spatial.quarters", "data.statent.statent",
                 "data.structural_survey.structural_survey", "synthesis.population.sampled", SCOPE):
        context.stage(name)


def execute(context):

    # Read the training diaries and both reference years in the same municipal geography.

    municipalities, mapping = context.stage("data.spatial.municipalities")
    quarters = context.stage("data.spatial.quarters")
    surveys = {year: read_survey(path, year, mapping, quarters) for year, path in survey_files(context).items()}
    persons = context.stage("synthesis.population.sampled")
    centres = persons.groupby(persons.home_municipality_id.astype(int))[["home_x", "home_y"]].median()
    centroids = municipalities.set_index("municipality_id").geometry.centroid
    centres = centres.combine_first(pd.DataFrame({"home_x": centroids.x, "home_y": centroids.y})) / 1000
    firms = context.stage("data.statent.statent")
    cantons = municipality_cantons(firms)
    training = pd.concat([surveys[2021], surveys[2022]], ignore_index=True)
    known = training.explicit & training.firm.isin(cantons.index) & training.home.isin(centres.index)
    donors = projection.Donors(training[known & training.country.isin([8100, -8, -9])], centres, cantons)
    sources = apprentice_references.verify_sources(context.config("data_path"))
    register = apprentice_references.employer_stock(sources / apprentice_references.EMPLOYER)
    regional_stock, share_under_25 = apprentice_references.residence_controls(sources)

    # Compare the production repair with the administrative benchmark for each survey year.

    folder = Path(context.path())
    repair_file = Path(context.path("data.structural_survey.structural_survey")) / "apprentice_repair.csv"
    audit = pd.read_csv(repair_file)
    audit["dual_training_places"] = audit.survey_year.astype(str).map(register.sum())
    gap = audit.dual_training_places - audit.raw_apprentices
    audit["implied_fraction"] = gap / audit.eligible_weight.replace(0, np.nan)
    audit.to_csv(folder / "national_apprentice_benchmark.csv", index=False)

    # Retain every reported workplace and estimate the missing apprentice contribution.

    cleaned = context.stage("data.structural_survey.structural_survey")
    repaired = cleaned.survey_year.eq(2023) & cleaned.job_position.eq(50) & cleaned.start_work.eq(-8)
    repaired_weight = cleaned.loc[repaired, "weight"].sum()
    references = []
    for year in (2023, 2024):
        survey = surveys[year]
        pool = survey.loc[survey.eligible]
        mass = repaired_weight if year == 2023 else pool.w.sum() * context.config("vocational_apprentice_repair_fraction")
        choices = donors.choices(pool)
        explicit = survey.loc[survey.explicit]
        reported = explicit.groupby(explicit.firm.map(cantons)).w.sum().reindex(register.index, fill_value=0)
        capacity = register[str(year)] * ((mass + reported.sum()) / register[str(year)].sum())
        # Keep observed reports even if one small canton already exceeds its register share.
        residual = (capacity - reported).clip(lower=.0001)
        if mass:
            target = (residual * (mass / residual.sum())).to_numpy()
            probabilities = projection.repair_projection(choices, explicit, target, regional_stock, share_under_25)
        else:
            probabilities = np.zeros_like(choices.prior)
        selected = probabilities.sum(axis=1)
        if (selected > 1 + 1e-8).any() or abs(selected @ pool.w - mass) >= 1:
            raise ValueError("Projected survey repairs do not conserve the national repair stock")
        estimated = pool.assign(w=pool.w * selected)
        rates = pool.groupby("home_ct").w.sum().to_frame("eligible")
        rates["estimated_apprentices"] = estimated.groupby("home_ct").w.sum()
        rates["fraction"] = rates.estimated_apprentices / rates.eligible
        rates.to_csv(folder / f"repair_rates_{year}.csv")
        references.append(reference_table(year, survey, choices, probabilities))
        if year == 2023:
            resident_reference = pd.concat([explicit, estimated])

    # Project the private-household apprentice cohort onto the employer register.

    scope = persons.person_id.map(context.stage(SCOPE))
    if scope.isna().any():
        raise ValueError("Missing survey eligibility for sampled residents")
    apprentices = persons.loc[scope & persons.job_position.eq(50), ["home_municipality_id", "age"]].copy()
    apprentices["home"] = apprentices.home_municipality_id.astype(int)
    apprentices["home_ct"] = apprentices.home.map(cantons)
    target = register["2023"].to_numpy()
    apprentices["w"] = projected_weights(apprentices, resident_reference, target.sum())
    choices = donors.choices(apprentices)
    probabilities = choices.fit(target, local=True)
    stock = choices.stock(probabilities)
    projected = np.bincount(choices.canton.ravel(), weights=(apprentices.w.to_numpy()[:, None] * probabilities).ravel(), minlength=26)
    if np.max(np.abs(projected - target)) >= 1:
        raise ValueError("Projected workplaces disagree with the employer register")
    register[["2023", "2024"]].assign(projected_2023=projected).to_csv(folder / "employer_cantons.csv")

    # Compare against explicitly reported destinations in the held-out year.

    survey = surveys[2024]
    held = survey[survey.explicit & survey.home.isin(cantons.index) & survey.firm.isin(cantons.index)]
    predictions = donors.choices(held)
    held_stock = predictions.stock(predictions.fit(target * (held.w.sum() / target.sum()), local=True))
    observed = held.groupby("firm").w
    table = observed.sum().reindex(CITIES, fill_value=0).to_frame("ss_2024")
    table["respondents"] = observed.size().reindex(CITIES, fill_value=0)
    table["approx_95_half"] = observed.apply(lambda w: 1.96 * np.linalg.norm(w)).reindex(CITIES, fill_value=0)
    table["predicted"] = held_stock.reindex(CITIES, fill_value=0)
    table.to_csv(folder / "heldout_workplaces.csv")
    commute_check(held, choices, probabilities).to_csv(folder / "heldout_commutes.csv", index=False)
    result = pd.concat(references, ignore_index=True).set_index(["year", "municipality"])
    municipalities = result.index.get_level_values("municipality").union(stock.index[stock.gt(0)]).unique()
    result = result.reindex(pd.MultiIndex.from_product([(2023, 2024), municipalities], names=result.index.names))
    columns = ["recorded", "respondents", "imputed", "reference"]
    result[columns] = result[columns].fillna(0)
    result = result.reset_index()
    result["projected_apprentices"] = result.municipality.map(stock).fillna(0)
    result.to_csv(folder / "workplace_reference.csv", index=False)
    return result


def survey_files(context):
    directory = Path(context.config("data_path")) / "structural_survey"
    paths = {year: directory / f"se_zpers_{year}_ch.csv" for year in range(2021, 2025)}
    validation = context.config("workplace_validation_survey_file")
    if validation:
        paths[2024] = Path(validation)
    return paths


def validate(context):
    apprentice_references.verify_sources(context.config("data_path"))
    files = [*survey_files(context).values(), Path(apprentice_references.__file__), Path(projection.__file__)]
    return {str(file): (file.stat().st_size, file.stat().st_mtime_ns) for file in files}


def read_survey(path, year, mapping, quarters):
    names = {"RES_CANTON": "home_ct", "RES_MUN": "home", "COMPANY_MUN": "work_municipality",
             "COMPANY_QUARTER": "work_quarter", "COMPANY_CTRY": "work_country", "STARTWORK": "start_work",
             "AGE": "age", "CURRACTIVITYSTATUSI": "employed", "STATUSINEMPL_DETAIL": "job",
             "CURRACTIVITY_STUDENT": "student", "ONGOINGEDUCATION": "edu", f"WEIGHT{year}": "w"}
    fields = [*names, "STARTWORK_MUN", "STARTWORK_CTRY", "STARTWORK_QUARTER"]
    frame = pd.read_csv(path, sep=";", encoding="latin1", usecols=fields).rename(columns=names)
    ids = mapping.set_index("deprecated_municipality_id").municipality_id
    frame["home"] = frame.home.map(ids)
    frame["firm"] = frame.work_municipality.map(ids)
    frame["country"] = frame.work_country
    frame = resolve_work_start(frame, mapping, quarters)
    frame["work"] = frame.work_municipality.map(ids).where(frame.start_work.ne(1), frame.home)
    frame["work"] = frame.work.where(frame.start_work.between(1, 6) & frame.employed.eq(1))
    frame["explicit"] = frame.job.eq(50)
    frame["eligible"] = (frame.edu.eq(6) & frame.student.eq(1) & frame.employed.isin([2, 3])
                         & frame.job.isin([60, 70]) & frame.age.ge(15))
    return frame


def municipality_cantons(firms):
    # Use the canton attached to the establishment municipality in STATENT.
    names = ("Zürich|Bern|Luzern|Uri|Schwyz|Obwalden|Nidwalden|Glarus|Zug|Fribourg|Solothurn|"
             "Basel-Stadt|Basel-Landschaft|Schaffhausen|Appenzell Ausserrhoden|Appenzell Innerrhoden|"
             "St. Gallen|Graubünden|Aargau|Thurgau|Ticino|Vaud|Valais|Neuchâtel|Genève|Jura").split("|")
    codes = {name: i for i, name in enumerate(names, start=1)}
    # Older STATENT files store UTF-8 canton names read with Latin-1 encoding.
    codes.update({name.encode().decode("latin1"): value for name, value in list(codes.items())})
    # A few establishments near canton boundaries carry a neighbouring canton.
    pairs = firms.groupby(["municipality_id", "canton_name"], observed=True).size()
    majority = pairs.sort_values(ascending=False, kind="stable").reset_index()
    result = majority.drop_duplicates("municipality_id").set_index("municipality_id").canton_name.map(codes)
    if result.isna().any():
        raise ValueError("Unknown STATENT canton name")
    return result


def projected_weights(apprentices, reference, total):
    def cells(frame):
        return pd.MultiIndex.from_arrays([frame.home_ct, np.digitize(frame.age, [20, 25])])
    totals = reference.groupby(cells(reference)).w.sum()
    model_cells = cells(apprentices)
    counts = apprentices.groupby(model_cells).size()
    weights = totals.reindex(counts.index, fill_value=0)
    for canton, band in totals.index.difference(counts.index):
        alternatives = [cell for cell in counts.index if cell[0] == canton]
        if not alternatives:
            raise ValueError(f"No model apprentices in reference canton {canton}")
        nearest = min(alternatives, key=lambda cell: abs(cell[1] - band))
        weights.at[nearest] += totals.at[(canton, band)]
    return (weights / counts).reindex(model_cells).to_numpy() * (total / weights.sum())


def reference_table(year, survey, choices, probabilities):
    observed = survey.groupby("work").w
    addition = choices.stock(probabilities)
    addition = addition[addition.gt(0)]
    result = pd.concat([observed.sum().rename("recorded"), observed.size().rename("respondents"),
                        addition.rename("imputed")], axis=1).fillna(0)
    result["reference"] = result.recorded + result.imputed
    for city in CITIES:
        component = choices.residents.w * (probabilities * (choices.municipality == city)).sum(axis=1)
        reported = survey.loc[survey.work.eq(city), "w"]
        result.loc[city, "conditional_95_half"] = 1.96 * np.hypot(np.linalg.norm(reported), np.linalg.norm(component))
    return result.rename_axis("municipality").reset_index().assign(year=year)


def commute_check(held, choices, probabilities):
    donors = choices.donors
    distance = np.linalg.norm(donors.centres.loc[held.home].values - donors.centres.loc[held.firm].values, axis=1)
    cross = held.home.map(donors.cantons).ne(held.firm.map(donors.cantons))
    mass = choices.residents.w.to_numpy()[:, None] * probabilities
    rows = []
    for canton in (0, 1, 22, 25):
        sample = (held.home_ct == canton) if canton else np.ones(len(held), bool)
        model = (choices.home_canton == canton) if canton else np.ones(len(mass), bool)
        row = {"home_canton": canton, "respondents": sample.sum()}
        for name, measured, projected in [("distance_km", distance, choices.distance),
                                          ("cross_canton", cross, choices.crossing),
                                          ("over_50km", distance > 50, choices.distance > 50)]:
            row["survey_" + name] = np.average(measured[sample], weights=held.w[sample])
            row["model_" + name] = np.average(projected[model], weights=mass[model])
        rows.append(row)
    return pd.DataFrame(rows)
