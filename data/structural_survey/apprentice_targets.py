"""Derive residence repair rates and employer shares from the original FSO releases."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data.structural_survey import apprentice_projection as projection, apprentice_references
from data.structural_survey.apprenticeship import vocational_candidates


def configure(context):
    context.config("data_path")
    for stage in ("data.structural_survey.raw", "data.spatial.municipalities", "data.statent.statent",
                  "data.spatial.zones"):
        context.stage(stage)


def execute(context):

    # Build the donor pool from reported employer addresses before any workplace imputation.

    raw = context.stage("data.structural_survey.raw")
    municipalities, mapping = context.stage("data.spatial.municipalities")
    locations = mapping.set_index("deprecated_municipality_id").municipality_id
    centroid = municipalities.set_index("municipality_id").geometry.centroid
    centres = pd.DataFrame({"x": centroid.x, "y": centroid.y}) / 1000
    companies = context.stage("data.statent.statent")
    zones = context.stage("data.spatial.zones").zone_id
    sites = companies.dropna(subset=["enterprise_id", "x", "y", "municipality_id"])
    sites = sites.loc[sites.number_employees.gt(0) & sites.zone_id.isin(zones)]
    cantons = projection.municipality_cantons(sites)
    frame = pd.DataFrame({"home": raw.home_municipality.map(locations), "home_ct": raw.canton_id,
                          "firm": raw.work_municipality.map(locations), "country": raw.work_country,
                          "age": raw.age, "w": raw.weight, "year": raw.survey_year,
                          "explicit": raw.job_position.eq(50), "eligible": vocational_candidates(raw)})
    explicit = frame.loc[frame.explicit]
    donors = explicit.loc[explicit.country.isin([8100, -8, -9]) & explicit.home.isin(cantons.index)
                          & explicit.firm.isin(cantons.index)].copy()

    # Fit each year's ambiguous pool using same-year workplace, residence and age controls.

    folder = apprentice_references.verify_sources(context.config("data_path"))
    register = apprentice_references.employer_stock(folder / apprentice_references.EMPLOYER)
    tables = []
    for year, respondents in frame.groupby("year"):
        pool = respondents.loc[respondents.eligible]
        observed = respondents.loc[respondents.explicit]
        if pool.empty:
            continue
        if not pool.home.isin(cantons.index).all():
            raise ValueError(f"Missing home municipality for an eligible {year} apprentice")
        other_years = donors.loc[donors.year.ne(year)]
        choices = projection.Donors(other_years, centres, cantons).choices(pool)
        target = register[str(year)]
        reported = observed.groupby(observed.firm.map(cantons)).w.sum().reindex(target.index, fill_value=0)
        # The trial treats the domestic training register as the national resident benchmark.
        mass = target.sum() - observed.w.sum()
        fraction = mass / pool.w.sum()
        if not 0 <= fraction <= 1:
            raise ValueError(f"{year} register outside the attainable survey repair range")
        if fraction in (0, 1):
            probability = np.full(len(pool), fraction)
        else:
            residual = (target * ((mass + reported.sum()) / target.sum()) - reported).clip(lower=.0001)
            regions, young = apprentice_references.residence_controls(folder, year)
            fitted = projection.repair_projection(choices, observed, (residual * (mass / residual.sum())).to_numpy(),
                                                  regions, young)
            probability = fitted.sum(axis=1)
        cells = pool.assign(young=pool.age.le(24), selected=pool.w * probability)
        table = cells.groupby(["year", "home_ct", "young"])[["w", "selected"]].sum()
        table["rate"] = table.selected / table.w
        tables.append(table)
    rates = pd.concat(tables)
    rates.to_csv(Path(context.path()) / "residence_repair_rates.csv")
    return rates.rate, register, donors, centres, cantons


def validate(context):
    apprentice_references.verify_sources(context.config("data_path"))
    return [hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (projection, apprentice_references)]
