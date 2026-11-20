"""Compare Swiss-resident workplace stocks against weighted Structural Survey geography."""
from pathlib import Path

import geopandas as gpd
import pandas as pd

from data.structural_survey.structural_survey import resolve_work_start

WORK = "synthesis.population.spatial.primary.work."


def configure(context):
    for stage in ("synthesis.population.sampled", "data.statent.statent", "data.spatial.quarters",
                  "data.spatial.municipalities", "data.structural_survey.structural_survey",
                  WORK + "work_locations", WORK + "survey_scope"):
        context.stage(stage)
    context.config("input_downsampling")
    context.config("workplace_reference_year", default=2023)
    context.config("workplace_validation_survey_file", default=None)
    context.config("workplace_validation_quarter_file", default=None)


def execute(context):

    # Join one assigned workplace per employed resident to the register geography.

    scope = context.stage(WORK + "survey_scope")
    model = model_workers(context.stage(WORK + "work_locations"), context.stage("synthesis.population.sampled"),
                          context.stage("data.statent.statent"), scope, context.config("input_downsampling"))

    # Compare against the fitted year and, when supplied, a separately surveyed year.

    training = context.stage("data.structural_survey.structural_survey")
    year = context.config("workplace_reference_year")
    references = {year: survey_workers(training[training.survey_year.eq(year)])}
    path = context.config("workplace_validation_survey_file")
    polygon_file = context.config("workplace_validation_quarter_file")
    comparison_models = {year: model}
    if polygon_file and not path:
        raise ValueError("workplace_validation_quarter_file requires a validation survey")
    if path:
        quarters = context.stage("data.spatial.quarters")
        if polygon_file:
            quarters = gpd.read_file(polygon_file).rename(columns={
                "GMDEQNR": "quarter_id", "GMDE": "source_municipality_id"})
        additional_year, frame = read_validation_survey(path, context.stage("data.spatial.municipalities")[1],
                                                       quarters)
        if additional_year == year:
            raise ValueError("The additional validation survey must cover a different year")
        references[additional_year] = frame
        comparison_models[additional_year] = locate_in_quarters(model, quarters) if polygon_file else model

    # Write aggregate municipal and quarter stocks with survey sample support.

    results = []
    for year, survey in references.items():
        for level, frame in stock_tables(comparison_models[year], survey).items():
            frame = frame.assign(survey_year=year)
            filename = f"workplace_{level}_{year}.csv"
            frame.to_csv(Path(context.path()) / filename, index=False)
            results.append(filename)
    return results


def validate(context):
    sources = {}
    for option in ("workplace_validation_survey_file", "workplace_validation_quarter_file"):
        value = context.config(option)
        if value:
            path = Path(value)
            parts = sorted(path.parent.glob(path.stem + ".*")) if path.suffix.lower() == ".shp" else [path]
            for part in parts:
                stat = part.stat()
                sources[str(part)] = [stat.st_size, stat.st_mtime_ns]
    return sources


def model_workers(work, persons, facilities, scope, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Workplace stock requires a sample fraction in (0, 1]")
    workers = persons[persons.employed.eq(1)].set_index("person_id", verify_integrity=True)
    model = work.set_index("person_id", verify_integrity=True).copy()
    if len(workers.index.symmetric_difference(model.index)):
        raise ValueError("Workplace assignment does not cover precisely the employed population")
    if scope.reindex(model.index).isna().any():
        raise ValueError("Missing survey-scope information")
    model = model.loc[scope.reindex(model.index)]
    firms = facilities.set_index("enterprise_id", verify_integrity=True)
    remote = model.work_location_type.eq("remote")
    for column in ["municipality_id", "quarter_id"]:
        model[column] = pd.to_numeric(model.destination_id.map(firms[column]))
        model.loc[remote, column] = pd.to_numeric(workers.loc[model.index[remote], "home_" + column])
    if model.municipality_id.isna().any():
        raise ValueError("Missing assigned workplace municipality")
    model["home_municipality_id"] = pd.to_numeric(workers.home_municipality_id)
    model["weight"] = 1 / fraction
    return model


def survey_workers(survey):
    frame = survey.loc[survey.employed.eq(1) & survey.start_work.isin(range(1, 7))].copy()
    remote = frame.start_work.eq(1)
    for column in ["municipality_id", "quarter_id"]:
        frame[column] = frame["work_" + column].where(~remote, frame["home_" + column])
    return frame[frame.municipality_id.gt(0)].copy()


def read_validation_survey(path, municipalities, quarters):
    names = {"CURRACTIVITYSTATUSI": "employed", "STARTWORK": "start_work", "RES_MUN": "home_municipality_id",
             "RES_QUARTER": "home_quarter_id", "COMPANY_CTRY": "work_country",
             "COMPANY_MUN": "work_municipality", "COMPANY_QUARTER": "work_quarter"}
    fields = list(names) + ["STATYEAR", "STARTWORK_MUN", "STARTWORK_QUARTER", "STARTWORK_CTRY"]
    frame = pd.read_csv(path, sep=";", encoding="latin1", usecols=lambda c: c in fields or c.startswith("WEIGHT"))
    years = frame.STATYEAR.unique()
    if len(years) != 1:
        raise ValueError("Expected a single validation survey year")
    frame = frame.rename(columns={**names, f"WEIGHT{years[0]}": "weight"})
    frame = resolve_work_start(frame, municipalities, quarters)
    lookup = municipalities.set_index("deprecated_municipality_id").municipality_id
    frame["work_municipality_id"] = frame.work_municipality.map(lookup)
    frame["home_municipality_id"] = frame.home_municipality_id.map(lookup)
    frame["work_quarter_id"] = frame.work_quarter
    workers = survey_workers(frame)
    unknown = workers.quarter_id.gt(0) & ~workers.quarter_id.isin(quarters.quarter_id)
    if unknown.any():
        # Codes can be reused with different boundaries. Suppress quarter estimates
        # for the whole affected municipality until the matching polygons are supplied.
        mixed = workers.municipality_id.isin(workers.loc[unknown, "municipality_id"])
        workers.loc[mixed, "quarter_id"] = float("nan")
    return years[0], workers


def locate_in_quarters(model, quarters):
    """Locate model points in the polygon vintage used for independent validation."""
    if "geometry" in model:
        points = gpd.GeoDataFrame(model, geometry="geometry").to_crs(quarters.crs)
    else:
        points = gpd.GeoDataFrame(model, geometry=gpd.points_from_xy(model.x, model.y),
                                  crs=2056).to_crs(quarters.crs)
    hits = gpd.sjoin(points[["municipality_id", "geometry"]],
                     quarters[["quarter_id", "source_municipality_id", "geometry"]], predicate="intersects")
    hits = hits[hits.municipality_id.eq(hits.source_municipality_id)].sort_values("quarter_id")
    return model.assign(quarter_id=hits.loc[~hits.index.duplicated(), "quarter_id"])


def stock_tables(model, survey):
    def aggregate(frame, keys, name):
        local = frame.home_municipality_id.eq(frame.municipality_id)
        return frame.assign(local=frame.weight.where(local, 0)).groupby(keys).agg(**{
            name: ("weight", "sum"), name + "_local": ("local", "sum"), name + "_records": ("weight", "size")})

    tables = {}
    for level, keys in [("municipalities", ["municipality_id"]), ("quarters", ["municipality_id", "quarter_id"])]:
        m, s = model, survey
        if level == "quarters":
            m = model[model.quarter_id.gt(0)]
            s = survey[survey.quarter_id.gt(0)]
        table = aggregate(m, keys, "model").join(aggregate(s, keys, "survey"), how="outer").fillna(0)
        if level == "quarters":
            # Scale known-quarter weights to the municipality total for a like-for-like comparison.
            # Report both the estimate and the observed coverage.
            total = table.index.get_level_values("municipality_id").map(tables["municipalities"].survey)
            known = table.groupby(level="municipality_id").survey.transform("sum")
            table["survey_quarter_coverage"] = known / total
            table["survey_imputed"] = table.survey / table.survey_quarter_coverage
        tables[level] = table
    return {name: t.reset_index() for name, t in tables.items()}
