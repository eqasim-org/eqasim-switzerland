"""Optional GA postcode ranks and separate ownership of GA and Junior cards."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

OPTION = "use_sbb_ga_rank_and_junior_split"
STAGE = "synthesis.population.models.ga_ownership"
SALES = "pt_reference_data/generalabo-halbtax-mit-bevolkerungsdaten.csv"
HOUSEHOLDS = "microcensus/haushalte.csv"
PERSONS = "microcensus/zielpersonen.csv"


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.postal_codes")
    context.stage("synthesis.population.enriched")


def execute(context):

    # Rank valid postcodes in the survey year and match respondents by reported home postcode.

    root = Path(context.config("data_path"))
    ranks = postcode_ranks(pd.read_csv(root / SALES, sep=";", encoding="utf-8-sig"))
    households = pd.read_csv(root / HOUSEHOLDS, usecols=["HHNR", "W_PLZ"], encoding="latin1")
    survey_ranks = households.set_index("HHNR").W_PLZ.map(ranks)
    if not survey_ranks.index.is_unique:
        raise ValueError("Duplicate MZ household numbers in the postcode lookup")

    # Locate synthetic residents using their home coordinates in LV95.

    persons = context.stage("synthesis.population.enriched")
    adults = persons.loc[persons.age.ge(16), ["person_id", "home_x", "home_y"]]
    population_ranks = home_ranks(adults, context.stage("data.spatial.postal_codes"), ranks)

    # Estimate the national product mix among weekday children reporting either card.

    survey = pd.read_csv(root / PERSONS, encoding="latin1", usecols=["tag", "alter", "WP", "f41610a", "f41610f"])
    children = survey.loc[survey.tag.between(1, 5) & survey.alter.between(6, 15)
                          & survey.f41610a.isin([1, 2]) & survey.f41610f.isin([1, 2])]
    # Codes 1, 2 and 3 correspond to GA only, Junior only and both.
    products = children.f41610a.eq(1).astype(int) + 2 * children.f41610f.eq(1).astype(int)
    weights = children.WP
    if not np.isfinite(weights).all() or not weights.gt(0).all():
        raise ValueError("Expected positive finite MZ person weights for the Junior card split")
    mass = weights.groupby(products).sum().reindex([1, 2, 3], fill_value=0)
    if mass.sum() <= 0:
        raise ValueError("No eligible MZ children reported a GA or Junior card")
    return survey_ranks, population_ranks, (mass / mass.sum()).to_numpy()


def postcode_ranks(sales):
    rate = pd.to_numeric(sales["ga_holder_percentage"], errors="raise")
    valid = sales["Jahr_An_Anno_Year"].eq(2015) & sales["GA_AG_flag"].isna() & rate.between(0, 100)
    by_postcode = rate[valid].set_axis(pd.to_numeric(sales.loc[valid, "Postal code"], errors="raise"))
    if by_postcode.empty or not by_postcode.index.is_unique:
        raise ValueError("Expected usable 2015 SBB GA rates with one record per postcode")
    return by_postcode.rank(method="average", pct=True)


def home_ranks(persons, polygons, ranks):
    # Query each distinct residential coordinate once, including shared household addresses.
    xy = ["home_x", "home_y"]
    homes = persons[xy].dropna().drop_duplicates().reset_index(drop=True)
    points = gpd.GeoDataFrame(homes, geometry=gpd.points_from_xy(homes.home_x, homes.home_y), crs=2056)
    areas = polygons[["postal_code", "geometry"]].to_crs(2056).copy()
    areas["postal_code"] = pd.to_numeric(areas.postal_code, errors="raise")
    matches = points.sjoin(areas, how="inner", predicate="intersects")
    # Resolve a home on a shared boundary by the lower postcode, as in the calibration trial.
    homes = homes.assign(rank=matches.groupby(level=0).postal_code.min().map(ranks))
    result = persons.merge(homes[xy + ["rank"]], on=xy, how="left", validate="many_to_one").set_index("person_id")
    if not result.index.is_unique:
        raise ValueError("Duplicate synthetic person IDs in the postcode lookup")
    return result["rank"]


def ranked_probabilities(model, train_x, pop_x, train_y, weights, survey_rank, pop_rank, base):
    """Redistribute baseline expected GA ownership across residents with known ranks."""
    known = np.isfinite(pop_rank) & (base[:, 1] > 0) & (base[:, 1] < 1)
    if not known.any() or not np.isfinite(survey_rank).any():
        return base
    model.fit(np.column_stack((train_x, survey_rank)), train_y, sample_weight=weights)
    # Bound the extra prediction matrix for full-population runs.
    indices = np.flatnonzero(known)
    ranked_ga = np.empty(len(indices))
    batch_size = 100_000
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        features = np.column_stack((pop_x[batch], pop_rank[batch]))
        ranked_ga[start:start + len(batch)] = model.predict_proba(features)[:, 1]
    old = base[known]
    target = old[:, 1].mean()
    eps = np.finfo(float).eps
    odds = logit(np.clip(ranked_ga, eps, 1 - eps))
    target_odds = logit(target)
    # These bounds bracket the target even when every postcode has the same score.
    offset = brentq(lambda shift: expit(odds + shift).mean() - target,
                    target_odds - odds.max() - 1, target_odds - odds.min() + 1)
    new_ga = expit(odds + offset)
    updated = base.copy()
    others = np.arange(base.shape[1]) != 1
    non_ga = old[:, others]
    shares = non_ga / non_ga.sum(axis=1, keepdims=True)
    updated[np.ix_(known, others)] = shares * (1 - new_ga[:, None])
    updated[known, 1] = new_ga
    return updated


def assign_ownership(persons, product_probabilities, seed):
    """Keep the pooled routing draw and label the actual products of Swiss residents."""
    pooled = persons.PT_SUB_draw.eq(1)
    persons["has_ga"] = pooled & persons.age.ge(16)
    persons["has_junior"] = False
    children = persons.loc[pooled & persons.age.between(6, 15), "person_id"].sort_values()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    products = rng.choice([1, 2, 3], size=len(children), p=product_probabilities)
    persons.loc[children.index, "has_ga"] = np.isin(products, [1, 3])
    persons.loc[children.index, "has_junior"] = np.isin(products, [2, 3])


def validate(context):
    root = Path(context.config("data_path"))
    return {name: (root / name).stat().st_mtime_ns for name in (SALES, HOUSEHOLDS, PERSONS)}
