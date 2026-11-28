"""Draw apprentice training municipalities using domestic survey commutes and register shares."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from data.structural_survey.apprentice_projection import Donors


def configure(context):
    context.config("random_seed")
    context.config("workplace_reference_year", default=2023)
    context.stage("data.structural_survey.apprentice_targets")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.primary.work.survey_scope")
    context.stage("synthesis.population.spatial.primary.work.work_remotly")


def execute(context):

    # Set the register shares on the actual private-resident cohort and count home workers in place.

    _, register, donors, centres, cantons = context.stage("data.structural_survey.apprentice_targets")
    people = context.stage("synthesis.population.sampled")
    scope = people.person_id.map(context.stage("synthesis.population.spatial.primary.work.survey_scope"))
    if scope.isna().any():
        raise ValueError("Missing survey eligibility for sampled residents")
    apprentices = people.loc[scope & people.job_position.eq(50) & people.employed.eq(1)].set_index("person_id")
    home = apprentices.home_municipality_id.astype(int)
    remote = apprentices.index.isin(context.stage("synthesis.population.spatial.primary.work.work_remotly").person_id)
    year = context.config("workplace_reference_year")
    stock = register[str(year)]
    expected = len(home) * stock / stock.sum()
    quota = np.floor(expected).astype(int)
    extra = (expected - quota).nlargest(len(home) - quota.sum()).index
    quota.loc[extra] += 1
    at_home = home[remote].map(cantons).value_counts().reindex(stock.index, fill_value=0)
    workers = pd.DataFrame({"home": home[~remote], "age": apprentices.loc[~remote, "age"]}).sort_index()

    # Draw among observed employers of apprentices of similar age living nearby.

    result = pd.Series(index=workers.index, dtype="int64", name="municipality_id")
    if len(workers):
        observed = Donors(donors, centres, cantons)
        # People of the same age living in the same municipality have the same donor set.
        cells = workers.drop_duplicates()
        choices = observed.choices(cells)
        cell_index = pd.MultiIndex.from_frame(cells).get_indexer(pd.MultiIndex.from_frame(workers))
        canton_probability = np.zeros((len(cells), len(stock)))
        np.add.at(canton_probability, (np.arange(len(cells))[:, None], choices.canton), choices.prior)
        rng = np.random.default_rng(context.config("random_seed"))
        selected = draw_cantons(canton_probability[cell_index], (quota - at_home).clip(lower=0).to_numpy(), rng)
        municipalities = np.empty(len(workers), dtype=int)
        for first in range(0, len(workers), 4096):
            ids = cell_index[first:first + 4096]
            target = selected[first:first + len(ids)]
            probabilities = choices.prior[ids] * (choices.canton[ids] == target[:, None])
            cumulative = probabilities.cumsum(axis=1)
            draw = rng.random(len(ids)) * cumulative[:, -1]
            positions = (cumulative <= draw[:, None]).sum(axis=1)
            municipalities[first:first + len(ids)] = choices.municipality[ids, positions]
        result = pd.Series(municipalities, index=workers.index, name="municipality_id")

    # Record the realised integer allocation, including any quotas outside the observed geography.

    assigned = result.map(cantons).value_counts().reindex(stock.index, fill_value=0)
    pd.DataFrame({"register_places": stock, "target_sample": quota, "at_home": at_home,
                  "assigned_away": assigned, "difference": at_home + assigned - quota}).to_csv(
        Path(context.path()) / "apprentice_workplace_cantons.csv")
    return result


def draw_cantons(probabilities, quota, rng):
    """Integral transport problem with one person per row and only observed destination edges."""
    person, canton = np.nonzero(probabilities)
    n, k = probabilities.shape
    edges = np.arange(len(person))
    # Random utility preserves donor preferences within the available canton totals.
    cost = -np.log(probabilities[person, canton]) - rng.gumbel(size=len(person))
    row = coo_matrix((np.ones(len(edges)), (person, edges)), shape=(n, len(edges) + k))
    col = coo_matrix((np.r_[np.ones(len(edges)), -np.ones(k)],
                      (np.r_[canton, np.arange(k)], np.r_[edges, len(edges) + np.arange(k)])),
                     shape=(k, len(edges) + k))
    # Overflow is possible only at the cost of more than any feasible chain of canton swaps.
    overflow_cost = (k + 1) * (1 + np.ptp(cost))
    fit = linprog(np.r_[cost, np.full(k, overflow_cost)], A_eq=row.tocsr(), b_eq=np.ones(n),
                  A_ub=col.tocsr(), b_ub=quota, bounds=(0, None), method="highs")
    if not fit.success:
        raise ValueError(f"Apprentice workplace allocation failed: {fit.message}")
    selected = fit.x[:len(edges)] > .5
    if not np.array_equal(person[selected], np.arange(n)):
        raise ValueError("Expected one training canton per apprentice")
    return canton[selected]
