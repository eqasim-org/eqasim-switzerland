"""Weighted apprenticeship and training-workplace projections used in the 2023 pilot."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


# FSO major regions, in the order of the residential stock vector.
REGIONS = [[25, 22, 23], [2, 10, 11, 24, 26], [12, 13, 19], [1],
           [8, 14, 15, 16, 17, 18, 20], [3, 4, 5, 6, 7, 9], [21]]
REGION_ID = {c: i for i, cantons in enumerate(REGIONS) for c in cantons}


def commute_groups(cantons):
    return np.select([cantons == c for c in (1, 22, 25)], [1, 2, 3], default=0)


def distance_bins(km):
    return np.digitize(km, [.00001, 10, 25, 50])


class Donors:
    """Borrow observed training municipalities from apprentices living nearby."""

    def __init__(self, survey, centres, cantons):
        self.survey = survey
        self.centres = centres
        self.cantons = cantons
        self.tree = cKDTree(centres.loc[survey.home].values)
        self.distance = np.linalg.norm(centres.loc[survey.home].values - centres.loc[survey.firm].values, axis=1)
        self.crossing = survey.home.map(cantons).ne(survey.firm.map(cantons)).to_numpy()

    def choices(self, residents):
        return Choices(self, residents)


class Choices:
    def __init__(self, donors, residents):
        self.residents = residents
        self.donors = donors
        if donors.survey.empty or residents.empty:
            raise ValueError("Training-place projection requires donors and residents")
        home_xy = donors.centres.loc[residents.home].to_numpy()
        k = min(180, len(donors.survey))
        distance, idx = donors.tree.query(home_xy, k=np.arange(1, k + 1))
        survey = donors.survey
        self.municipality = survey.firm.to_numpy(dtype=int)[idx]
        self.canton = survey.firm.map(donors.cantons).to_numpy(dtype=int)[idx] - 1
        self.home_canton = residents.home.map(donors.cantons).to_numpy(dtype=int)
        age_gap = np.minimum(abs(residents.age.to_numpy()[:, None] - survey.age.to_numpy()[idx]), 18)
        different_home_canton = self.home_canton[:, None] != survey.home.map(donors.cantons).to_numpy()[idx]
        weight = survey.w.to_numpy()[idx] * np.exp(-distance / 5 - age_gap / 5 - 1.4 * different_home_canton)
        self.prior = weight / weight.sum(axis=1, keepdims=True)
        work_xy = donors.centres.loc[self.municipality.ravel()].to_numpy().reshape((*idx.shape, 2))
        self.distance = np.linalg.norm(work_xy - home_xy[:, None, :], axis=2)
        self.crossing = self.home_canton[:, None] != self.canton + 1

    def stock(self, probabilities):
        mass = self.residents.w.to_numpy()[:, None] * probabilities
        return pd.Series(np.bincount(self.municipality.ravel(), weights=mass.ravel()))

    def fit(self, employer_target, *, local=False, residence_target=None):
        """Balance employer mass and commute patterns, allowing unselected survey candidates.

        A row's remaining probability is the chance of keeping its original non-apprentice
        status. At the projection step every row receives a training workplace.
        """
        total = self.residents.w.sum()
        weights = self.residents.w.to_numpy() / total
        share = employer_target.sum() / total
        if not 0 < share <= 1 + 1e-9 or min(employer_target) < 0:
            raise ValueError("Employer target exceeds the available resident weight")
        share = min(1.0, share)
        groups = commute_groups(self.home_canton) if local else np.zeros(len(weights), int)
        donor_groups = commute_groups(self.donors.survey.home.map(self.donors.cantons)) if local else np.zeros(len(self.donors.survey), int)
        n_groups = 4 if local else 1

        # Aggregate equivalent choices before optimisation. Each category represents
        # a home group, workplace canton, distance band and inter-canton indicator.
        categories = 260 * groups[:, None] + 10 * self.canton + 2 * distance_bins(self.distance) + self.crossing
        row = np.broadcast_to(np.arange(len(weights))[:, None], categories.shape)
        matrix = coo_matrix((self.prior.ravel(), (row.ravel(), categories.ravel())),
                            shape=(len(weights), 260 * n_groups)).tocsr()
        ids = np.arange(matrix.shape[1])
        group = ids // 260
        canton = ids % 260 // 10
        band = group * 5 + ids % 10 // 2
        crossing = ids % 2
        design = np.column_stack([np.eye(26)[canton], np.eye(5 * n_groups)[band],
                                  np.eye(n_groups)[group] * crossing[:, None]])
        group_mass = np.bincount(groups, weights, minlength=n_groups)
        target_bins = np.zeros((n_groups, 5))
        target_cross = np.zeros(n_groups)
        for i in range(n_groups):
            take = donor_groups == i
            w = self.donors.survey.w.to_numpy()[take]
            if not w.size:
                raise ValueError(f"No apprentice donors in commute group {i}")
            mass = share * group_mass[i]
            target_bins[i] = mass * np.bincount(distance_bins(self.donors.distance[take]), w, minlength=5) / w.sum()
            target_cross[i] = mass * np.average(self.donors.crossing[take], weights=w)
        target = np.r_[employer_target / total, target_bins.ravel(), target_cross]
        penalty = np.r_[np.zeros(26), np.repeat(group_mass, 5), group_mass] * (.0005 if local else 0.)
        resident_design = np.empty((len(weights), 0))
        if residence_target is not None:
            regions = self.residents.home_ct.map(REGION_ID).to_numpy(dtype=int)
            resident_design = np.column_stack([np.eye(7)[regions], self.residents.age.le(24)])
            target = np.r_[target, residence_target / total]
            penalty = np.r_[penalty, np.zeros(8)]
        cut = design.shape[1]

        def evaluate(x, probabilities=False):
            odds = np.exp(design @ x[:cut])
            row_odds = share * np.exp(resident_design @ x[cut:])
            denom = (1 - share) + row_odds * (matrix @ odds)
            if probabilities:
                return (row_odds / denom)[:, None] * self.prior * odds[categories]
            mass = (matrix.T @ (weights * row_odds / denom)) * odds
            selected_mass = weights * (1 - (1 - share) / denom)
            grad = np.r_[mass @ design, selected_mass @ resident_design] - target + penalty * x
            value = weights @ np.log(denom) - target @ x + .5 * (penalty @ (x * x))
            return value, grad

        fit = minimize(evaluate, np.zeros(len(target)), jac=True, method="L-BFGS-B",
                       bounds=[(-24, 24)] * len(target),
                       options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 1000})
        _, gradient = evaluate(fit.x)
        error = abs(gradient).max() * total
        if not np.isfinite(error) or error >= 1:
            raise ValueError(f"Apprentice fit residual {error:.2f} weighted persons ({fit.message})")
        return evaluate(fit.x, probabilities=True)


def repair_projection(options, explicit, residual, regional_stock, share_under_25):
    initial = options.fit(residual)
    pool = options.residents
    region = pool.home_ct.map(REGION_ID)
    base = np.bincount(region, pool.w * initial.sum(axis=1), minlength=7)
    final_stock = residual.sum() + explicit.w.sum()
    regional_target = final_stock * regional_stock / regional_stock.sum()
    regional_target -= np.bincount(explicit.home_ct.map(REGION_ID), explicit.w, minlength=7)
    young_target = share_under_25 * final_stock - explicit.loc[explicit.age.le(24), 'w'].sum()
    return options.fit(residual, residence_target=np.r_[.5 * (base + regional_target), young_target])
