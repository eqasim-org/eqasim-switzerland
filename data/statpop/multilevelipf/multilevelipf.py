import numpy as np


def add_expansion_factor_column(df):
    if "expansion_factor" not in list(df.columns):
        df["expansion_factor"] = 1.0
    return df


def check_control_has_weight_column(controls):
    for control in controls:
        if "weight" not in list(control.columns):
            raise Exception('Each control dataframe must have a weight column!')
    return controls


def compute_group_filters(df, group_controls):
    # create filters for group level controls
    group_filters = []
    for control in group_controls:
        for _, row in control.iterrows():
            group_filter = [row["weight"]]

            # build filter
            f = np.ones(df.shape[0], dtype=np.bool)
            for c in list(row.drop("weight").index):
                f &= (df[c] == row[c])

            group_filter.append(f)
            group_filters.append(group_filter)

    return group_filters


def compute_individual_filters(df, group_id, individual_controls):
    # create filters for individual level controls
    individual_filters = []
    for control in individual_controls:
        for _, row in control.iterrows():
            individual_filter = [row["weight"]]

            # build a filter to select all individuals that match current control values
            f_individual = np.ones(df.shape[0], dtype=np.bool)
            for c in list(row.drop("weight").index):
                f_individual &= (df[c] == row[c])

            individual_filter.append(f_individual)

            # select group ids corresponding to individuals to rescale
            group_ids = list(df[f_individual][group_id].unique())
            f_group = df[group_id].isin(group_ids)

            individual_filter.append(f_group)
            individual_filters.append(individual_filter)

    return individual_filters


class FittingProblem:
    def __init__(self, df, group_controls, group_id, individual_controls=None, individual_id=""):
        if individual_controls is None:
            individual_controls = []
        self.df = df
        self.group_controls = group_controls
        self.group_id = group_id
        self.individual_controls = individual_controls
        self.individual_id = individual_id

    def get_group_expansion_factors(self, group_filter):
        return self.df[group_filter][[self.group_id, "expansion_factor"]].drop_duplicates(self.group_id)["expansion_factor"]

    def get_individual_expansion_factors(self, individual_filter):
        return self.df[individual_filter]["expansion_factor"]


class IPUSolver:
    def __init__(self, group_rel_tol=1e-3, group_abs_tol=10, ind_rel_tol=1e-3, ind_abs_tol=10, max_iter=2000):
        self.group_rel_tol = group_rel_tol
        self.group_abs_tol = group_abs_tol
        self.ind_rel_tol = ind_rel_tol
        self.ind_abs_tol = ind_abs_tol
        self.max_iter = max_iter

    def _group_fit(self, df, group_controls, group_id):
        for group_control in group_controls:
            group_weight = group_control[0]
            group_filter = group_control[1]
            df = self._group_adjust(df, group_filter, group_weight, group_id)
        return df

    @staticmethod
    def _group_adjust(df, group_filter, group_weight, group_id):
        # rescale expansion factors
        total = np.sum(df[group_filter][[group_id, "expansion_factor"]].drop_duplicates(group_id)["expansion_factor"])
        r = group_weight / total
        df.loc[group_filter, "expansion_factor"] *= r

        return df

    def _individual_fit(self, df, controls):
        for control in controls:
            weight = control[0]
            individual_filter = control[1]
            group_filter = control[2]
            df = self._individual_adjust(df, individual_filter, group_filter, weight)

        return df

    @staticmethod
    def _individual_adjust(df, f_individual, f_group, weight):
        # compute scaling factor
        total = np.sum(df[f_individual]["expansion_factor"])
        r = weight / total

        # assign to groups
        df.loc[f_group, "expansion_factor"] *= r

        return df

    def _is_converged(self, df, group_controls, group_id, individual_controls):

        if len(group_controls) == 0:
            return True

        # compute WMAPE and WMAE at group level
        nominator_wmape = 0
        nominator_wmae = 0
        denominator = 0

        for group_control in group_controls:
            weight = group_control[0]
            f_group = group_control[1]
            total = np.sum(df[f_group][[group_id, "expansion_factor"]].drop_duplicates(group_id)["expansion_factor"])

            nominator_wmape += np.abs(total - weight)
            nominator_wmae += np.abs(total - weight) * np.abs(weight)
            denominator += np.abs(weight)

        wmape = nominator_wmape / denominator
        wmae = nominator_wmae / denominator

        if wmape > self.group_rel_tol and wmae > self.group_abs_tol:
            return False

        # compute WMAPE and WMAE at individual level
        nominator_wmape = 0
        nominator_wmae = 0
        denominator = 0

        for individual_control in individual_controls:
            weight = individual_control[0]
            f_individual = individual_control[1]
            total = np.sum(df[f_individual]["expansion_factor"])

            nominator_wmape += np.abs(total - weight)
            nominator_wmae += np.abs(total - weight) * np.abs(weight)
            denominator += np.abs(weight)

        wmape = nominator_wmape / denominator
        wmae = nominator_wmae / denominator

        if wmape > self.ind_rel_tol and wmae > self.ind_abs_tol:
            return False

        return True

    def fit(self, problem):

        df = problem.df
        group_controls = problem.group_controls
        group_id = problem.group_id
        individual_controls = problem.individual_controls

        for i in range(self.max_iter):
            # group fit and check convergence
            df = self._group_fit(df=df, group_controls=group_controls, group_id=group_id)
            if self._is_converged(df=df, group_controls=group_controls, group_id=group_id,
                                  individual_controls=individual_controls):
                return df, True

            # individual fit and check convergence
            df = self._individual_fit(df=df, controls=individual_controls)
            if self._is_converged(df=df, group_controls=group_controls, group_id=group_id,
                                  individual_controls=individual_controls):
                return df, True

        return df, False
