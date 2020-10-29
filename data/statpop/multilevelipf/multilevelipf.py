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


class IPUSolver:
    def __init__(self, tol_abs=1e-3, tol_rel=1e-3, max_iter=2000):
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
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
        df.loc[group_filter, "r_factor"] = r
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
        df.loc[f_group, "r_factor"] = r
        df.loc[f_group, "expansion_factor"] *= r

        return df

    def _is_converged(self, f, r):
        if np.all(f * np.abs(1 - 1 / r) < self.tol_abs) and np.all(np.abs(1 - r) < self.tol_rel):
            return True
        return False

    def fit(self, args):

        index, problem = args

        df = problem.df
        group_controls = problem.group_controls
        group_id = problem.group_id
        individual_controls = problem.individual_controls

        for i in range(self.max_iter):
            df["r_factor"] = 1.0
            df = self._group_fit(df=df, group_controls=group_controls, group_id=group_id)
            df = self._individual_fit(df=df, controls=individual_controls)

            if self._is_converged(f=df["expansion_factor"], r=df["r_factor"]):
                df = df.drop("r_factor", axis=1)
                return df, True

        df = df.drop("r_factor", axis=1)

        return df, False
