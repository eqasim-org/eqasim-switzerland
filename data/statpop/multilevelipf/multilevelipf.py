import numpy as np
import pandas as pd


def add_expansion_factor_column(df):
    if "expansion_factor" not in list(df.columns):
        df["expansion_factor"] = 1.0
    return df


def check_control_has_weight_column(controls):
    for control in controls:
        if "weight" not in list(control.columns):
            raise Exception('Each control dataframe must have a weight column!')
    return controls


class FittingProblem:
    def __init__(self, df, group_controls, group_id,
                 individual_controls=None, individual_id=""):
        if individual_controls is None:
            individual_controls = []
        self.df = add_expansion_factor_column(df)
        self.group_controls = check_control_has_weight_column(group_controls)
        self.group_id = group_id
        self.individual_controls = check_control_has_weight_column(individual_controls)
        self.individual_id = individual_id


def compute_filters(fitting_problem):
    df = fitting_problem.df
    group_controls = []
    individual_controls = []

    # create filters for group level controls
    for control in fitting_problem.group_controls:
        for _, row in control.iterrows():
            group_control = [row["weight"]]

            # build filter
            f = np.ones(df.shape[0], dtype=np.bool)
            for c in list(row.drop("weight").index):
                f &= (df[c] == row[c])

            group_control.append(f)
            group_controls.append(group_control)

    # create filters for individual level controls
    for control in fitting_problem.individual_controls:
        for _, row in control.iterrows():
            individual_control = [row["weight"]]

            # build a filter to select all individuals that match current control values
            f_individual = np.ones(df.shape[0], dtype=np.bool)
            for c in list(row.drop("weight").index):
                f_individual &= (df[c] == row[c])

            individual_control.append(f_individual)

            # select group ids corresponding to individuals to rescale
            group_ids = list(df[f_individual][fitting_problem.group_id].unique())
            f_group = df[fitting_problem.group_id].isin(group_ids)

            individual_control.append(f_group)
            individual_controls.append(individual_control)

    return group_controls, individual_controls


def group_fit(df, group_controls, group_id):
    for group_control in group_controls:
        group_weight = group_control[0]
        group_filter = group_control[1]
        df = group_adjust(df, group_filter, group_weight, group_id)
    return df


def group_adjust(df, group_filter, group_weight, group_id):
    # rescale expansion factors
    total = np.sum(df[group_filter][[group_id, "expansion_factor"]].drop_duplicates(group_id)["expansion_factor"])
    r = group_weight / total
    df.loc[group_filter, "r_factor"] = r
    df.loc[group_filter, "expansion_factor"] *= r

    return df


def individual_fit(df, controls, group_id, algorithm="ipu"):
    for control in controls:
        weight = control[0]
        individual_filter = control[1]
        group_filter = control[2]

        if algorithm is "ipu":
            df = individual_adjust_ipu(df, individual_filter, group_filter, weight)
        elif algorithm is "ent":
            df = individual_adjust_ent(df, control, group_id)
    return df


def individual_adjust_ipu(df, f_individual, f_group, weight):
    # compute scaling factor
    total = np.sum(df[f_individual]["expansion_factor"])
    r = weight / total

    # assign to groups
    df.loc[f_group, "r_factor"] = r
    df.loc[f_group, "expansion_factor"] *= r

    return df


def individual_adjust_ent(df, group_id, f, weight):
    return df


def is_converged(f, r, tol_abs, tol_rel):
    if np.all(f * np.abs(1 - 1 / r) < tol_abs) and np.all(np.abs(1 - r) < tol_rel):
        print("Expansion factors have converged.")
        return True
    return False


def parallel_fit(context, args):
    index, task = args
    fitting_problem, algorithm, tol_abs, tol_rel, maxiter = task

    df = fitting_problem.df
    group_controls, individual_controls = compute_filters(fitting_problem)

    with context.progress(total=maxiter, position=index, desc="progress #%s" % str(index)) as progress:
        for i in range(maxiter):
            df["r_factor"] = 1.0
            df = group_fit(df, group_controls, fitting_problem.group_id)

            if algorithm == "ipu":
                df = individual_fit(df=df, controls=individual_controls,
                                    group_id=fitting_problem.group_id, algorithm="ipu")
            elif algorithm == "ent":
                df = individual_fit(df=df, controls=fitting_problem.individual_controls,
                                    group_id=fitting_problem.group_id, algorithm="ent")

            progress.update()

            if is_converged(f=df["expansion_factor"], r=df["r_factor"], tol_abs=tol_abs, tol_rel=tol_rel):
                df = df.drop("r_factor", axis=1)
                return df

    print("Reached max iteration ", maxiter)
    df = df.drop("r_factor", axis=1)
    return df


def fit(context, fitting_problem, algorithm="ipu", tol_abs=1e-3, tol_rel=1e-3, max_iter=2000, parallelize_on=None):
    tasks = []

    if parallelize_on is None:
        tasks.append((fitting_problem, algorithm, tol_abs, tol_rel, max_iter))
    else:
        categories = list(fitting_problem.df[parallelize_on].unique())
        for category in categories:
            sub_df = fitting_problem.df[fitting_problem.df[parallelize_on] == category]

            sub_group_controls = []
            for group_control in fitting_problem.group_controls:
                sub_group_controls.append(group_control[group_control[parallelize_on] == category])

            sub_individual_controls = []
            for individual_control in fitting_problem.individual_controls:
                sub_individual_controls.append(individual_control[individual_control[parallelize_on] == category])

            sub_problem = FittingProblem(df=sub_df,
                                         group_controls=sub_group_controls,
                                         group_id=fitting_problem.group_id,
                                         individual_controls=sub_individual_controls,
                                         individual_id=fitting_problem.individual_id)
            tasks.append((sub_problem, algorithm, tol_abs, tol_rel, max_iter))

    print("Fitting to data to controls...")
    with context.parallel() as parallel:
        result = parallel.map(parallel_fit, tasks)

    return pd.concat(result)
