import numpy as np
import pandas as pd
from tqdm import tqdm

def add_expansion_factor_column(df):
    if (df.columns.contains("expansion_factor") == False):
        df["expansion_factor"] = 1.0
    return df

def check_control_has_weight_column(controls):
    for control in controls:
        if (control.columns.contains("weight") == False):
            raise Exception('Each control dataframe must have a weight column!')
    return controls

class fitting_problem:
    def __init__(self, df, group_controls, group_id,
                    individual_controls=[], individual_id=""):
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
    print("Computing group-level control filters...")
    for control in fitting_problem.group_controls:
        for _, row in control.iterrows():
            group_control = []
            group_control.append(row["weight"])

            # build filter
            filter = np.ones(df.shape[0], dtype=np.bool)
            for c in list(row.drop("weight").index):
                filter &= (df[c] == row[c])

            group_control.append(filter)
            group_controls.append(group_control)

    # create filters for individual level controls
    print("Computing individual-level control filters...")
    for control in fitting_problem.individual_controls:
        for _, row in control.iterrows():
            individual_control = []
            individual_control.append(row["weight"])

            # build a filter to select all individuals that match current control values
            individual_filter = np.ones(df.shape[0], dtype=np.bool)
            for c in list(row.drop("weight").index):
                individual_filter &= (df[c] == row[c])

            individual_control.append(individual_filter)

            # select group ids corresponding to individuals to rescale
            group_ids = list(df[individual_filter][fitting_problem.group_id].unique())
            group_filter = df[fitting_problem.group_id].isin(group_ids)

            individual_control.append(group_filter)
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

        if (algorithm is "ipu"):
            df = individual_adjust_ipu(df, individual_filter, group_filter, weight)
        elif (algorithm is "ent"):
            df = individual_adjust_ent(df, control, group_id)
    return df

def individual_adjust_ipu(df, individual_filter, group_filter, weight):

    # compute scaling factor
    total = np.sum(df[individual_filter]["expansion_factor"])
    r = weight / total

    # assign to groups
    df.loc[group_filter, "r_factor"] = r
    df.loc[group_filter, "expansion_factor"] *= r

    return df

def individual_adjust_ent(df, group_id, filter, weight):

    return df

def is_converged(f, r, tol_abs, tol_rel):
    if((f * np.abs(1 - 1 / r) < tol_abs).all() and (np.abs(1 - r) < tol_rel).all()):
        print("Expansion factors have converged.")
        return True
    return False

def fit(fitting_problem, algorithm="ipu", tol_abs=1e-3, tol_rel=1e-3, maxiter=2000):
    df = fitting_problem.df
    group_controls, individual_controls = compute_filters(fitting_problem)

    print("Fitting to data to controls...")
    for i in tqdm(range(maxiter)):
        df["r_factor"] = 1.0
        df = group_fit(df, group_controls, fitting_problem.group_id)
        if (algorithm is "ipu"):
            df = individual_fit(df, individual_controls, fitting_problem.group_id, algorithm="ipu")
        elif (algorithm is "ent"):
            df = individual_fit(df, fitting_problem.individual_controls, fitting_problem.group_id, algorithm="ent")

        if (is_converged(df["expansion_factor"], df["r_factor"], tol_abs, tol_rel)):
            df = df.drop("r_factor", axis=1)
            return df

    print("Reached max iteration ", maxiter)
    df = df.drop("r_factor", axis=1)
    return df

def fit_ipf(fitting_problem, tol_abs=1e-3, tol_rel=1e-3, maxiter=2000):
    return fit(fitting_problem, algorithm="ipf", tol_abs=tol_abs, tol_rel=tol_rel, maxiter=maxiter)

def fit_ipu(fitting_problem, tol_abs=1e-3, tol_rel=1e-3, maxiter=2000):
    return fit(fitting_problem, algorithm="ipu", tol_abs=tol_abs, tol_rel=tol_rel, maxiter=maxiter)

def fit_ent(fitting_problem, tol_abs=1e-3, tol_rel=1e-3, maxiter=2000):
    return fit(fitting_problem, algorithm="ent", tol_abs=tol_abs, tol_rel=tol_rel, maxiter=maxiter)

