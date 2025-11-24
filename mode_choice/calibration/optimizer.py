import numpy as np
from mode_choice.dmc.utilities.BaseUtility import BaseUtility
from mode_choice.dmc.utilities.TourUtility import TourUtility
from mode_choice.dmc.selector.Selector import Selector
import pandas as pd
import polars as pl
import logging

logger = logging.getLogger(__name__)

class ModeShares():
    def __init__(self, path_to_global_mode_shares: str, path_to_cantonal_mode_shares: str):
        self.path_to_global_mode_shares = path_to_global_mode_shares
        self.path_to_cantonal_mode_shares = path_to_cantonal_mode_shares
        self.global_mode_shares = pd.read_csv(path_to_global_mode_shares).set_index("mode")
        self.cantonal_mode_shares = pd.read_csv(path_to_cantonal_mode_shares).reset_index("canton")

    def get_mode_share(self, mode: str, level: str = "global"):
        # the level should be global or a canton name
        if level == "global":
            share = self.global_mode_shares.loc[mode, "share"]
        else:
            share = self.cantonal_mode_shares.loc[level, mode]
        return share

    def get_actual_mode_shares(self, modes):
        return [self.get_mode_share(mode) for mode in modes]

    def compute_mode_shares(self, modes):
        tours = TourUtility.get_all_utilities().collect()
        tours = Selector.select(tours)
        tours = (tours.select(["person_id","trip_id","mode_candidates"])
                      .explode("mode_candidates")
                      .filter(pl.col("euclidean_distance") > 1e-3))
            
        counts_df = tours.group_by("mode_candidates").agg(pl.count().alias("count"))
        total = counts_df["count"].sum()
        counts_df = counts_df.with_columns( (pl.col("count") / total).alias("share") )
        counts_dict = dict(zip(counts_df["mode_candidates"], counts_df["share"]))
        return [counts_dict.get(mode, 0.0) for mode in modes]

    def get_estimated_mode_shares(self, modes=None):
        return self.objective_function.get_estimated_mode_shares(modes)
    
    def get_parameters(self, params):
        return BaseUtility.get_parameters(params)


class KaiOptimizer():

    def __init__(self, mode_shares: ModeShares, max_evals = 100, tol=1e-3):
        self.mode_shares = mode_shares
        self.max_evals = max_evals
        self.tol = tol

    def run(self):        
        logger.info("Using Kai utility calibration formula...")
        modes = ["pt", "car", "walk", "bike", "car_passenger"]
        actual_mode_shares = self.mode_shares.get_actual_mode_shares(modes)
        reference_mode = "pt"
        logger.info(f"The reference mode used in Kai optimizer is: {reference_mode}")

        max_iter = self.max_evals
        prev_mode_shares = None
        optimal_params = None        
        tol = self.tol
        for i in range(max_iter):
            simulated_mode_shares = self.mode_shares.get_estimated_mode_shares(modes)

            # Compute difference for convergence
            diff = np.linalg.norm(
                np.array([simulated_mode_shares[m] for m in modes if m in simulated_mode_shares]) -
                np.array([prev_mode_shares[m] for m in modes if m in prev_mode_shares])
            ) if prev_mode_shares is not None else np.inf
            logger.info(f"KaiOptimizer (it.{i}): Change in mode shares: {diff:.6f}")

            prev_mode_shares = simulated_mode_shares

            # Update parameters
            optimal_params = self._one_iteration(
                simulated_mode_shares=simulated_mode_shares,
                actual_mode_shares=actual_mode_shares,
                iteration=i,
                reference_mode=reference_mode
            )
            BaseUtility.set_parameters(optimal_params)

            # Check for convergence at the end, because one more adjustment would give better results, and it is not expensive
            # the get_estimated_mode_shares is the one that is expensive
            if diff < tol:
                logger.info("Converged.")
                break

        return {"params": optimal_params, "loss": diff if optimal_params is not None else np.nan}

    def _one_iteration(self, simulated_mode_shares, actual_mode_shares, iteration, beta = 0.8, reference_mode = "pt"):        
        calibrated_modes = self.modes_to_calibrate.copy()
        calibrated_modes.remove(reference_mode)

        params = [f"{mode.replace('car_passenger','cp')}.alpha_u" for mode in calibrated_modes]

        initial_parameters_values = self.get_current_parameters(params)

        z0 = actual_mode_shares[reference_mode][0]  # Reference (e.g., pt)
        m0 = simulated_mode_shares[reference_mode][0]  # Simulated reference share

        zi = np.array([actual_mode_shares[i][0] for i in calibrated_modes])  # Others: car, walk, bike
        mi = np.array([simulated_mode_shares[i][0] for i in calibrated_modes])
        asci = np.array([initial_parameters_values[i] for i in params])

        # Update parameters using Kai's formula
        new_parameters_values = (
            asci +
            (np.log(zi) - np.log(mi)) -
            (np.log(z0) - np.log(m0))
        )
        
        beta = min(beta, 1-1/(0.5*iteration+1))
        new_parameters_values = beta*asci+(1-beta)*new_parameters_values        
        return dict(zip(params, new_parameters_values.tolist()))



def configure(context):
    context.stage("data.microcensus.shares")

def execute(context):
    global_shares_output_path, cantonal_shares_output_path = context.stage("data.microcensus.shares")
    mode_shares = ModeShares(
        path_to_global_mode_shares=global_shares_output_path,
        path_to_cantonal_mode_shares=cantonal_shares_output_path
    )

    optimizer = KaiOptimizer(
        mode_shares=mode_shares,
        max_evals=50,
        tol=1e-4
    )

    return optimizer
