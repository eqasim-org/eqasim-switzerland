import numpy as np
from mode_choice.dmc.utilities.BaseUtility import BaseUtility
from mode_choice.dmc.utilities.TourUtility import TourUtility
from mode_choice.dmc.selector.Selector import Selector
from mode_choice.dmc.run_dmc import DMC
import pandas as pd
import polars as pl
import logging
from typing import List

logger = logging.getLogger(__name__)

class ModeShares():
    def __init__(self, context, dmc: DMC = None):
        global_shares_output_path, cantonal_shares_output_path = context.stage("data.microcensus.shares")
        self.global_mode_shares = pd.read_csv(global_shares_output_path).set_index("mode")
        self.cantonal_mode_shares = pd.read_csv(cantonal_shares_output_path).set_index("canton_name")
        self.dmc = dmc

    def set_dmc(self, dmc: DMC):
        if not isinstance(dmc, DMC):
            raise ValueError("DMC model must be provided to compute mode shares.")
        self.dmc = dmc

    def get_mode_share(self, mode: str, level: str = "global"):
        # the level should be global or a canton name
        if level == "global":
            share = self.global_mode_shares.loc[mode, "mode_share"]
        else:
            share = self.cantonal_mode_shares.loc[level, mode]
        return share

    def get_actual_mode_shares(self, modes: List[str], level: str = "global"):
        return {mode: self.get_mode_share(mode, level) for mode in modes}

    def compute_mode_shares(self, modes: List[str]):
        tours = self.dmc.run(verbose=False)
        tours = (tours.select(["person_id","trip_id","mode_candidates", "euclidean_distance_km"])
                  .explode(["mode_candidates","euclidean_distance_km"])
                  .filter(pl.col("euclidean_distance_km") > 1e-3))
            
        counts_df = tours.group_by("mode_candidates").agg(pl.count().alias("count"))
        total = counts_df["count"].sum()
        counts_df = counts_df.with_columns( (pl.col("count") / total).alias("share") )
        counts_dict = dict(zip(counts_df["mode_candidates"], counts_df["share"]))
        return {mode: counts_dict.get(mode, 0.0) for mode in modes}

class KaiOptimizer():

    def __init__(self, mode_shares: ModeShares, max_evals = 100, tol=1e-3):
        self.mode_shares = mode_shares
        self.max_evals = max_evals
        self.tol = tol
        self.available_modes = ["pt", "car", "walk", "bike", "car_passenger"]
        self.reference_mode = "pt"
        
    def run(self, dmc: DMC):        
        logger.info("\t Calibration of ASCs...")
        self.mode_shares.set_dmc(dmc)

        modes = self.available_modes.copy()
        # get actual mode shares
        actual_mode_shares = self.mode_shares.get_actual_mode_shares(modes, level="global")
        logger.info(f"\t Target (actual) mode shares: {actual_mode_shares}")

        max_iter = self.max_evals
        prev_mode_shares = None
        optimal_params = None        
        tol = self.tol
        for i in range(max_iter):
            simulated_mode_shares = self.mode_shares.compute_mode_shares(modes)

            # Compute difference for convergence
            diff = np.linalg.norm(
                np.array([simulated_mode_shares[m] for m in modes if m in simulated_mode_shares]) -
                np.array([prev_mode_shares[m] for m in modes if m in prev_mode_shares])
            ) if prev_mode_shares is not None else np.inf

            logger.info(
                f"\t\tKaiOptimizer (iteration {i}): "
                f"Mode share change: {diff:.4f}. "
                f"Simulated shares: { {k: round(v, 3) for k, v in simulated_mode_shares.items()} }"
            )

            prev_mode_shares = simulated_mode_shares

            # Update parameters
            optimal_params = self._one_iteration(
                simulated_mode_shares=simulated_mode_shares,
                actual_mode_shares=actual_mode_shares,
                iteration=i
            )
            BaseUtility.set_parameters(optimal_params)

            # Check for convergence at the end, because one more adjustment would give better results, and it is not expensive
            # the get_estimated_mode_shares is the one that is expensive
            if diff < tol:
                logger.info("Converged.")
                break

        return {"params": optimal_params, "loss": diff if optimal_params is not None else np.nan}

    def _one_iteration(self, simulated_mode_shares, actual_mode_shares, iteration, beta = 0.8):        
        calibrated_modes = self.available_modes.copy()
        reference_mode = self.reference_mode
        calibrated_modes.remove(reference_mode)

        params = [f"{mode.replace('car_passenger','cp')}.alpha_u" for mode in calibrated_modes]
        initial_parameters_values = self.get_current_parameters(params)

        z0 = actual_mode_shares[reference_mode]
        m0 = simulated_mode_shares[reference_mode]

        zi = np.array([actual_mode_shares[i] for i in calibrated_modes])
        mi = np.array([simulated_mode_shares[i] for i in calibrated_modes])
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

    def get_current_parameters(self, params:List[str]):
        return BaseUtility.get_parameters(params)


def configure(context):
    context.stage("data.microcensus.shares")

def execute(context):
    mode_shares = ModeShares(context)

    optimizer = KaiOptimizer(
        mode_shares=mode_shares,
        max_evals=50,
        tol=1e-4
    )

    return optimizer
