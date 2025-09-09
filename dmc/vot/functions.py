import numpy as np
import pandas as pd
import seaborn as sns
import biogeme.database as db
import biogeme.biogeme as bio
import matplotlib.pyplot as plt


class vot_utils:
    @staticmethod
    def compute_vot(df, res, utilities, time_col="car_travel_time_min", cost_col="car_cost_CHF", eps=1e-2, time_unit_per_hour=60.0):
        """
        Return a DataFrame with utilities and VOT (CHF per time_unit, default CHF/minute).
        If you want CHF/hour, multiply by time_unit_per_hour afterwards.
        """

        # 1) Retrieve estimated betas
        beta_values = res.getBetaValues()

        # 2) Prepare perturbed dataframes (vectorized)          
        df_t_plus  = df.copy(); df_t_plus[time_col]  = df_t_plus[time_col]  + eps
        df_t_minus = df.copy(); df_t_minus[time_col] = df_t_minus[time_col] - eps
        df_c_plus  = df.copy(); df_c_plus[cost_col]  = df_c_plus[cost_col]  + eps
        df_c_minus = df.copy(); df_c_minus[cost_col] = df_c_minus[cost_col] - eps

        # 3) Utility simulation helper (recreate Database + BIOGEME per full dataset)    
        def simulate_for_dataframe(df_input):
            database = db.Database("data", df_input)
            simulate = bio.BIOGEME(database, utilities)
            sim_df = simulate.simulate(beta_values) 
            sim_df.index = df_input.index
            return sim_df

        U_t_plus  = simulate_for_dataframe(df_t_plus)
        U_t_minus = simulate_for_dataframe(df_t_minus)
        U_c_plus  = simulate_for_dataframe(df_c_plus)
        U_c_minus = simulate_for_dataframe(df_c_minus)

        # 4) central differences per alternative (columns of sim_df)
        dudt = (U_t_plus - U_t_minus) / (2.0 * eps)
        dudc = (U_c_plus - U_c_minus) / (2.0 * eps)

        # 5) VOT per alternative: (∂U/∂T) / (∂U/∂C)    
        vot = dudt / dudc
        vot = vot * time_unit_per_hour  # optional conversion to CHF/hour if eps in minutes
        return vot

    @staticmethod
    def get_car_vot(df, res, utilities, modes, eps=1e-2):
        """
        Return the average VOT for car users (CHF per hour).
        """
        vot_car = vot_utils.compute_vot(df, res, utilities, time_col="car_travel_time_min", cost_col="car_cost_CHF", eps=eps)
        vot_car.columns = [modes[i] for i in vot_car.columns]

        car_data = vot_car.loc[df["car_availability"].astype(bool), "car"]
        return car_data[car_data.notna()].reset_index(drop=True)

    @staticmethod
    def get_pt_vot(df, res, utilities, modes, eps=1e-2):
        """
        Return the average VOT for public transport users (CHF per hour).
        """
        pt_vots = {}
        for col in ['pt_in_vehicle_time_min', 'pt_access_egress_time_min', 'pt_transfer_time_min']:
            pt_vots[col] = vot_utils.compute_vot(df, res, utilities, time_col=col, cost_col="pt_cost_CHF", eps=eps)
            pt_vots[col].columns = [modes[i] for i in pt_vots[col].columns]

        # we remove those who have subscriptions (zero cost), and we estimate the average VoT by considering access/egress/transfer times
        sel = df['pt_availability'].astype(bool) & (df.pt_cost_CHF>0)
        overall_time = df.loc[sel, ['pt_in_vehicle_time_min', 'pt_access_egress_time_min', 'pt_transfer_time_min']].sum(axis=1)
        pt_data = (df.loc[sel, "pt_in_vehicle_time_min"] * pt_vots["pt_in_vehicle_time_min"].loc[sel, "pt"] + 
                   df.loc[sel, "pt_access_egress_time_min"] * pt_vots["pt_access_egress_time_min"].loc[sel, "pt"] + 
                   df.loc[sel, "pt_transfer_time_min"] * pt_vots["pt_transfer_time_min"].loc[sel, "pt"]
                   )/ overall_time

        return pt_data[pt_data.notna()].reset_index(drop=True)

    @staticmethod
    def plot_vot(car_data, pt_data, figure_path, return_figure=False):

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(car_data, bins=50, color="#3c116a", label='Car', stat='density', alpha=0.6, ax=ax)
        sns.histplot(pt_data, bins=50, color="#b0691c", label='PT', stat='density', alpha=0.6, ax=ax)

        car_mean = car_data.mean()
        pt_mean = pt_data.mean()
        ax.text(0.98, 0.95, f"Car Mean: {car_mean:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#1f77b4', weight='bold')
        ax.text(0.98, 0.88, f"PT Mean: {pt_mean:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#ff7f0e', weight='bold')

        ax.legend()
        ax.set_xlim([0, 60])
        ax.set_xlabel("VoT [CHF/h]", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Value of Time Distribution", fontsize=14, weight='bold')

        plt.tight_layout()    
        
        if figure_path is not None:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        
        if return_figure:
            return fig, ax
        
        plt.close()
