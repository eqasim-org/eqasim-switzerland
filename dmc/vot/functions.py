import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dmc.constants import constants


class vot_utils:

    @staticmethod
    def get_car_vot(context, df, res, modes):
        """
        Return the average VOT for car users (CHF per hour).
        
        VOT is calculated as the marginal rate of substitution between time and cost:
        VOT = -(∂U/∂time) / (∂U/∂cost) * 60 (to convert from minutes to hours)
        
        For car:
        - ∂U/∂car_time_min = beta_car_travel_time_min * lambda_car_travel_time * car_time^(lambda-1) / TIME_SCALE_MIN
        - ∂U/∂cost = beta_cost_CHF * cost_interaction
        
        Note: The division by TIME_SCALE_MIN accounts for the fact that car_time is scaled in the utility function.
        """
        # Extract estimated parameters
        params = res.get_beta_values()
        
        beta_car_time = params.get("beta_car_travel_time_min")
        lambda_car_time = params.get("lambda_car_travel_time", 1.0)
        beta_cost = params.get("beta_cost_CHF")
        lambda_cost_distance = params.get("lambda_cost_distance", 0.0)
        lambda_cost_income = params.get("lambda_cost_income", 0.0)
        
        # Calculate cost interaction terms
        ref_euclidean_distance_km = context.config("reference_euclidean_distance_km")
        ref_income_chf = constants.REF_INCOME_CHF
        TIME_SCALE_MIN = constants.TIME_SCALE_MIN
        
        euclidean_interaction_cost = (df["euclidean_distance_km"] / ref_euclidean_distance_km) ** lambda_cost_distance
        income_interaction_cost = (df["income"] / ref_income_chf) ** lambda_cost_income
        cost_interaction = euclidean_interaction_cost * income_interaction_cost
        
        # Calculate car time (same as in model)
        car_time = (df["car_travel_time_min"] + df["parking_searching_duration_min"]) / TIME_SCALE_MIN
        
        # Calculate marginal utilities
        # ∂U/∂car_time_min = beta_car_time * lambda * car_time^(lambda-1) / TIME_SCALE_MIN
        # The division by TIME_SCALE_MIN comes from the chain rule: d(car_time)/d(car_time_min) = 1/TIME_SCALE_MIN
        marginal_utility_time = beta_car_time * lambda_car_time * (car_time ** (lambda_car_time - 1)) / TIME_SCALE_MIN
        marginal_utility_cost = beta_cost * cost_interaction
        
        # VOT in CHF per minute, then convert to CHF per hour
        vot_car = (marginal_utility_time / marginal_utility_cost) * 60
        
        # Filter for car users only and with a positive cost
        # sel = ((df["mode"] == modes.index('car')) & (df["car_cost_CHF"] > 0) & (vot_car.notna()) & (vot_car > 0) & np.isfinite(vot_car))
        sel = ((df["car_cost_CHF"] > 0) & (vot_car.notna()) & np.isfinite(vot_car))
        vot_car = vot_car[sel]
        
        # compute the average overall
        mean_vot_car = np.average(vot_car) #, weights=df["person_weight"][sel])

        return vot_car, mean_vot_car

    @staticmethod
    def get_pt_vot(context, df, res, modes):
        """
        Return the average VOT for public transport users (CHF per hour).
        
        For PT, we calculate VOT for multiple time components:
        - In-vehicle time: main component of travel
        - Access/egress time: time to reach/leave PT
        - Transfer time: waiting time between connections
        
        We return the weighted average VOT based on the time composition.
        
        Note: All time components are scaled by TIME_SCALE_MIN in the utility function,
        so we must account for this when calculating marginal utilities.
        """
        # Extract estimated parameters
        params = res.get_beta_values()
        
        beta_pt_in_vehicle = params.get("beta_pt_in_vehicle_time_min")
        lambda_pt_in_vehicle = params.get("lambda_pt_in_vehicle_time", 1.0)
        
        beta_pt_access_egress = params.get("beta_pt_access_egress_time_min")
        lambda_pt_access_egress = params.get("lambda_pt_access_egress_time", 1.0)
        
        beta_pt_transfer_time = params.get("beta_pt_transfer_time_min")
        lambda_pt_transfer_time = params.get("lambda_pt_transfer_time", 1.0)
        
        beta_cost = params.get("beta_cost_CHF")
        lambda_cost_distance = params.get("lambda_cost_distance", 0.0)
        lambda_cost_income = params.get("lambda_cost_income", 0.0)
        
        # Calculate cost interaction terms
        ref_euclidean_distance_km = context.config("reference_euclidean_distance_km")
        ref_income_chf = constants.REF_INCOME_CHF
        TIME_SCALE_MIN = constants.TIME_SCALE_MIN
        
        euclidean_interaction_cost = (df["euclidean_distance_km"] / ref_euclidean_distance_km) ** lambda_cost_distance
        income_interaction_cost = (df["income"] / ref_income_chf) ** lambda_cost_income
        cost_interaction = euclidean_interaction_cost * income_interaction_cost
        
        # Marginal utility of cost
        marginal_utility_cost = beta_cost * cost_interaction
        
        # Calculate VOT for each time component
        # In-vehicle time VOT
        pt_in_vehicle_time = df["pt_in_vehicle_time_min"] / TIME_SCALE_MIN
        marginal_utility_in_vehicle = beta_pt_in_vehicle * lambda_pt_in_vehicle * (pt_in_vehicle_time ** (lambda_pt_in_vehicle - 1)) / TIME_SCALE_MIN
        vot_in_vehicle = (marginal_utility_in_vehicle / marginal_utility_cost) * 60
        
        # Access/egress time VOT
        pt_access_egress_time = df["pt_access_egress_time_min"] / TIME_SCALE_MIN
        marginal_utility_access_egress = beta_pt_access_egress * lambda_pt_access_egress * (pt_access_egress_time ** (lambda_pt_access_egress - 1)) / TIME_SCALE_MIN
        vot_access_egress = (marginal_utility_access_egress / marginal_utility_cost) * 60
        
        # Transfer time VOT
        pt_transfer_time = df["pt_transfer_time_min"] / TIME_SCALE_MIN
        marginal_utility_transfer = beta_pt_transfer_time * lambda_pt_transfer_time * (pt_transfer_time ** (lambda_pt_transfer_time - 1)) / TIME_SCALE_MIN
        vot_transfer = (marginal_utility_transfer / marginal_utility_cost) * 60
        
        # Calculate weighted average VOT based on time composition
        total_time = df["pt_in_vehicle_time_min"] + df["pt_access_egress_time_min"] + df["pt_transfer_time_min"]
        
        # Avoid division by zero
        total_time = total_time.replace(0, np.nan)
        
        weight_in_vehicle = df["pt_in_vehicle_time_min"] / total_time
        weight_access_egress = df["pt_access_egress_time_min"] / total_time
        weight_transfer = df["pt_transfer_time_min"] / total_time
        
        # Weighted average VOT
        vot_pt = (weight_in_vehicle * vot_in_vehicle + 
                  weight_access_egress * vot_access_egress + 
                  weight_transfer * vot_transfer)

        # Filter for PT users only and with a positive cost
        # sel = ((df["mode"] == modes.index('pt')) & (df["pt_cost_CHF"] > 0) & (vot_pt.notna()) & (vot_pt > 0) & np.isfinite(vot_pt))
        sel = ((df["pt_cost_CHF"] > 0) & (vot_pt.notna()) & (vot_pt > 0) & np.isfinite(vot_pt))
        vot_pt = vot_pt[sel]
        vot_in_vehicle = vot_in_vehicle[sel]
        vot_access_egress = vot_access_egress[sel]
        vot_transfer = vot_transfer[sel]
        
        # compute the average overall        
        mean_vot_pt = np.average(vot_pt) #, weights=df["person_weight"][sel])

        return (vot_pt, mean_vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer)

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
