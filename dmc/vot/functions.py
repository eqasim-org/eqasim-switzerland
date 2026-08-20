import numpy as np
import matplotlib.pyplot as plt
from dmc.constants import constants


class vot_utils:

    @staticmethod
    def _mean(values, weights, use_weights):
        """Mean with optional survey weights and aligned pandas indices."""
        if not use_weights:
            return float(np.mean(values))
        return float(np.average(values, weights=weights.loc[values.index]))

    @staticmethod
    def _power_marginal_utility(beta, exponent, scaled_time, time_scale):
        """Derivative of ``beta * scaled_time**exponent`` per minute.

        The explicit cases at zero avoid the indeterminate ``0 * inf`` that
        otherwise appears for power transformations.
        """
        scaled_time = scaled_time.astype(float)
        result = scaled_time.copy() * np.nan
        positive = scaled_time > 0
        result.loc[positive] = (
            beta
            * exponent
            * scaled_time.loc[positive] ** (exponent - 1)
            / time_scale
        )
        if np.isclose(exponent, 1.0):
            result.loc[~positive] = beta / time_scale
        elif exponent > 1.0:
            result.loc[scaled_time == 0] = 0.0
        return result

    @staticmethod
    def get_car_vot(context, df, res, modes):
        """
        Return model-implied marginal WTP for car-time savings (CHF/hour).
        
        VOT is calculated as the marginal rate of substitution between time and cost:
        For a time reduction and compensating cost increase, the WTP is:
        WTP = (∂U/∂time) / (∂U/∂cost) * 60.

        Both marginal utilities are normally negative, so their ratio is
        positive. The distribution is restricted to observed car trips and
        its mean optionally uses the survey person weights, controlled by
        ``use_weights_for_vot``. It is a trip-level distribution, not a
        distribution of unique people.
        
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
        ref_income_chf = context.config("reference_income_chf")
        TIME_SCALE_MIN = constants.TIME_SCALE_MIN
        
        euclidean_interaction_cost = (df["euclidean_distance_km"] / ref_euclidean_distance_km) ** lambda_cost_distance
        income_interaction_cost = (df["income"] / ref_income_chf) ** lambda_cost_income
        cost_interaction = euclidean_interaction_cost * income_interaction_cost
        
        # Calculate car time (same as in model)
        car_time = (df["car_travel_time_min"] + df["parking_searching_duration_min"]) / TIME_SCALE_MIN
        
        # Calculate marginal utilities
        # ∂U/∂car_time_min = beta_car_time * lambda * car_time^(lambda-1) / TIME_SCALE_MIN
        # The division by TIME_SCALE_MIN comes from the chain rule: d(car_time)/d(car_time_min) = 1/TIME_SCALE_MIN
        marginal_utility_time = vot_utils._power_marginal_utility(
            beta_car_time, lambda_car_time, car_time, TIME_SCALE_MIN
        )
        marginal_utility_cost = beta_cost * cost_interaction
        
        # VOT in CHF per minute, then convert to CHF per hour
        vot_car = (marginal_utility_time / marginal_utility_cost) * 60
        
        # Marginal WTP is defined even when the observed trip cost is zero: it
        # evaluates a hypothetical one-CHF cost increase. Do not condition the
        # sample on positive observed cost.
        sel = (
            (df["mode"] == modes.index("car"))
            & (df["car_availability"] > 0)
            & vot_car.notna()
            & (vot_car > 0)
            & np.isfinite(vot_car)
        )
        vot_car = vot_car[sel]
        vot_car.name = "car_wtp_chf_h"
        
        # compute the average overall
        mean_vot_car = vot_utils._mean(
            vot_car,
            df["person_weight"],
            context.config("use_weights_for_vot"),
        )

        return vot_car, mean_vot_car

    @staticmethod
    def get_pt_vot(context, df, res, modes):
        """
        Return model-implied marginal WTP for PT-time savings (CHF/hour).
        
        For PT, we calculate VOT for multiple time components:
        - In-vehicle time: main component of travel
        - Access/egress time: time to reach/leave PT
        - Transfer time: waiting time between connections
        
        The trip-level PT value is the WTP for saving one minute distributed
        proportionally across the trip's non-negative in-vehicle,
        access/egress and transfer durations. Component-specific WTP values are
        returned as well. Only observed PT trips enter the distribution and its
        mean optionally uses the survey person weights, controlled by
        ``use_weights_for_vot``. It is a trip-level distribution, not a
        distribution of unique people.
        
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
        ref_income_chf = context.config("reference_income_chf")
        TIME_SCALE_MIN = constants.TIME_SCALE_MIN
        
        euclidean_interaction_cost = (df["euclidean_distance_km"] / ref_euclidean_distance_km) ** lambda_cost_distance
        income_interaction_cost = (df["income"] / ref_income_chf) ** lambda_cost_income
        cost_interaction = euclidean_interaction_cost * income_interaction_cost
        
        # Marginal utility of cost
        marginal_utility_cost = beta_cost * cost_interaction
        
        # Calculate VOT for each time component
        # In-vehicle time VOT
        pt_in_vehicle_time = df["pt_in_vehicle_time_min"] / TIME_SCALE_MIN
        marginal_utility_in_vehicle = vot_utils._power_marginal_utility(
            beta_pt_in_vehicle,
            lambda_pt_in_vehicle,
            pt_in_vehicle_time,
            TIME_SCALE_MIN,
        )
        vot_in_vehicle = (marginal_utility_in_vehicle / marginal_utility_cost) * 60
        
        # Access/egress time VOT
        pt_access_egress_time = df["pt_access_egress_time_min"] / TIME_SCALE_MIN
        marginal_utility_access_egress = vot_utils._power_marginal_utility(
            beta_pt_access_egress,
            lambda_pt_access_egress,
            pt_access_egress_time,
            TIME_SCALE_MIN,
        )
        vot_access_egress = (marginal_utility_access_egress / marginal_utility_cost) * 60
        
        # Transfer time VOT
        pt_transfer_time = df["pt_transfer_time_min"] / TIME_SCALE_MIN
        marginal_utility_transfer = vot_utils._power_marginal_utility(
            beta_pt_transfer_time,
            lambda_pt_transfer_time,
            pt_transfer_time,
            TIME_SCALE_MIN,
        )
        vot_transfer = (marginal_utility_transfer / marginal_utility_cost) * 60
        
        # WTP for a one-minute total PT saving allocated proportionally over
        # physical component durations. Negative transfer-time corrections in
        # the estimation data are not physical time shares and are clipped to
        # zero for this aggregation.
        in_vehicle_duration = df["pt_in_vehicle_time_min"].clip(lower=0)
        access_egress_duration = df["pt_access_egress_time_min"].clip(lower=0)
        transfer_duration = df["pt_transfer_time_min"].clip(lower=0)
        total_time = (
            in_vehicle_duration + access_egress_duration + transfer_duration
        ).replace(0, np.nan)

        # Compute time-share contributions directly. This remains finite when
        # a component is zero even if its power exponent is below one.
        composite_marginal_utility_time = (
            beta_pt_in_vehicle
            * lambda_pt_in_vehicle
            * (in_vehicle_duration / TIME_SCALE_MIN) ** lambda_pt_in_vehicle
            + beta_pt_access_egress
            * lambda_pt_access_egress
            * (access_egress_duration / TIME_SCALE_MIN) ** lambda_pt_access_egress
            + beta_pt_transfer_time
            * lambda_pt_transfer_time
            * (transfer_duration / TIME_SCALE_MIN) ** lambda_pt_transfer_time
        ) / total_time
        vot_pt = (composite_marginal_utility_time / marginal_utility_cost) * 60

        sel = (
            (df["mode"] == modes.index("pt"))
            & (df["pt_availability"] > 0)
            & vot_pt.notna()
            & (vot_pt > 0)
            & np.isfinite(vot_pt)
        )
        vot_pt = vot_pt[sel]
        vot_in_vehicle = vot_in_vehicle[sel]
        vot_access_egress = vot_access_egress[sel]
        vot_transfer = vot_transfer[sel]
        vot_pt.name = "pt_composite_wtp_chf_h"
        
        # compute the average overall        
        mean_vot_pt = vot_utils._mean(
            vot_pt,
            df["person_weight"],
            context.config("use_weights_for_vot"),
        )

        return (vot_pt, mean_vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer)

    @staticmethod
    def plot_vot(
        car_data,
        pt_data,
        figure_path,
        car_weights=None,
        pt_weights=None,
        return_figure=False,
    ):
        """Plot optionally weighted model-implied marginal WTP distributions."""

        def prepare(values, weights):
            values = np.asarray(values, dtype=float)
            weights = (
                np.ones(len(values), dtype=float)
                if weights is None
                else np.asarray(weights, dtype=float)
            )
            selected = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
            return values[selected], weights[selected]

        def weighted_quantile(values, weights, quantile):
            order = np.argsort(values)
            values, weights = values[order], weights[order]
            cumulative = np.cumsum(weights) - 0.5 * weights
            return float(np.interp(quantile, cumulative / cumulative[-1], values))

        use_weights = car_weights is not None or pt_weights is not None
        car_values, car_weights = prepare(car_data, car_weights)
        pt_values, pt_weights = prepare(pt_data, pt_weights)

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 5))

        upper_limit = 5 * np.ceil(max(car_values.max(), pt_values.max()) / 5)
        bins = np.linspace(0, upper_limit, 51)
        ax.hist(
            car_values,
            weights=car_weights,
            bins=bins,
            color="#3c116a",
            label="Observed car trips",
            density=True,
            alpha=0.6,
            edgecolor="black",
        )
        ax.hist(
            pt_values,
            weights=pt_weights,
            bins=bins,
            color="#b0691c",
            label="Observed PT trips (composite time)",
            density=True,
            alpha=0.6,
            edgecolor="black",
        )

        car_mean = np.average(car_values, weights=car_weights)
        pt_mean = np.average(pt_values, weights=pt_weights)
        car_median = weighted_quantile(car_values, car_weights, 0.5)
        pt_median = weighted_quantile(pt_values, pt_weights, 0.5)
        ax.text(0.98, 0.95, f"Car — mean: {car_mean:.2f}, median: {car_median:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#3c116a', weight='bold')
        ax.text(0.98, 0.88, f"PT — mean: {pt_mean:.2f}, median: {pt_median:.2f}", transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#b0691c', weight='bold')

        ax.legend()
        ax.set_xlim([0, upper_limit])
        ax.set_xlabel("Marginal WTP [CHF per hour of travel time saved]", fontsize=11)
        density_label = "Survey-weighted trip density" if use_weights else "Unweighted trip density"
        ax.set_ylabel(density_label, fontsize=11)
        ax.set_title("Model-implied marginal WTP for travel-time savings", fontsize=13, weight='bold')

        plt.tight_layout()    
        
        if figure_path is not None:
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        
        if return_figure:
            return fig, ax
        
        plt.close()
