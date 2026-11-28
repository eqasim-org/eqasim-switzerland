"""Polars implementation of the current DMC export's utility equations."""
import polars as pl
from dmc.parameters import NAMES_CONVERSION

from mode_choice.defaults import (MODES, POWER_FLOOR,
                                  DISTANCE_COST_INTERACTION, INCOME_COST_INTERACTION, CANTON_ALPHA_KEYS, CALIBRATION_CANTON_COLUMN)


def power(base, exponent):
    # Match dmc.models.model_jax._safe_power, including zero ** zero.
    if exponent == 1.0:
        return pl.when(base > 0).then(base).otherwise(0.0)
    return pl.when(base > 0).then(base.clip(lower_bound=POWER_FLOOR).pow(exponent)).otherwise(0.0)


def expressions(p, *, distance_interaction=DISTANCE_COST_INTERACTION, income_interaction=INCOME_COST_INTERACTION):
    """Current static utilities; conditional history needs a sequence sampler."""
    from dmc.models.declarations import define_parameters
    from dmc.models.utility import build
    validate_for_population(p)
    declarations = define_parameters(False, True, True)
    coefficients = {name: p[NAMES_CONVERSION[name]] for name in declarations}

    class Columns(dict):
        def __missing__(self, name):
            return pl.col(name)

    class Context:
        def config(self, name):
            return {'distance_cost_interaction': distance_interaction,
                    'income_cost_interaction': income_interaction,
                    'use_income_in_dmc': income_interaction,
                    'reference_euclidean_distance_km': p['referenceEuclideanDistance_km'],
                    'reference_income_chf': p['referenceIncome']}[name]

    # This constant is valid only for a model fitted with zero history effect.
    columns = Columns(passenger_history=pl.lit(0.))
    utility = build(Context(), columns, coefficients, MODES, power=power,
                    maximum=lambda a, b: pl.max_horizontal(a, b), exp=lambda x: x.exp())
    result = []
    for mode in MODES:
        value = utility[mode]
        cantons = p.get(CANTON_ALPHA_KEYS[mode], {})
        if cantons:
            value += pl.col(CALIBRATION_CANTON_COLUMN).cast(pl.String).replace_strict(cantons, default=0.0, return_dtype=pl.Float64)
        result.append(pl.when(pl.col(f'{mode}_availability')).then(value).otherwise(float('-inf')).alias(f'utility_{mode}'))
    return result


def validate_for_population(p):
    if p.get('utilityModelVersion') != 2:
        raise ValueError('Population mode choice requires parameter files for the current DMC model; re-estimate or provide current fitted artifacts')
    if p.get('cp.betaPreviousPassenger_u', 0.) != 0.:
        raise ValueError('The fitted conditional passenger-history model requires a compatible daily sequence sampler. '
                         'The current whole-tour sampler cannot use this coefficient; do not substitute observed history or zero it after fitting.')
