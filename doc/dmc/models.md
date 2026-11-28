# Current DMC estimation model

`dmc.models.model_jax` and `dmc.models.model_biogeme` now estimate the latest
conditional five-mode MNL specification. `analysis.dmc` evaluates completed
fits and does not estimate models or redefine their utilities.
`dmc.models.declarations` owns the shared parameter bounds, initial values and
normalizations; `dmc.models.utility` owns the utility equations. Neither
production estimator imports experimental utilities.

Both estimator stages validate fingerprints of shared preprocessing, equations,
declarations, export code and VOT helpers. Changing these sources invalidates
cached fits even when the estimator's own module has not changed.

## Specification

- PT transfers use the increasing, concave count transform
  `((1 + transfers)**q - 1)/(2**q - 1)`, multiplied by
  `1 + amplitude * ((distance/peak) * exp(1-distance/peak))**shape`.
  `dmc_transfer_peak_km` defaults to **7 km** and is fixed during estimation.
- Short-distance effects use `amplitude * max(1-distance/5, 0)**power` and
  vanish at and above 5 km. Their amplitudes and powers are estimated.
- Age uses a quadratic in `(age-40)/10`; income uses the same quadratic
  structure in `income/training_mean_income - 1`. The income reference is the
  weighted estimation-sample mean, stored as a fixed parameter. Income-class,
  junior, retired and old categorical distance effects are absent.
- PT is the reference for ASC, age/income curves and short-distance effects.
  Normalization identifies relative effects; it does not prove a unique global
  optimum for the nonlinear model.
- PT quality is measured at the actual origin and destination. A/B/C/D/none
  map to 0/1/2/3/4. Both negative quality coefficients start at -0.05 and their
  estimated exponents start at one. Home quality is retained only as source
  data for historical comparisons.
- PT uses prepared fare directly, without the first-10-km empirical fare
  correction. Negative cached waiting time contributes zero utility.
- Passenger utility includes four separate origin-home, origin-work,
  destination-home and destination-work indicators; destination-other is the
  reference. The routed-distance coefficient applies from zero distance and
  may reach its nonpositive bound at zero.
- Only passenger utility has an additional long-distance term:
  `beta_passenger_long_distance * max(car_passenger_distance_km - 25, 0)`.
  It uses routed passenger kilometres without distance scaling. The threshold
  is fixed at 25 km; the coefficient is estimated with upper bound **-0.01**.
  It contributes zero through 25 km and strictly negative utility beyond it.
- Passenger utility includes both history effects by default:
  `beta_passenger_history_any * passenger_history_any + beta_passenger_history * passenger_history`.
  `passenger_history_any` is one after **any earlier passenger trip today** and
  remains on until day end. `passenger_history` is one only if the **immediately
  preceding trip** was passenger and switches off after another mode. Both
  exclude the current trip and reset to zero at day start. If the previous trip
  was passenger, both indicators are active. Both coefficients are trainable,
  initialized at zero and jointly fitted with all other free coefficients.
  They represent a persistent daily association plus an additional immediate
  repeat effect; they are not causal habit estimates.

`use_exponents: false` fixes the travel-time exponents to one. The defining
short-distance and transfer shapes remain part of the model. Turning off
income or a cost interaction fixes its corresponding coefficients rather than
leaving inactive free parameters in the estimator.

## Data and estimation

Both estimators use the same preprocessing. Person/trip identifiers must be
unique. With passengers enabled, estimation retains uninterrupted sequences
starting at trip one and stops at a filtered-out trip; it never stitches
histories across gaps. Both use the same normalized, existing pipeline trip
weights. The conditional likelihood scores observed sequences using their
observed prior choices. Autonomous replay must instead propagate generated
history.

Survey preparation now preserves origin PT quality and passenger origin
purpose flags through training. Final daily trips keep their intentional
missing parking-duration sentinel and receive zero parking cost. Rebuild
`dmc.data.prepare_survey_data`, its dependent routing data and
`dmc.data.training_data` before estimating with existing older caches. No cache
was rewritten and no full estimation or population run was launched during
this code promotion.

Select the desired estimator through the existing `dmc_estimator` option or
run its stage directly in the working pipeline configuration. JAX retains the
external car **32 CHF/hour** and combined PT **18 CHF/hour** mean marginal-WTP
constraints by default. These are calibration assumptions. Biogeme still
requires `dmc_vot_constraints: false`; its utility and likelihood match JAX,
but it does not implement those equality constraints.

## Export and population application

The new model is the production baseline and the sole supported export format,
identified by numeric `utilityModelVersion: 2`. The writer exports the declared
estimated/fixed coefficients, with one explicit adjustment: the new passenger
long-distance coefficient is multiplied by three for application. All numeric
values in both mode and cost YAML files are rounded to **four decimal places**,
including fixed and auxiliary values. The fitted result retains full precision.
In particular, `cp.betaDistance_u_km` is the estimated passenger distance
coefficient, rounded for export, with no multiplier or export-time cap. It
applies at every routed distance. The additional long-distance coefficient is
exported as `cp.betaLongDistance_u_km = round(3 * estimate, 4)` alongside
`cp.longDistanceThreshold_km: 25.0`. The multiplier is applied once, before
rounding, including for fixed coefficients omitted by Biogeme. Fitted results
retain the original estimate. The short-distance curve vanishes at 5 km.
There are no categorical long-distance or very-long-distance dummies.
There is no compatibility export branch or implicit coefficient default.

Fitted ASCs are preserved. `dmc.parameter_files` returns the estimator's exports
or explicitly supplied current fitted artifacts, without substituting old
calibrated ASCs or selecting old default parameter files. With
`estimate_dmc: false`, both fitted paths must be supplied. Older parameter files
are rejected by the production entry points rather than interpreted through
another utility specification.

The production population selector uses these exported utility equations to
sample whole feasible tours. As an application assumption, both passenger-history
indicators are scoped to the current primary-home tour, rather than the survey
day: they reset at each tour boundary, but persist across computational splits.
The sampler evaluates them on candidate mode sequences, alongside car and bike
continuity. Estimation still uses observed daily histories and sequential
conditional MNL; whole-tour logit normalization in population choice differs
from that estimation likelihood.
The writer exports the two fitted effects as `cp.betaAnyPreviousPassenger_u`
and `cp.betaPreviousPassenger_u`, each rounded to four decimals without extra
multipliers. Older single-indicator fits must be re-estimated.
Static version-2 utilities share the same equations; the tour sampler adds both
conditional history effects. Java utility support is a separate integration
step; this change does not modify the eqasim Java repository.

Use [the fitted-model report](../analysis/dmc/README.md) to replay this model on
its exact MZ cohort, compare fitted and exported coefficients, and inspect VOT,
mode-share curves and elasticities. The cached-only command never estimates or
rebuilds inputs:

```bash
uv run -m analysis.dmc.run --config config_dib_10.yml
```
