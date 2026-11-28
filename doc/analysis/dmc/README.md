# Analyse the fitted mode-choice model

`analysis.dmc.run` evaluates the model estimated by `dmc.models.model_jax` or
`dmc.models.model_biogeme` on its exact Microcensus estimation cohort. It has no
utility specification, parameter estimation or historical-model selection of
its own. The active equations and parameter definitions live in `dmc.models`.

## Run

From the repository root, use the completed cache selected by the working
configuration:

```bash
uv run -m analysis.dmc.run --config config_dib_10.yml
```

This command is **cached-only**: it loads completed model, training and diary
stage results from the configured working directory. It never invokes Synpp,
rebuilds survey/routing data or estimates a model. It checks configuration,
coefficient completeness/bounds, the exact preprocessed fitting cohort,
convergence and the original conditional likelihood. Missing or incompatible
fits fail explicitly. After changing a utility, first estimate the current
model through the pipeline, then run this analysis.

The default destination is `<analysis_path>/dmc/index.html`. To choose another:

```bash
uv run -m analysis.dmc.run --config config_dib_10.yml --output /path/to/report
```

The same module is a Synpp stage. Set the working configuration's run list to
`analysis.dmc.run` and run `uv run -m synpp config_dib_10.yml`. Synpp resolves the
selected estimator and its upstream dependencies normally; a missing model
cache can therefore cause **upstream estimation** in this pipeline mode. The
analysis stage itself only evaluates the completed fit. Its internal report
cache and estimator inputs remain intact; exported reports are copies.

Optional analysis settings, with their defaults:

```yaml
analysis.dmc.repetitions: 200
analysis.dmc.elasticity_step: 0.01
```

The configured `random_seed` controls common choice draws. The cached-only CLI
uses at most four available CPUs unless `--threads` is specified.

## Report contents

- Observed MZ mode shares versus exact generated-history probabilities and a
  sampled replay, for the **fitted coefficients** and the **actual exported
  parameters** of the same model. The latter include four-decimal rounding and
  the threefold long-distance passenger coefficient beyond 25 routed km.
- Mode-share comparisons by equivalent income, age, Euclidean OD distance,
  passenger routed distance, transfers, car/PT time, waiting and trip position.
  Sparse-bin counts and alternative-availability domains are explicit.
- Car and combined-PT value-of-time distributions, observed-mode summaries,
  component PT marginal willingness-to-pay values, histograms and full ECDFs.
- Own/cross expected-share elasticities with respect to income, age, distances,
  travel times, car/PT cost, transfers and waiting. Every intervention recomputes
  generated passenger history. A half-step check accompanies the default ±1%
  finite difference. Separate +10% and one-extra-transfer scenarios are saved.
- A coefficient table, replay probabilities/one sampled choice per trip,
  numerical CSV/Parquet diagnostics, source/input fingerprints and independent
  PNG/SVG figures. Main HTML embeds its figures and needs no internet access.

## Interpretation

All comparisons are **in-sample**. No held-out accuracy or parameter-uncertainty
claim is made. Original MZ person weights, joined by `(person_id, trip_id)`, are
used for the principal survey comparisons and elasticities. The conditional
likelihood uses the pipeline's mode-adjusted training weights; an additional
aggregate table uses these fit weights. The report documents this distinction.

Conditional sequence scoring uses the observed previous choice. Autonomous
replay uses a two-state forward recursion and generates its own passenger
history, resetting at each person-day. It never inserts observed previous
modes into a free-running replay. Computation is linear in trip count; sampling
runs in bounded batches. Sampling ranges describe choice-draw noise only.

The cohort follows the shared production preprocessing. Histories stop at
removed trips, and the report counts complete days and retained final trips
against the processed MZ diaries. Incomplete prefixes do not establish final
return-trip or full-day performance. The final-trip free-parking correction is
in the production data preparation, not patched into analysis caches.

VOT summaries follow `use_weights_for_vot`, as estimation does. JAX's configured
car/PT VOT targets are external assumptions; Biogeme does not implement those
equality constraints. A local elasticity of −0.3 means approximately a 0.3%
relative share decrease for a 1% increase in the named input, holding other
attributes/availability fixed. It does not mean −0.3 percentage points or prove
a causal effect. Transfer derivatives relax the integer count continuously;
the discrete +1 scenario is reported separately. No own monetary-cost feature
exists for bicycle, walking or car passenger.

This replay is sequential conditional MNL. It does not match passengers to
drivers or enforce vehicle locations across trips. Production population/Java
history integration remains a separate task, as documented in
[the production model guide](../../dmc/models.md).

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `run.py` | Synpp entry point and cached-only CLI. |
| `data.py` | Completed-stage loading, exact cohort alignment, fitted/exported coefficients and audits. |
| `replay.py` | Canonical utility evaluation, exact two-state probabilities and seeded sampling. |
| `elasticities.py` | Explicit attribute interventions, share derivatives and finite changes. |
| `report.py` | Diagnostic tables, scientific figures and self-contained HTML. |

Earlier development experiments are not dependencies of this workflow and are
not included in the production analysis package.
