# Conservative tertiary distance regularisation

The pooled departure-origin OD model retains its smoothing settings (effective sample prior 100, 50 nearby municipalities, 20 km bandwidth). A small grid of alternative smoothing settings validated by holding out each survey year favoured the existing settings.

The additional distance preference is `exp(-beta * (1 - observed_share) * distance_km)`, with one coefficient shared across ages, study levels and campuses. `observed_share` is the existing effective-sample-size shrinkage weight, so strong local evidence receives less adjustment. Origins with fully observed profiles receive none. Row normalisation preserves all supported destinations; campus balancing and controlled rounding follow the adjustment.

`tertiary_od_distance_decay` defaults to 0.005 per kilometre. Its unscaled e-folding distance is 200 km. Set it to zero to disable the change. No separate age parameters, hard distance caps, moved homes or new OD support are introduced.

The fixed validation grid is 0, 0.005, 0.01 and 0.02. Holding out each of 2021, 2022 and 2023, 0.005 improves weighted destination log loss and Brier score in every year. Mean log loss changes from 4.52003 to 4.51303; mean Brier changes from 0.87684 to 0.87630. The improvement is small, not evidence of a major predictive gain. The 0.01 setting offers no aggregate advantage; 0.02 worsens log loss. The smallest beneficial tested penalty is retained.

Run `analysis.tertiary_education_distance_regularisation` to reproduce the held-out scores and setting comparison. It follows the existing `configure`/`execute` structure and writes under `<analysis_path>/tertiary_education_distance_regularisation`. Candidate selection is based on survey destination prediction, not fitting synthetic age-bin averages. Age-level distance comparisons remain validation checks; gains in one group can accompany losses in another.
