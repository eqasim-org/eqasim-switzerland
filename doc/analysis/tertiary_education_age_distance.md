# Age-sensitive preferences within pooled tertiary flows

`education_tertiary_age_distance` defaults to `true` when tertiary campus
balancing is enabled. Set it to `false` to retain the previous pooled assignment.
Disabling `education_tertiary_balance` also disables this preference.

The implementation retains one pooled tertiary OD model. Students aged 15–23
receive a soft multiplier `exp(-beta * distance_km)` on its supported destination
probabilities. There is no hard distance cutoff, new destination support, degree
classification, home relocation or student-stock correction. The age threshold
is fixed at 24; it is not selected independently for cantons or municipalities.

`data.od.tertiary_education_age` fits one national coefficient by conditional
destination likelihood using the pooled SS survey years. It uses observed
school-departure locations and OD-zone centroid distances. A reported numeric
distance is not required when the departure and destination zones are usable.
Residence-fallback observations remain in the underlying pooled OD model but do
not teach this additional preference an assumed commute from registered residence.
The coefficient is constrained to [0, 0.05] per kilometre, with zero retained when
the fit offers no improvement. A bound hit is logged for inspection.

Students aged 24+ retain their original seed probabilities. An exploratory
longer-distance preference for them worsened held-out log loss and was rejected.
Balancing can still change their final destinations as campus totals are shared.

`balancing.py` fits origin-by-age rows jointly to the existing campus/type
margins. `tertiary.py` samples the corresponding rounded rows. Each origin/age
count, the OD structural zeros and the campus floor/ceiling rounding constraints
are retained. Integer campus counts can differ from the old run by one sampled
person because the fractional allocation being rounded has changed.

## Validation

Run `analysis.tertiary_education_age_distance` to fit the base OD model and age
coefficient on two years and evaluate the omitted year, rotating 2021–2023.
Outputs under `<analysis_path>/tertiary_education_age_distance` include fitted
coefficients, weighted destination log loss, Brier score and zero-support share.
Results distinguish younger/older students and reported-departure/fallback rows.
The fixed comparison is baseline, half-strength and full fitted preference.

On the October 2026 unscaled 2023 audit, all three held-out years favour the full
young preference over baseline and half strength in both log loss and Brier score.
Coefficients range from 0.00885 to 0.00940 per kilometre. The production fit on
all years is 0.00887, based on 14,013 young reported-departure observations.
Prediction gains are small; this is not a claim to have resolved term-time homes.

An isolated replay assigned all 34,557 domestic tertiary agents and conserved all
47 campus/type margins within the existing rounding tolerance. Non-tertiary
anchors were unchanged. Holding the previous diary visitors fixed, private-household
ages 18–23 move from 17.94/11.63 to 16.46/11.20 km mean/median. These are conditional
diagnostics, **not new daily attendance outcomes**. Later donor matching uses the
new education commute, so rerun through `analysis.tertiary_education_assignment`
before interpreting final activity distances, attendance or maps.
