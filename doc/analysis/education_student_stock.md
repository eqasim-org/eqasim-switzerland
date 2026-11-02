# Student status and degree-stock audit

`analysis.education_student_stock` follows the existing `configure`/`execute`
structure. It audits sampled `synthesis.population.enriched` and activity outputs,
the raw and processed structural survey, and BFS enrolment. It changes no models,
student flags, employment, diaries, campus targets or assignments.

Run with the current population configuration. Outputs are written to
`<analysis_path>/education_student_stock`. `input_downsampling` supplies the
sampling fraction; `education_stock_survey_year` defaults to 2023 and
`education_enrolment_year` defaults to 2023/24. Full-population estimates are
sample counts divided by that fraction, rather than exact census counts.

The audit separates private households, recovered dorm candidates, younger
ordinary recovered residents and home-only residents. Age-15+ student stock is
counted independently of whether a person has education in that day's diary.
Domestic, abroad and unknown study status remain separate. The shared imputer's
country parser is used for the raw/processed survey and repeated raw-year row
indices are reset so pooled years cannot duplicate records.

Survey results are exported per year. Average annual totals are used for pooled
comparisons; summing three annual population weights would triple the reference.
Student-rate comparisons standardise observed survey rates to the private
population's age/sex/canton composition, with and without conditioning on
employment. Pooled and 2023-only references are compared. This diagnoses broad
margin differences, not the classifier's held-out individual accuracy.

The existing level imputer is evaluated on saved students using probabilities
only: pooled smoothing, no smoothing, 2023 only, for all modelled residents. Nobody receives a new random level. Conditional draw variability is
reported separately from the much broader uncertainty of survey/sampling frames.

Outputs include pooled stock against BFS and survey references, national student
rates, employment/study overlap, institution evidence, imputation support and
counterfactual expected level totals. Explicit degree reports with zero student
flags remain visible even though the production imputer excludes them.
The residence-evidence table separates generic communal-building proximity from
explicit student-housing names/FMEL matches. This audits the uncertainty of the
descriptive accommodation flag without imposing student status. Current versus completed
education is also exported; an already held degree alone does not prove that a
new degree report is incorrect or that it describes continuing education.

The supplied 2023 BFS codebook, sheet 25, defines code 11 to include FH/PH diplomas,
code 12 to include university/ETH diplomas/licentiates, and code 13 to include
habilitation. The model labels therefore do not exactly match three pure BFS
degree categories. Also, current school country and educational origin are
different concepts: BFS educational origin is not a current-residence filter.
Scope/date differences must be reconciled before an enrolment calibration is
treated as a correct national target.

`degree_stock_comparison.csv` therefore includes both strict degree enrolment
and a separate codebook-aligned total adding FH/PH diplomas to Bachelor and
UH licentiates/diplomas to Master. `bfs_degree_definition_bridge.csv` preserves
all source categories. The diploma bridge remains source evidence. Production campus/type shares now
pool all published university programmes and the national higher-vocational reference. This audit does not modify stock or assignment. Continuing-education categories have no reliable survey-code
correspondence here and are not arbitrarily added to Bachelor/Master totals.


Synthetic tertiary is now one group. `pooled_tertiary_stock_comparison.csv` reports
all-country and domestic model/SS stock, model domestic daily visitors and annual
BFS all-programme enrolment separately. `degree_stock_comparison.csv` retains
detailed source BFS/SS evidence but leaves synthetic programme counts missing;
it must not be interpreted as zero Bachelor/Master/doctoral students. Conditional
imputation checks use the same departure-geography training copy as production.
