# Education anchors before donor matching

Every sampled student receives one usual education location before MZ matching.
The stage depends on `synthesis.population.sampled`, not enriched people or daily
activities. Student status and employment continue to use ordinary survey models
for private and younger recovered residents alike. Dorm evidence does not force
status or reserve places. Older recovered residents keep their home-only scenario.

For children under 15, weighted MZ student distances are drawn independently of the
future donor, using the existing age/school pools. Where no under-six distance
reference exists, nearest-kindergarten assignment is explicitly labelled. For
classified domestic students aged 15+, secondary (SS codes 2–8) and tertiary
(codes 9–13 pooled before classification) have separate pooled SS OD matrices.
Both use commute departure first, residence fallback, domestic flows only.
Secondary zone choice is softly conditioned on the existing SS age/programme distance targets, then retains programme-compatible schools. The pooled OD row still supplies route support and direction. Tertiary choice uses one
reviewed teaching-site pool across UH, FH/PH and higher vocational education.

Sparse/unsupported origins use the existing geographic/parent-origin smoothing
and explicit compatibility fallbacks. Foreign study, unknown study country and
unresolved classification retain labelled location proxies rather than fabricated
observed domestic flows. Assignment conserves student identities and supplies
finite home-to-anchor distance for matching, including students who will not
attend on the diary day. Proxies remain an uncertainty in those matching distances.

## Annual campus shares and daily attendance

BFS 2023/24 totals pool all published disjoint university programmes, including
continuing/other and legacy diplomas. The higher-vocational national reference is
versioned in `<data_path>/education_enrolment/education_higher_vocational_enrolment.csv`. The pooled
reference has 169,249 UH, 107,196 FH/PH and 60,854 higher-vocational enrolments.
Preparation-course coverage is incomplete and institution and resident frames
differ. These shares are modelling assumptions, **not absolute resident-stock caps**.

Targets normalise these annual shares to all classified domestic tertiary students
in the sample. Linked institutions use eligible teaching-site FTE to split annual
enrolment between campuses. Named campus groups and residual UH/FH/PH/higher-
vocational groups avoid allocating unlinked enrolment to the five linked institutions.
Research-only locations remain excluded from this pooled regular teaching model.

The existing IPF and controlled rounding preserve supported origin totals and OD
structural zeros while fitting campus/type margins. Unsupported students keep
labelled compatible proxies. Infeasible margins produce residuals and warnings;
they do not fabricate support. `education_tertiary_balance: false` retains pooled
OD without balancing. Within a selected campus, the existing distance/employee
preference chooses a compatible establishment inside the sampled destination zone.

With `education_tertiary_age_distance: true` (default), a national SS-fitted
distance preference for tertiary students under 24 is applied before balancing.
Balancing then preserves origin-by-age rows and the same campus/type margins.
Older students keep their original seed probabilities; no separate degree OD
matrices are introduced. See [age-distance validation](tertiary_education_age_distance.md)
for held-out-year evidence, the disable switch and conditional replay results.

Daily attendance is determined later by the donor diary. No student is forced to
attend every day. Matching uses the longer available work or education commute;
reported MZ school coordinates are used even on days without an education visit.
See `data/microcensus/commute.md` for missing-distance handling.

## Analysis and implementation validation

`analysis.tertiary_education_assignment` writes annual anchor flows and campus
counts alongside `daily_assigned_persons` and `daily_attendance_share`.
`calibrated_group_comparison.csv` compares annual anchors with integer margins;
`student_anchor_and_attendance.csv` separates sampled students, anchors, matched
students and actual daily visitors. The stock companion compares pooled tertiary
stock with SS/BFS; detailed model degree counts are unavailable rather than zero.
Nonstudent course visits remain secondary `other` activities.

A read-only replay of 2,000 saved students across five age bands produced 2,000
unique finite anchors, including 700 secondary OD and 793 balanced tertiary
anchors. All 47 fitted margins converged and assigned counts equalled integer
targets. This is an implementation fixture, not a newly generated population.
The configured dependency graph is acyclic, with anchors before matching and
matching before PT subscription modelling. The user will rerun the production
pipeline before campus attendance, distances and maps are interpreted.
