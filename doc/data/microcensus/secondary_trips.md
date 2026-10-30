# Nonstudent education visits in secondary assignment

The synthetic trip stage groups education visits of recipients with final
`is_student=0` under `other`. Both trip endpoints are recoded before activities
are built; primary and secondary education purpose variants are included.
Student flags, enrolment stocks, trip modes, times and trip counts are preserved.
This is a modelling approximation for an unknown education/course provider,
not a claim that every visit is continuing education.

The shared `data.microcensus.secondary_trips` stage makes the same derived purpose
grouping for MZ respondents with employment status 0 or 1. Status 2 or 3 remains
student education; unknown status is not recoded. The original MZ trip stage,
donor matching, and survey education-commute records remain intact. This derived
table feeds RDA secondary distance distributions, neural training models and
their chain features, and secondary analysis. RDA learns its distance reference
by mode and travel-time bin; neural models also use activity-purpose features.
No circular dependency is introduced into MZ person/trip processing.

The existing `other` destination pool includes all STATENT establishments and
uses the existing secondary algorithm. It does not identify actual course
providers. Nonstudents no longer receive a primary education anchor. The dorm
nearest-five-campus rule is unchanged by this correction.

Trip caches/exports retain `original_preceding_purpose` and
`original_following_purpose`. Activity caches/exports retain `original_purpose`
and `nonstudent_education_recoded`. These distinguish visits reclassified as
other from ordinary other activities. Newly constructed home/international/border
activities are not labelled as reclassified education.

`analysis.tertiary_education_assignment` exports
`education_activity_reclassification.csv` with original/modelled purposes,
student status, distinct people and activity counts. Primary-campus education
counts exclude the reclassified visits; their new buildings can be inspected
after running `synthesis.population.spatial.locations` or the usual output stage.
Use `students_only` for like-for-like MZ/SS/population degree/school distance
inspection: raw MZ `all_education` still includes reported nonstudent education,
whereas population primary education excludes the newly secondary visits.

Synpp will rebuild affected trip/activity/assignment stages and secondary
reference stages on the next run. For neural assignment, the changed reference
requires retraining the four choice models: retain the default true values of
`overwrite_regional_model_if_exists`, `overwrite_subregional_model_if_exists`,
`overwrite_local_model_if_exists` and `overwrite_short_range_model_if_exists`.
No pipeline run is needed to apply the source changes; rerun results are needed
to assess the new secondary locations and distances.
