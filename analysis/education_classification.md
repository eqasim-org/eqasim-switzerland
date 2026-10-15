# Student status and pooled tertiary classification

The ordinary CatBoost student classifier keeps its existing predictors and labels.
`data.structural_survey.student_training` supplies a separate survey copy: student
school-commute departure geography is used when domestic and available; residence
is the fallback. Nonstudents retain residence geography and stay in training.
Municipality type, district and canton are mapped together. Explicit foreign
origins stay foreign. Other models retain the original processed survey.

`synthesis.population.models.education` runs after the binary classifier. It
preserves student and employment flags and samples a joint study/country outcome
from weighted SS profiles, smoothed toward broader age/employment/sex/canton
profiles. Secondary codes 2–8 remain available for school compatibility. SS codes
9–13 are pooled **before fitting** into one tertiary outcome. Synthetic tertiary
students have `study_level=tertiary`, `education_group=tertiary` and missing
`current_education`; no fine programme is invented. Internal outcome index 14 is
never an SS code. Raw/processed SS programme detail remains available for audits.

Students under 15 retain `below_survey_age_coverage`. Abroad, unknown country and
unresolved level remain explicit outcomes. Accommodation is descriptive and does
not force student status, employment or degree. Older recovered home-only people
retain their existing model bypass.

Leave-one-year-out validation on the saved 2021–2023 survey, using the new geography
and pooled outcomes, produced joint log losses of 0.873, 0.849 and 0.848 with the
existing smoothing strength 20. Unsmoothed losses were 1.363, 1.260 and 1.317;
global baselines were 1.673, 1.624 and 1.626. These validate conditional coarse
classification, not the binary CatBoost model or national stock after a rerun.
The binary model has not been refitted during this implementation check.

`python -m analysis.education_classification --data-path INPUT --output-path OUTPUT`
retains the standalone source audit. The pipeline analysis uses the transformed
training stage and reports synthetic outcomes after the user's next rerun.
