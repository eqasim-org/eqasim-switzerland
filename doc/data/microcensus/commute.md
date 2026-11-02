# Education commute home origin

The commute stage retains the existing primary education destination: the education activity with maximum duration, or an origin-only education activity when no destination education activity appears.

Its home origin is selected from the processed diary, including both `home` and `home_secondary*` labels. Prefer the last home activity preceding that education activity. If none exists, use the first home activity afterwards. If no finite home coordinate is observed, fall back to the household coordinate from `haushalte.csv`. Diary order is defined by trip ID, with origin activity before destination activity within a trip. A nonhome trip origin is never treated as a home, and selection does not minimise distance to school.

Education commute outputs retain `commute_home_x`, `commute_home_y`, `commute_home_source`, and `commute_household_home_distance`. `commute_home_distance` now measures straight-line distance from the selected home to the same primary education destination. Household/person home fields and work commute handling are unchanged.

The distance analysis uses these selected home coordinates consistently for native distances, domestic-endpoint filtering and centroid comparisons. The diary-home comparison does not relocate synthetic homes or implement term-time residence in the population. Child reference-distance draws and the distance analysis use this diary-based measure independently of donor matching.


## Usual distance for donor matching

`data.microcensus.persons` also retains the reported usual school address from
`zielpersonen.csv`: AU_X/AU_Y (WGS84), AU_LND and AU_QAL. Valid coordinates are
transformed to LV95; missing/sentinel values remain missing. As with reported
workplace coordinates, a school address can be known without a diary-day visit.
Reported school distance uses the household home; an observed diary education
place supplies the diary-home distance only when the usual school distance is absent.

`synthesis.population.matching.commute` uses max(valid work distance, valid school
distance) for donors and recipients. Zero is a real zero commute; -1 means neither
is known. Remaining missing student distances are drawn with MZ person weights
from observed student peers in the same age band and employment group, then age
band, then all students if necessary. Overseas reported schools are retained for
their respondents but excluded from domestic imputation references; diary peers
must have domestic-only diaries. Imputed values never become reference peers.
Draws are seeded and explicitly flagged. They do not change diaries or attendance.

For the audited weekday matching pool, 6,286 of 7,982 student donors have a reported
school distance, 372 need diary fallback and 1,324 (16.6%) need peer imputation.
The latter include 37.0% of ages 18–23. Missingness is not random: after exhausting
diary fallback, these respondents have no education visit on that diary day.
The peer-distance assumption needs checking after matching, particularly for adults.
The matching stage writes MZ and population commute diagnostics by age, employment
and source, including observed donor attendance and weighted attendance rates.
