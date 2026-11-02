# Home–education distance comparison

Run `analysis.education_distances` using the existing population configuration.
The stage follows the usual `configure`/`execute` structure and writes its results
to `<analysis_path>/education_distances`. It depends on assigned primary education
locations and the actual activity table; it does not use an exported MATSim XML.

Each person contributes one primary home-to-education distance. Repeated visits
to the same education activity do not give that person extra weight. Microcensus
uses the primary education destination selected by `data.microcensus.commute`,
including education activities appearing only at the start of the trip chain.
Synthetic people enter only when they have an actual `education` activity.

MZ home-to-education distance uses the last observed home activity preceding the
primary education activity; if none exists, use the first observed home afterwards.
This includes `home_secondary` locations and follows diary order, not whichever
home happens to be closest to school. Household coordinates are an explicit
fallback. The commute stage retains selected home coordinates, source and the
original household-based distance. Household municipality distance also remains
available in the analysis observation CSV's residence comparison field.

The stage produces two comparisons for age groups 0–5, 6–14, 15–17, 18–23,
24–29, 30–39, 40–49, 50–59 and 60+:

- Native distances: microcensus and population coordinate distances in EPSG:2056;
  structural-survey `DISTTOSCHOOL_AIR`, assumed to be kilometres as confirmed by
  the user. Set `education_distance_ss_unit: m` for an extract using metres.
- Municipality-centroid distances: all sources use the same municipality
  centroids. This deliberately removes differences in address/quarter precision;
  two places within the same municipality consequently have distance zero.

Structural-survey origin is `DEPFORSCHOOLMUN`, the actual departure municipality
of the education commute, rather than registered `RES_MUN`. This matches the
origin of `DISTTOSCHOOL_AIR`. Unknown or foreign departure origins are excluded
from the domestic comparison; they are not replaced by registered residence.
`distance_registered_residence_municipality_km` in the observation CSV retains
the residence-based measure separately. MZ uses its observed diary home;
population still uses its assigned home. Term-time residence remains an explicit
interpretation limit until represented in the synthetic population.

Both endpoints must be domestic. Positive foreign education-country codes are
excluded from the structural survey; missing-country codes do not override a
valid domestic municipal destination. Missing distances and negative BFS status
codes are excluded from native statistics, while missing coordinates/geography
remain visible in coverage counts. Weights are microcensus `person_weight`,
structural-survey `weight`, and one per synthetic person. Distribution shares
do not require scaling the population sample to full population.

The all-education view includes nonstudent education activities in microcensus
and population. A second student-only view helps inspect differences in this
scope. Structural survey covers recorded students and usual education commuting,
not a diary day. Microcensus and population use `specific_day_scenario`, which
includes school holidays when set to `workday`. Different survey years and
attendance frequencies therefore remain interpretation limits. Structural survey
does not cover children under 15; microcensus does not cover those under 6.

Outputs include `coverage.csv`, person-level `distance_observations.csv`, weighted
means and quantiles in `distance_summary.csv`, weighted CDFs in `distance_cdf.csv`,
and standalone PNG/PDF figures for each scope and distance definition. Every age
group/source is reported, including groups with no valid observations.
