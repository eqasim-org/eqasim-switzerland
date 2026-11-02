# Current education-location eligibility

`synthesis.population.spatial.primary.education.locations` keeps the existing
configure/execute structure and returns person ID, facility ID and geometry plus
assignment metadata and home-to-anchor distance. Every sampled student gets a
usual anchor before diary matching. See `tertiary_education_assignment.md` for the
annual versus daily cohort definitions and SS OD assignment.

Students under 15 use existing school-age pools and independent weighted MZ
reference distances. Domestic secondary students aged 15+ retain programme codes
2–8 and the documented NOGA pools in `doc/synthesis/population/spatial/primary/education/secondary.md`.
A separate pooled secondary SS matrix chooses a compatible destination zone.

All tertiary programmes form one model outcome and one SS OD cohort.
`pooled_compatibility` takes the union of reviewed regular programme eligibility,
restricted to teaching establishments. It does not make synthetic programme
claims. A site excluded for every programme stays excluded; exclusion for only
one former programme does not exclude it from the pooled cohort. Research-only
sites are not regular pooled teaching destinations. Detailed source compatibility
and programme-specific audit functions remain available as evidence.

BFS annual pooled enrolment shares, with eligible-site FTE campus splits, supply
provisional campus/type balancing margins. Daily attendance is a later donor
diary outcome. Abroad/unknown/unresolved classifications and unsupported OD rows
retain labelled compatible proxies, conserving the student's anchor. Proxy
coordinates are not evidence of an observed domestic commute. Children retain
age coverage labels. Nonstudent course activities are recoded to secondary other
and do not receive student anchors.

Implementation validation covered 1,026 eligible tertiary teaching establishments
and 344 campuses in the saved registry. Registry identity and FTE capacity remain
modelling assumptions; those counts are not observed campus student populations.
