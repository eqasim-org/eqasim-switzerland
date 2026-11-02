# Recovered residents: accommodation evidence and ordinary behaviour

Recovered geocoded people missing from the private-household table keep their
coordinates and technical singleton households. Under-16s remain excluded.

With `non_private_residences_enabled: true`, ages 16–38 within
`non_private_residence_radius: 250` metres of the GWR/name/FMEL reference are
marked `non_private_dorm_candidate_households`. This is possible accommodation
evidence, not proof of dorm occupancy, student status or university affiliation.
Distance, candidate EGID and the evidence source remain available for inspection.

All younger recovered residents use the ordinary employment, income, student,
study-level, cross-border, licence/car and diary models. A candidate can be
employed, a nonstudent, a secondary student or a tertiary student. No dorm swaps,
municipality stock corrections or priority campus reservations are applied.
Students with education activities use the same compatible primary assignment,
pooled tertiary OD and provisional BFS/FTE campus shares as private residents.
Nonstudent courses are recoded to `other` and assigned through secondary activity
locations. BFS annual enrolment times historical attendance fractions remains a
comparison benchmark rather than an absolute attendance cap.

The agreed older-resident policy is separate: recovered residents aged
`non_private_home_only_minimum_age: 39` and above bypass behavioural predictions
and diagnostics, have no MZ donor and receive exactly one home activity. This is
an explicit simulation assumption, not observed absence of travel.

`analysis.non_private_residents` reports cohorts and attendance, candidate
student/employment/study outcomes, destination assignment status and home-only
invariants. Its CSVs describe actual model outcomes rather than reserved places.
