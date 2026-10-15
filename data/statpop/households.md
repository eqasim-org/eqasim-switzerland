STATPOP household recovery
=========================

The 2023 household extract contains private households only. Some people have a
record in the person–household link table but their linked household ID is absent
from that extract. This does not imply that they live alone or have no housing.

`data.statpop.statpop` uses `attach_persons` in `households.py` after joining
persons to links. With `retain_geocoded_missing_households: true` (the default),
it retains missing-household records only when `INDIC_EGID=1`, both home
coordinates are finite and positive, and the legacy municipality-centre/unknown
location flag is false. Administrative collection coordinates are not recovered.
Setting the option to false restores the former private-household inner join.

Each recovered person receives a distinct singleton modelling household, even
when their original linked ID or home coordinate is shared with other people.
The new IDs start above all IDs in the household table and linked population,
are allocated in person-ID order, and remain within int64. Existing private
household membership is unchanged.

Three fields are retained in the prepared population:

* `is_synthetic_household`: household size one is a modelling assumption.
* `statpop_source_household_id`: original linked ID, including missing codes.
* `statpop_geocoding_indicator`: original `INDIC_EGID` value.

`statpop_household_id` keeps the unique modelling ID expected by the existing
scaling stage. The source ID must not replace it: multiple institution residents
can share a source ID, which would regroup these singleton households during IPU.
Sampling and enrichment retain the additional fields through their existing
DataFrame operations. The legacy `collective_housing_resident` field refers to
coordinate quality; it is not a verified institutional-residence classification.

The existing minimum-household-age rule still applies: households whose oldest
member is under 16 are excluded. Handling younger synthetic singletons requires
a separate decision about technical representatives and household-model support.
The agreed follow-up explicitly keeps recovered under-16s excluded. The residence
scenario for retained people is documented in `analysis/non_private_residents.md`:
selected ages 16–38 receive a descriptive dorm-candidate flag and ages 39+ use an unmodelled home-only
scenario. All younger recovered residents continue through ordinary modelling.
Missing household records alone do not establish student or employment status.
Dorm proximity is descriptive and does not override employment, student status,
study level or diary matching.

MZ donor limitations
--------------------

There is no direct individual identity link from these STATPOP records to our MZ
extract. MZ surveys private households aged six and above, so matching against it
is a proxy for institutional residents; it cannot distinguish verified dormitory,
care-home or prison behaviour. See the BFS SHAPE documentation, appendix 4.1:
https://www.bfs.admin.ch/bfsstatic/dam/assets/14467538/master

In particular, singleton representation affects income and vehicle models as
well as diary matching. Do not interpret inferred assets as institution assets.
The home-only handling of recovered residents aged 39+ is an explicit user-chosen
scenario, not an observed absence of mobility. They bypass behavioural models
and their validation. The separate non-private-residents audit reports the
scenario counts and activity invariants. Verified facility classifications or
additional institutional data would be needed to refine these assumptions.

The current cached MZ person table has 55,873 respondents, including 127 aged
18–23 and 500 aged 24–29 in one-person private households. There are no such
donors aged 6–14. These are raw donor counts before matching-stage day and other
filters, not verified institutional residents. Requiring singleton MZ donors
for every recovered person would therefore be inappropriate, especially for
children; this change does not introduce that requirement into matching.

Validation (2023 main permanent residents)
------------------------------------------

An isolated replay of the household join and existing size/age filters recovered
131,592 geocoded records. Of these, 7,320 are under 16 and remain excluded by the
existing age rule; 124,272 survive it. The prepared population increases from
8,774,423 to 8,898,695. The remaining 52,602 household-join exclusions have
administrative-centre locations. Existing private memberships are identical,
recovered IDs are unique and do not collide, and all recovered locations pass
the geocoding checks. This validates preprocessing, not downstream behaviour.

Focused tests: `python -m unittest discover -s tests -p 'test_statpop_missing_households.py'`.
No production pipeline cache was refreshed for this validation.
