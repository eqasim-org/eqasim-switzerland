# Stochastic airport activity chains: evidence and proposal

The selected implementation is **direct `home-airport` / `airport-home` only**,
using a hotel as home for visitors. Intermediate activities and later outings
are deferred. The stochastic proposal below is retained as background analysis,
not an implementation plan. Airport stages return municipal ODs and located
access/egress movements with explicit direct-chain purposes. The evidence
below comes from executing `data.microcensus.airports` against the locally
available 2015 and 2021 surveys on 16 November 2026. No respondent records are
exported into the data folder.

## What is observed

The stage returns 280 connected Swiss segments from 273 respondents: 206 segments
from the 2015 edition and 74 from 2021. Flights and foreign activities are cut at
the airport boundary. These are unweighted descriptive sample counts, not airport
demand estimates or ready-to-use generation probabilities.

Exactly 198 segments (70.7%) consist only of `home-airport` or `airport-home`.
Looking specifically at the transfer between the airport and the nearest home
activity gives a different, useful distinction:

| Ground direction | Segments | Home observed | Direct transfer | Transfer with intermediate activities |
| --- | ---: | ---: | ---: | ---: |
| To airport | 25 | 24 | 22 (91.7%) | 2 (8.3%) |
| From airport | 255 | 249 | 221 (88.8%) | 28 (11.2%) |

For outbound transfers, home means the last home before the airport; for inbound
transfers it means the first home after the airport. Seven segments have no
observed home and cannot identify a complete home-airport transfer. Among the
28 inbound transfers with intermediate activities, 24 contain one, two contain
two, and two contain three. Intermediate inbound activities include leisure,
shopping, work and other purposes. Their observed dwell-time median is 22.5
minutes, with quartiles 7 and 126.5 minutes. These describe individual stops,
not complete detours; the distribution is broad.

A separate 43 of the 249 inbound segments (17.3%) contain a non-home activity
after first reaching home. This can overlap with an intermediate airport-home
stop. The first later activity is leisure in 23 cases, shopping in 11, other in
6, and work in 3. The median observed wait at home before this outing is 90
minutes (quartiles 38.5 and 137.5). Consecutive home activities are not counted as
outings.

There is a strong time-of-day association in these diary observations:

| Departure from airport | Home observed | Intermediate transfer stop | Later non-home activity after home |
| --- | ---: | ---: | ---: |
| Before 12:00 | 25 | 6 | 13 (52.0%) |
| 12:00–18:00 | 81 | 12 | 23 (28.4%) |
| From 18:00 | 143 | 10 | 7 (4.9%) |

This partly reflects the shorter remaining observation window after late arrival.
It is not evidence that late travellers never go out the following day.
For inbound transfers, stop proportions across home-airport Euclidean distance
bands are similar: 16/143 below 50 km, 8/74 at 50–100 km, and 4/32 from 100 km.
The current sample does not establish a reliable distance effect on stopping.
Later outings do decline across these bands (35/143, 6/74, 2/32), but time,
traveller characteristics and distance may be confounded.

## Suggested first model

Use a small stochastic model with strong pooling, keeping the fixed municipal
demand and the already sampled airport event time:

1. Start with `home-airport` or `airport-home`, substituting the sampled hotel
   for home when `is_visitor` is true. Draw a smoothed stop/no-stop decision for
   the transfer. Initially allow at most one intermediate activity, since most
   observed transfers with stops contain one.
2. For inbound movements, make a second decision about a later home-based outing.
   Condition this on the predicted time of reaching home and remaining daily time
   budget. Treat an outing as `home-activity-home` when feasible, using complete
   observed tours as donors; do not invent a second airport movement.
3. Select compatible donor activity bundles, retaining the joint relationship
   between purpose, dwell time, detour and event time. Resample normalized spatial
   features, such as position along the home-airport corridor and lateral detour,
   then select a suitable activity location. Independent random distances and
   durations could generate implausible detours or impossible schedules.
4. Recompute leg travel times. For outbound travel, hold airport arrival fixed and
   move departure from home earlier to accommodate a stop. For inbound travel,
   hold airport departure fixed and let stops delay home arrival. Enforce temporal
   ordering, feasible opening times and consistent modes/vehicle availability;
   fall back to the direct transfer when a proposed activity cannot fit.

A regularized Bernoulli model or smoothed coarse probabilities are sufficient.
Distance can help select geographically comparable donors and check feasibility
without fitting an unsupported distance coefficient. Avoid separate fine-grained
models by airport, mode, age, distance and hour: only 25 outbound observations
exist, and only two show an intermediate stop. Use respondent-level resampling
when assessing uncertainty, since some people contribute multiple segments.

Retain survey edition and compare results by edition; the inbound stop count is
25/182 with a home in 2015 versus 3/67 in 2021. Raw survey weights should be
handled within edition with an explicit pooling convention, rather than adding
weights across editions and interpreting them as passenger demand. Existing
airport reports and flight snapshots continue to set demand and time profiles.

For residents, a later population adapter should replace selected synthetic
agents' plans and use their actual homes, with work/student purposes compatible
with their attributes. For visitors, replacing home by hotel is a useful initial
proxy, but the household survey does not identify foreign hotel-visitor behaviour.
Use explicit visitor assumptions, especially for later work or leisure tours,
and retain the visitor tag. See [population integration](population_integration.md)
for the proposed small integration surface; no ordinary population stages need
to change for this design discussion.
