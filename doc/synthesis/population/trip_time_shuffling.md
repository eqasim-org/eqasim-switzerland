# Trip-time shuffling

`synthesis.population.trips` calls `shuffle_trip_times` at the end of its module.
Set `trip_time_shuffle` to one of:

- `half_hour` (default): shuffle each destination activity's start within its
  original half-hour of the day. An 08:12 start stays in [08:00, 08:30); a 14:49
  start stays in [14:30, 15:00). Integer seconds exclude the upper endpoint.
- `legacy`: retain the former common offset per person, uniformly within
  ±30 minutes, limited by the first departure's distance from midnight.
- `none`: use unshuffled, second-rounded donor times, useful as a reference.

The half-hour method anchors **arrivals**, which become activity start times
in `synthesis.population.activities`. Initial activities still start at zero.
Each trip's departure moves with its arrival, preserving travel duration.
Departures do not have a separate half-hour-bin restriction: activity ends and
durations can change, rather than moving the entire day's schedule rigidly.

Forward/backward bounds reserve enough time to complete the remaining chain.
Each initial and intermediate activity retains at least one minute, including
activities originally shorter than one minute or zero. Feasible arrivals are sampled forward within their
remaining bounds, so closely spaced trips can restrict the sampling interval.
If the one-minute requirement cannot fit inside the original bins, their upper
bounds expand only as far as the earliest feasible schedule requires; the
number of expanded bounds and largest extension are logged. The minimum
duration takes priority over strict bin membership in these exceptional cases.
The conditional distribution is not uniform over all feasible daily schedules.
Activities in different bins may change duration by up to almost an hour;
this method does not impose a separate duration-change cap.

After-midnight times retain their day offset. Infeasible source chains (such as
negative departures or overlapping donor trips) are logged and kept at their
second-rounded original times, rather than silently changing travel durations
or moving activities into different bins. Non-finite times and duplicate trip
keys raise errors. Valid source chains gain no overlaps or negative departures.

All methods use the configured `random_seed`, preserve input row order and
person/trip identifiers, and leave upstream frames unchanged. The calling RNG
advances by the legacy number of draws in every mode; per-trip half-hour draws
use a copied stream. This keeps the subsequent Swiss border-event draws
identical across modes. Separately constructed border trips retain their
sampling rules and are not subject to the donor-diary bin constraint; in the
new mode their initial departure is delayed to at least 00:01 when needed,
moving departure and arrival together. `synthesis.population.activities` also
extends the final activity's end beyond 30h when necessary to leave one full
minute after a late arrival. The legacy mode retains the original end times.

Opening hours are not consulted yet. This change preserves the survey's
activity-start half-hours as a basis for future opening-hour constraints.

## Comparison analysis

Add `analysis.synthesis_trips` to a Synpp run list. It calls the actual trips
`execute` three times with the same inputs and seed, overriding only the shuffle
method. Outputs are written under `analysis_path/synthesis_trips`:

- Activity-start histograms, including positions inside the half-hour.
- Intermediate-activity duration histograms, including a short-duration view.
- Activity-start and duration distributions by destination purpose.
- Changes in activity starts, activity durations and trip durations.
- `summary.csv` / `summary.json` with chain and bin checks.
- Separate border-event audit CSVs in each method's subdirectory.

Start-time, activity-duration and their change histograms use 10-minute bins,
including the purpose panels and the within-half-hour view. The trip-duration
change diagnostic retains second-level bins to expose any travel-time changes.

Initial home and final open-ended activity durations are excluded from duration
plots; their starts and destination arrivals are not interchangeable. No
person-level trip tables are exported by this analysis.

For a focused comparison using existing inputs without rebuilding the
population:

```bash
uv run -m analysis.synthesis_trips --config config_dib_10.yml --cached-inputs
```

This explicitly reuses the existing upstream pickles in that configuration's
working directory read-only. It requires exactly one cache per dependency and
does not validate their freshness against changed upstream code. A normal
Synpp run uses its normal dependency and cache validation.
