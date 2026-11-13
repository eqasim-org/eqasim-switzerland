# Endpoint walking speed with a flat zone

`calibration.walking_speed.model` returns only a dictionary of parameters.
`estimate_walking_model: false` (default) returns stored estimates immediately,
without requesting the MOBIS dataset or elevations. With estimation enabled,
the stage requests `calibration.walking_speed.dataset` and fits on training
participants only. It does not change eqasim automatically.

## Equation and units

```text
d = hypot(x_destination-x_origin, y_destination-y_origin)
L = detour_factor * d
g = (z_destination-z_origin) / L
v = v0_m_s * exp(-k * max(abs(g)-flat_grade, 0)
                -uphill_penalty * max(g-flat_grade, 0))
t = L / v
```

Coordinates and elevations are metres (EPSG:2056), speed is m/s and time seconds.
Grade is a rise/run ratio: `flat_grade=0.002` means **0.2%**, not 20%.
Within `[-flat_grade, +flat_grade]`, speed is exactly `v0_m_s`.
Beyond this zone it decreases continuously, at downhill rate `k` and uphill
rate `k+uphill_penalty`. Both fitted coefficients are nonnegative, bounded by
30. There is no downhill speed bonus or optimum-grade parameter in this model.

`v0_m_s` is fixed to the **arithmetic mean** effective speed `L/duration`
of training legs within the flat zone. Other legs cannot shift this average.
Robust least squares fits log-speed residuals outside the zone. Missing flat
observations, or missing observations beyond either boundary, cause an explicit
error rather than an unsupported fit. The width is configured, not estimated.

Use `travel_time(origin_xyz, destination_xyz, **parameters)` to apply the result.
Coincident points return zero; purely vertical displacement is rejected.
The equation describes effective elapsed walking speed, which can include stops.
Endpoint grades miss intervening hills, stairs and route barriers.

## Configuration and data

```yaml
run:
  - calibration.walking_speed.model
config:
  estimate_walking_model: true
  calibration.walking_speed.flat_grade: 0.002
  calibration.walking_speed.test_fraction: 0.1
```

Run `uv run -m synpp <configuration.yml>`. The dataset reads the full legs table from
`data_path/GPS_data/MOBIS_Covid_version_3.zip`, member `tracking/legs.csv`.
It filters implausible/non-Swiss legs, missing identifiers and invalid coordinates;
deduplicates exports and excludes conflicting records. The fixed quality filters select
at least 300m Euclidean distance,
30s–4h duration, observed speed 0.2–3m/s, route/Euclidean ratio 0.8–4 and absolute
endpoint grade at most 0.3. All dates in the archive are included. All eligible legs are used; there is no
sampling or row limit.

Endpoint heights reuse `ElevationEstimator` and `data_path/osm/elevations.csv`.
Missing heights remain missing and are excluded, without arbitrary interpolation.
The dataset calls `ElevationEstimator.run_on_nodes` directly; the estimator
handles cache reuse and missing requests. Repeated endpoints are deduplicated
before this call. The dataset writes `walking_legs.parquet`. No intermediate terrain points are requested.

The detour factor is the training median GPS length / Euclidean distance, floored
at one, or an explicit `calibration.walking_speed.detour_factor >= 1`.
The same factor is used to calculate slope, fit speeds and predict times.
Participants are split reproducibly with `random_seed` (10% test by default); test legs cannot affect
the detour factor, flat average or fitted slope coefficients.

Estimation writes `model.json`, `equation.txt`, `predictions.parquet`,
`metrics.csv`, `validation.png` and `speed_vs_slope.png` to the stage cache.
The final helper `plot_speed_curve(parameters, output)` can also plot stored
parameters without estimation. It shades the flat zone.

## Current defaults: full dataset

Refitted on 2026-11-12 using `config_dib_10.yml` paths and seed 518745.
All dates in MOBIS were included, without sampling. Quality filters retained
791,900 legs from 4,567 participants. Training used
709,772 legs/4,110 participants, and testing used
82,128 legs/457 participants.

The stored defaults returned when estimation is disabled are:

```python
{
    "v0_m_s": 1.4299713498762905,
    "k": 3.1589037650039673,
    "uphill_penalty": 0.869597698757856,
    "flat_grade": 0.002,
    "detour_factor": 1.3106159889893567
}
```

Flat speed is the arithmetic mean of 112,599 training
legs inside ±0.2% slope: 1.4300m/s
(5.15km/h). Held-out MAE is 307.5s, RMSE
584.3s and median absolute percentage error
28.49%.

Results, plots, predictions and `fit_summary.json` are saved in
`walking_speed_full/` under the configured analysis directory. Previous pilot
artifacts are preserved. Re-estimation returns new coefficients; updating the
stored defaults in source remains a separate deliberate step.
