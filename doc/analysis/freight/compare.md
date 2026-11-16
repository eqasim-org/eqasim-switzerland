# NPVM versus legacy freight comparison

Generate the report using the chosen pipeline configuration:

```bash
uv run -m analysis.freight.compare config_dib_10.yml
```

Output: `<analysis_path>/freight_comparison/comparison.html`, accompanied by CSV
aggregates and `provenance.json`. The HTML embeds Plotly, Leaflet, canton outlines
and all chart/map statistics. Only the swisstopo background tiles need internet
access when opening it. The background opacity can be adjusted; maps and charts
remain usable when tiles fail. Keep the CSV files beside the HTML for its download
links to work.

The standalone command resolves exact Synpp hashes using the supplied configuration.
It reads existing prerequisite caches and executes only the NPVM freight stages,
the legacy freight wrapper and the comparison code. It never executes the
configuration's production run list, launches population/MATSim stages, or writes
to upstream caches. Missing prerequisite caches raise an error identifying the
required input. In particular, legacy GTE/GQGV coordinates are read from their
existing configured caches, while NPVM freight is generated from the current
configured matrix files. These differences in input provenance are recorded.

The module is also a regular Synpp stage:

```yaml
run:
  - analysis.freight.compare
```

Run it with `uv run -m synpp your_config.yml` to let Synpp build dependencies as
usual. Both methods are compared regardless of production `use_npvm_freight`:
NPVM is requested directly and the legacy wrapper is explicitly parameterized
with `use_npvm_freight: false`.

## Contents and controls

- Total sampled vehicle trips and NPVM-minus-legacy differences.
- Total straight-line OD distance (sum across all valid trips in the selected
  scope), with legacy, NPVM and difference cards plus summary/CSV fields.
- Domestic, one-foreign-endpoint and two-foreign-endpoint source-OD composition.
- Common 5 km straight-line distance histograms and cumulative distributions;
  mean, median, 90th/95th/99th percentiles and zero/invalid-distance counts.
- Departure-hour distributions, preserving hours beyond 24.
- Side-by-side canton maps with a shared scale, origin/destination selector,
  trip-endpoint/distinct-location selector and optional 5 km location cells.
- A signed NPVM-minus-legacy difference map on all published domestic NPVM
  zones, an origin/destination trip selector, a searchable difference table,
  and a complete `npvm_zone_counts.csv`.
- Canton bar charts, NPVM-minus-legacy bars, and a numerical table including
  external, unassigned domestic and invalid endpoint categories.
- Aggregate CSV downloads: summary, canton counts, distance bins, departure
  hours, GTE/GQGV and LW/LZ totals, canton OD counts and location cells.

The traffic-scope selector filters all spatial and distribution views. The
composition chart deliberately retains all traffic categories as a reference.
Distance/departure distributions can show counts or percentages within each method.
Cumulative curves are represented by 1,001 quantiles of all valid distances.
The histogram includes 500+ km as an overflow bin. Empty scopes stay usable.

## Interpretation

Counts describe sampled trip records, not unique physical trucks, delivery tours
or an expansion to the full population. Distinct locations are unique generated
coordinate pairs rounded to 1 mm. Foreign locations are border proxies, not
foreign establishments. Optional 5 km cells aggregate every valid endpoint without
additional sampling; grey cells identify foreign-source endpoints.

Distance is Euclidean distance between generated EPSG:2056 coordinates, **not road
or simulated distance**. Ordinary foreign endpoints in both generators are border proxies, so
international distances generally exclude foreign portions of journeys. NPVM
ODs involving model-domestic Liechtenstein can retain a foreign road access point
when their route never crosses Switzerland. These are not Swiss border crossings.
Compare domestic-only distances when assessing within-Switzerland placement.

Canton assignment uses generated coordinates only for domestic-source endpoints.
Original source-zone IDs identify foreign endpoints, which stay **External** even
when their border coordinates touch a Swiss canton. The report retains unassigned
domestic endpoints separately, including NPVM model-domestic zones in Liechtenstein.
If an endpoint intersects multiple canton boundaries, the lowest canton ID wins
once; such ambiguities are counted. No nearest-canton assignment fills genuine
unmatched points. Canton endpoint counts plus the external/unassigned/invalid
categories always sum to each method's total trips. Same-canton trips contribute
one origin and one destination to their respective views.

Domestic/cross-border/transit scopes describe the original OD-zone classification.
Two-foreign-endpoint ODs are not proof that a subsequently simulated route crosses
Switzerland. The NPVM model-domestic zone layer also includes Liechtenstein and
some enclaves, while legacy domestic identity uses Swiss NUTS IDs.

Both methods use the same freight departure-time profile. Differences in departure
histograms are mainly sampling variation, not independent observed schedules.

The configured scenarios need not have equal year/scaling assumptions. Legacy
freight applies its survey-based projections when `enable_scaling` is true; NPVM
uses supplied daily matrix demand (default release: 2023), multiplied by
`npvm_freight_demand_factor`. Both use the same configured `input_downsampling`.
The report makes no additional adjustment to equalize totals or target years;
its notice and source information show the assumptions explicitly.

## NPVM zone difference map and total distance

The NPVM difference map uses the full configured domestic NPVM polygon layer,
including zero-difference zones. NPVM counts retain native domestic source-zone IDs; legacy
counts assign each generated domestic endpoint to the same polygons. A point on
multiple polygon boundaries is assigned once to the lowest numeric zone ID.
Foreign source endpoints stay External; they are not reassigned to Swiss border
zones. Domestic points/IDs without a matching NPVM polygon stay Unassigned.
Thus, zone endpoint counts plus External, Unassigned and Invalid always sum to
the selected trip total. The domestic NPVM map includes Liechtenstein/enclaves,
which can have no Swiss canton assignment in the canton maps.

The canton maps retain their shared origin/destination and trip-endpoint/distinct-
coordinate controls. The NPVM map has its own **Difference based on** selector:
origin trips or destination trips. It always shows **NPVM minus legacy trip
counts**, independently of the canton measure selection. Blue means more NPVM
trips, orange means fewer, and neutral means zero difference. Both signs use one
symmetric range based on the largest absolute difference in the selected scope;
fill opacity follows the square root of magnitude. Traffic scope and background
opacity apply to all three maps; location-cell overlays apply to the canton maps.
Hover or click an NPVM zone for the signed difference and underlying method counts.
The zone table shows up to 100 matching zones with nonzero counts, sorted by
absolute difference, plus the three special categories. The table foregrounds
differences; the CSV retains complete counts for every zone and scope.

`total_distance_km` is the sum of the same individual distances used in the
histogram. Zero-length trips contribute zero; invalid distances are excluded and
counted separately. It is not multiplied by the inverse sample fraction and is
not routed vehicle-kilometres. The total for all traffic equals the sum of the
three disjoint source-OD scopes (domestic, cross-border and transit).
