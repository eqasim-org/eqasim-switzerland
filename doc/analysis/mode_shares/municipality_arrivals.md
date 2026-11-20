# Municipality arrival mode shares

Generate the interactive comparison for Geneva and Vaud from an already analysed
simulation snapshot:

```bash
uv run python -m analysis.mode_shares.municipality_arrivals \
  --config config_dib_10.yml \
  --simulation-path /path/to/simulation_output
```

The snapshot must contain `output_persons.csv.gz` and
`mode_shares/snapshot_mode_shares_sources.json` from the snapshot mode-share
analysis. The latter identifies the MATSim trips file and cached Microcensus
inputs. Inputs and caches are read without modification.

The script writes `municipality_arrivals_mode_shares.html`, `.json` and `.csv`
under the snapshot's `mode_shares/` directory. The HTML embeds the polygons,
statistics and Leaflet library. Only the optional swisstopo background requires
internet access; its opacity is adjustable. Clicking a municipality opens grouped
mode-share bars, exact shares, trip counts and unique-person counts. Map colours
can represent a mode share or the Microcensus sample size.

## Populations and filters

- Microcensus: valid weekday trips by Swiss residents, weighted by person weight.
- MATSim Swiss: personal trips by residents with home canton ID 1–26.
- MATSim all: personal trips, also including French and cross-border travellers.
  Freight is excluded.

All groups retain only positive, finite distances and the five modes car, public
transport, walking, cycling and car passenger. Loop variants use their base mode;
remote/transit walking uses walking. Origins and activity purposes are unrestricted.
MATSim shares count each retained trip once. Counts describe the simulation sample,
without expansion to the full population.

“Microcensus people” counts unique respondents with at least one retained trip
**ending** in the municipality, rather than residents living there. Someone can
therefore appear in several municipalities. Trip counts are unweighted sample
counts; only the survey shares use person weights. Municipalities without survey
arrivals have unavailable shares, displayed as dashes rather than zero.

Destinations are assigned to the official 2023 municipal boundaries in the
configured data directory: 45 Geneva and 300 Vaud municipalities. Shared-boundary
points are assigned once, to the lowest BFS municipality ID.

The generated iteration-100 snapshot comparison was checked against the earlier
Geneva-city analysis: 5,392 Microcensus trips from 1,967 respondents and 70,004
simulated Swiss-resident arrivals, with the same mode shares.
