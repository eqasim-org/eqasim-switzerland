# Scaling from official raw BFS projections

Scaling now reads unchanged BFS CSV responses under
`<data_path>/projections/raw/`. The pipeline does not download data automatically.
The previous `population_year_end_2025.csv` and `households_2025.csv` inputs are
no longer read. They can be retained as historical comparison files.

## Obtain and share the inputs

Download a new snapshot explicitly, from the repository root:

```bash
uv run python -m data.statpop.projections.download \
  --output-path /path/to/data/projections/raw
```

The output directory must not already exist. To refresh references, download
into a separate candidate directory and review it before replacing the active
snapshot. The downloader selects population years 2024–2055 by default;
`--start-year` and `--end-year` can select a smaller range within the published
edition. Household projection controls are available for 2025–2055.
The downloader is intentionally tied to the 2025 reference editions; a new
edition requires reviewing the table identity and scenario assumptions.

Share the entire directory:

```text
projections/raw/
    px-x-0104020000_101.metadata.json
    px-x-0104020000_101_2024_2027.csv
    px-x-0104020000_101_2024_2027.query.json
    ... remaining population year chunks and queries ...
    DF_SCEN_MENAGES.csv
    manifest.json
```

Each CSV is the original HTTP response payload, without rewriting numbers,
labels, delimiters or encoding. The population responses are comma-delimited
Windows-1252; the household response is UTF-8 SDMX CSV. Population requests use
four-year chunks to respect the API cell limit. Queries, original population
metadata, URLs, content types, response hashes and retrieval time are archived.
Failed downloads are marked unusable in the manifest and are not accepted by
the pipeline. The raw files and generated reports remain outside Git.

The official sources are:

- [BFS cantonal population projections, reference scenario AR-00-2025](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-0104020000_101/-/px-x-0104020000_101.px/).
  Requests select all 26 cantons, Swiss/foreign nationality, both sexes, ages
  0–100+, and **population on 31 December**. Totals and other demographic
  measures are excluded at the API query, without modifying returned cells.
- [BFS household scenarios, DF_SCEN_MENAGES](https://stats.swiss/vis?lc=en&df%5Bds%5D=disseminate&df%5Bid%5D=DF_SCEN_MENAGES&df%5Bag%5D=CH1.SCEN&df%5Bvs%5D=1.0.0).
  The SDMX request selects scenario AM-00-2025, total household size, annual
  projections, and retains both household and person measures. The processing
  stage selects private-household counts explicitly.

## Pipeline processing

With `enable_scaling: true`, both existing projection stages depend on
`data.statpop.projections.raw`. This new offline stage:

1. Validates required files and any download manifest; hashes the raw CSVs for
   cache invalidation, rejecting edits inconsistent with the saved manifest.
2. Decodes the official formats and combines non-overlapping population chunks.
   Missing year-end headers, duplicate cells and unexpected household dataflows
   are rejected.
3. Returns the two source tables and writes normalized `population.csv` and
   `households.csv` into its own cache for inspection. External inputs stay intact.

`data.statpop.projections.population` then selects `scaling_year`, validates full
age/sex/nationality/canton coverage, maps labels and aggregates the existing age
classes. `data.statpop.projections.households` selects the same year, scenario
AM-00-2025 and measure `MEN`, excluding national totals and requiring one target
per canton. Their return contracts remain `(controls, year)`.

No extra YAML options are required:

```yaml
config:
    enable_scaling: true
    scaling_year: 2025
```

With scaling disabled, the raw stage is not registered as a dependency and no
projection files are required. There is no silent fallback to older prepared
inputs or the AM-00-2020 Excel workbook. To use manually downloaded exports,
keep the documented raw CSV schemas and filenames; a single population export
named `px-x-0104020000_101.csv` is also accepted. A manifest is optional for such
manual exports, but if present must be complete and match the raw CSV hashes.

## Validation of this migration

The official exports downloaded on 30 October 2026 reproduce all **336,128**
population cells in the previous input exactly. Both population and household
control tables match the previous implementation for every year **2025–2055**.
The 2025 targets remain **9,110,251 people** and **4,101,264 private households**.
The migration changes input provenance and processing, not the scaling targets
or household fitting method. Local fixtures cover unchanged source bytes,
chunk overlaps, wrong measures, missing inputs, manifest changes and disabled
scaling, alongside existing population-scaling regressions.
