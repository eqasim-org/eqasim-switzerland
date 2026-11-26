# Swiss-resident workplace assignment

Each employed resident receives one primary work location. This stock includes people who stay home on the simulated day. For the survey category “works in different locations”, the location represents the employer base. Residents classified as working at home receive their home coordinates. Other residents draw a STATENT establishment using Structural Survey commute flows by home zone, sex and age. The survey weights determine OD shares. Remaining establishment employment, PT access for workers without a car and the existing fixed-worker distance preference determine the draw within feasible zones.

The domestic OD matrices use the pooled 2021–2023 Structural Surveys. For a fixed worker who reports a separate work-start address (STARTWORK codes 4–6), the pipeline uses the reported municipality or quarter, retaining the employer as a fallback for an unresolved start. An explicitly foreign start is excluded from domestic OD training. An empty demographic OD row first borrows the all-worker row of the same workplace category and origin.

A municipal workplace report covers the entire municipality. Its quota is shared across feasible quarters using observed quarter shares for the workplace category, or available employment when quarter observations are absent. Facilities outside all quarters retain their fraction of municipal employment. A quarter without establishments uses the same municipal distribution. If the municipality has no establishment, its zone's representative point selects the nearest establishment zone. A worker with no usable domestic origin distribution uses the nearest establishment to home.

For French external workers, only commute endpoints in Switzerland reserve employment. Each source record represents `1 / fr_sample_rate` workers. Establishments at a shared coordinate split the reservation in proportion to their remaining employment. Cross-border survey agents and already assigned Swiss workers use `1 / input_downsampling`. The existing 1.4 capacity factor and positive floors still apply. The weights describe employment opportunities and may be exceeded.

## Scenario settings

The following settings enable the three-city scenario tested on the 3% population. Values go under `config` in the scenario YAML.

```yaml
workplace_reference_year: 2023
workplace_unbiased_rounding: true
workplace_calibration_municipalities: [261, 6621, 5586]
```

The reference year defaults to 2023 and selects the home-based-work training year and optional municipal calibration totals. Unbiased rounding defaults to false. Enabling it preserves every origin/segment worker total while allowing low-probability destinations to retain their expected shares over repeated seeds.

The municipality list defaults to empty. Entries are BFS municipality IDs. Fixed and mobile workers are calibrated separately to weighted domestic workplace totals for the reference year, scaled by the Swiss population sample fraction. The fitted multipliers preserve each origin total and its supported destinations. Integer rounding and home-based workers can leave a difference in the final city stock. The calibration applies to a survey-coverage proxy drawn from STATPOP main-residence private households, excluding residents with permits 11 or 12. Other employed residents still receive workplaces. Calibration targets are absolute stocks for the selected year, including when applied to a population projection.

## Repeat the validation

```sh
uv run -m synpp config_dib_10.yml --run analysis.workplace_stock
```

The analysis stage writes municipal and quarter CSVs in its cache, counting each survey-eligible employed resident once. `model` and `survey` are expanded worker stocks. `*_records` are unexpanded counts, and `*_local` counts residents who live and work in the same municipality. Quarter rows include the observed survey weight and an estimate that allocates municipality-only answers in proportion to the reported quarters. The coverage column gives the fraction of municipal weight with a known quarter.

To check an independent year, set `workplace_validation_survey_file` to the single-year Structural Survey person CSV. Optionally set `workplace_validation_quarter_file` to the matching BFS polygon file (e.g. the extracted `quart24.geojson`). The audit then locates model coordinates directly in those polygons. When a city has survey quarter codes absent from the supplied geography, the independent-year audit suppresses its survey quarter estimates and keeps its municipal result. The 2024 boundary release includes 11 cities. [BFS 2024 statistical quarters](https://statistik.admin.ch/de/asset/30487002)

## Validation on the November 2026 3% sample

The comparison holds the 266,847 sampled residents fixed and reruns survey preparation, worker classification, OD construction and workplace assignment. All 139,391 employed agents received one workplace, including 138,097 in the survey-comparable population. Counts expand by 1 / 0.03. Fixed, mobile and home-based workers are included. The baseline is the workplace assignment in the existing prepared scenario. The independent Structural Survey is the 2024 release. It does not supply any calibration margin.

| Absolute error as a share of survey stock in 2024 | Baseline | Branch defaults | Unbiased rounding | Rounding plus city calibration |
|---|---:|---:|---:|---:|
| Zürich, Genève and Lausanne municipalities | 3.56% | 2.91% | 2.22% | 0.37% |
| Other 293 municipalities with at least 100 reported workers | 6.85% | 6.76% | 5.87% | 5.91% |
| 24 neighbouring municipalities meeting the same threshold | 8.20% | 8.14% | 6.81% | 7.80% |
| 162 quarters with at least 50 reported workers and 75% municipal quarter coverage | 10.94% | 10.71% | 9.59% | 9.07% |

Errors are the sum of absolute area errors divided by the sum of survey workplace weights for those areas. The quarter check places model points in the 2024 BFS polygons. Other municipality and neighbouring municipality rows overlap. The last column uses the scenario seed 518745. Two additional assignment seeds gave three-city errors of 0.19% and 0.31%, and quarter errors of 8.96% and 8.84%. The city fit improved each pilot city's internal-versus-inbound commuter split at the scenario seed. Basel and St. Gallen quarter errors increased. Luzern retained an error near 30% despite a close municipal total.

The national fixed-workplace mean and median crow-fly distances in the survey-comparable population moved from 10.59 / 5.71 km to 11.63 / 6.16 km. The weighted 2023 Structural Survey gives 11.44 / 7.02 km. The model's 95th percentile rose from 33.12 to 38.38 km against a survey value of 35.78 km. The home-based classification changed from 8,576 to 7,542 sampled workers.

## Apprentice workplace projection

The production Structural Survey repair uses the configured `vocational_apprentice_repair_fraction` (default 0.65). Within each survey year it reclassifies that fraction of the weighted eligible pool as employed apprentices. Eligible people are students aged 15 or older in vocational programme 6 who report unemployment or inactivity with the corresponding job-position code. The seed determines the selection. The fraction was chosen using the national stock comparison. The register counts training places by employer and the Structural Survey counts residents, so their totals can differ.

To reproduce the workplace pilot, download the three pinned Federal Statistical Office releases once on a machine with Internet access. Use the same `data_path` as the pipeline.

```sh
uv run -m data.structural_survey.apprentice_references --data-path /path/to/pipeline/data
uv run -m synpp config_dib_10.yml --run analysis.apprentices.workplaces
```

The first command archives the original PX and Excel files under `apprentices/fso/` in the data folder. Both the downloader and the analysis verify their SHA-256 digests. The analysis runs offline with these files and the 2021–2024 Structural Survey person files in `structural_survey/`. If the 2024 survey is elsewhere, set `workplace_validation_survey_file` to its path.

`national_apprentice_benchmark.csv` compares the weighted explicit and repaired survey totals with dual-training places for each production survey year. Its `implied_fraction` is the difference between register and explicit survey counts divided by the eligible survey weight. This is a sensitivity check on the configured rate.

The pilot retains reported apprentice statuses and redistributes the national weight of the survey's vocational-student repair across eligible respondents. Weighted training commutes from 2021–2022 supply possible destinations. The repair uses employer-canton shares, distance and inter-canton commuting, with a half-strength regional residence adjustment and a national under-25 margin. Residence controls come from the 2023 Swiss Labour Force Survey. [FSO regional series](https://statistik.admin.ch/de/asset/36890545) · [FSO age series](https://statistik.admin.ch/de/asset/36890553)

For a separate employer-side scenario, the model's survey-eligible apprentice cohort supplies the residential geography and the register supplies the 2023 total and all 26 employer-canton totals. The register counts training places, including those occupied by inbound commuters. Projecting the full stock onto resident homes is therefore a scenario assumption. `employer_cantons.csv` checks the imposed totals. `heldout_workplaces.csv` and `heldout_commutes.csv` compare the projections with explicit 2024 survey reports. `workplace_reference.csv` gives the reported and imputed municipal stocks separately. Its approximate intervals condition on the fitted imputation and omit model and survey-design uncertainty. The stage produces aggregate estimates and leaves simulated persons and establishments as configured by the production pipeline. [FSO vocational training by employer canton](https://statistik.admin.ch/de/asset/35887278)

Run `uv run -m synpp config_dib_10.yml --run analysis.apprentices.calibration` to compare municipal workplace controls after this correction. The stage replays ordinary-worker assignment with no city controls, controls for Genève and Lausanne, and controls for all three cities including Zürich. The first two variants also run with the next random seed. Each replay uses the same 2023 apprentice projection and home-based workers. Its fixed-worker establishments reserve capacity before the moving-worker draw. The ordinary-worker controls exclude apprentices from both the assigned cohort and the survey target. The output `municipal_workplace_comparison.csv` gives expanded stocks and survey references for both years.
