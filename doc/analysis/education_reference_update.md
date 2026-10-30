# Maintaining education reference files

Use `python -m analysis.education_reference_update` to prepare a **new candidate
directory** and review it against a STATENT registry. This is a standalone
maintenance tool, not a population pipeline stage. It never changes the active
configuration or promotes a candidate automatically. Keep generated CSVs, raw
responses and reports in the external data directory, outside Git.

## What A–D mean

| File | Purpose and use | Origin | How to update |
| --- | --- | --- | --- |
| **A — `education_institution_links.csv`** | Links a named BFS institution to an anonymous STATENT enterprise. Used in enrolment linking and campus targets; it affects assignment. | Project-prepared crosswalk from our institution review, using STATENT enterprise/site structure and official campus addresses. Spatial agreement does **not** independently verify an anonymous identity. | Carry forward as a candidate; check identifiers, institution types, locations and evidence against the new extract. Edit only after reviewing evidence. Keep `proposed` unless independently verified. |
| **B — `education_site_eligibility.csv`** | Explicit site/programme exceptions to default NOGA compatibility, including regular attendance versus placements. Used in campus weighting and assignment. | Project modelling decisions informed by official descriptions of site use. Not an official national eligibility register. | Review each exception against the target STATENT site and its current use. An absent university is not automatically excluded: this is an exceptions file, not a university inventory. |
| **C — `education_higher_vocational_enrolment.csv`** | National higher-vocational enrolment by year and three source groups; joins university totals in the pooled tertiary reference. Does not supply named institutional/campus counts. | Derived from official BFS table `px-x-1502030000_112`. | Rebuild from an archived official response with the converter described below. Never infer missing counts as zero. |
| **D — `education_reference_sites.csv`** | Named campus/address anchors for institution matching audits and maps. Supplies spatial evidence, not assignment capacities or campus boundaries. | Project-prepared official campus addresses, geocoded through swisstopo; rows retain source URLs, coordinate evidence and review dates. | Check official address/use, inspect the geocoding, update the evidence and review date. Review distance to STATENT sites; do not automatically replace an enterprise with its nearest neighbour. |

A, B and D were assembled during this project's education work; they are not
official downloads bearing those filenames. Preserve their row-level evidence
and assumptions rather than presenting them as BFS-verified crosswalks.

The reference directory inspected on **21 October 2026** contained 5 links in A
(EPFL, UNIL, ZHAW, ETH Zürich and UZH, all `proposed`), 7 rules in B and 21 points
in D. Its C reference was 2023/24; the cached registry used for validation was
STATENT 2022. These are distinct vintages: the tool records them separately. It
does not invent original retrieval dates for legacy files. This maintenance
work does not expand institution coverage or rebuild file E (GWR residences).

## Prepare an annual candidate

Run from the repository using its Python environment. Replace the example
paths with external data paths. The output directory must **not already exist**.

```bash
uv run python -m analysis.education_reference_update prepare \
  --reference-path /data/pipeline/education_enrolment \
  --output-path /data/education_references/candidate_2024_25 \
  --years 2023/24 2024/25 \
  --download
```

Choose explicit academic years published in the source; there is no automatic
“latest year” selection. Preparation copies A/B/D byte unchanged, downloads the
two existing university tables plus the higher-vocational table, and generates C.
For each table it saves `.metadata.json`, `.query.json` and the `.json` response.
The existing university downloader remains unchanged by default; this command
explicitly opts into the additional higher-vocational table.

Official table pages:

- [University institutions (UH), `px-x-1502040100_108`](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-1502040100_108/-/px-x-1502040100_108.px/).
- [Applied sciences and teacher education (FH/PH), `px-x-1502040400_136`](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-1502040400_136/-/px-x-1502040400_136.px/).
- [Higher vocational education, `px-x-1502030000_112`](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-1502030000_112/-/px-x-1502030000_112.px/).

Additional outputs:

- `manifest.json`: official endpoints, requested years, retrieval timestamps and
  response hashes.
- `reference_manifest.json`: source-file hashes, tool-source hashes, requested
  years, operation status and output hashes. Copied A/B/D are explicitly **not
  newly verified**. Failed preparation is marked `failed`, not usable.
- `enrolment_totals.csv`: yearly UH, FH/PH and higher-vocational totals, using
  the production pooled-reference helper. These include published continuing
  education categories; they are enrolment stocks, not daily attendance or
  domestic-resident quotas.
- `higher_vocational_conservation.csv`: mapped versus published national totals.
- `higher_vocational_previous_comparison.csv`: comparison to existing C where
  available. Missing years stay missing rather than becoming zeros.

### How C is generated

The query selects the **total only** for field, sex and nationality. It selects
the education types separately, retaining the published type total solely for
a conservation check:

| BFS `Bildungstyp` | C study level |
| --- | --- |
| `311` Höhere Fachschulen | `higher_vocational_college` |
| `312` Vorbereitung auf Berufsprüfung + `313` Vorbereitung auf höhere Fachprüfung | `higher_professional_examination` |
| `315` Nicht BBG-reglementierte höhere Berufsbildung | `higher_vocational_non_bbg_regulated` |

Historic categories `310` and `314` must be explicitly zero. Nonzero historic
categories, changed definitions, missing/suppressed cells, invalid counts or a
sum different from the published total cause failure and require a mapping
review. The examination group measures reported preparation-course enrolment
with incomplete coverage, **not examination awards**.

### Reproduce without network access

```bash
uv run python -m analysis.education_reference_update prepare \
  --reference-path /data/pipeline/education_enrolment \
  --snapshot-path /data/education_references/candidate_2024_25 \
  --output-path /data/education_references/replayed_2024_25 \
  --years 2023/24 2024/25
```

Supply the same set of years as the frozen cubes. Offline replay requires all
three cubes, their metadata and queries; a legacy hand-aggregated C alone cannot
reconstruct its raw source. It preserves available original manifests as
`source_manifest.json` and `source_reference_manifest.json`. A/B/D come from
`--reference-path`, allowing their versions to be selected explicitly. Byte-identical
CSV output was checked with the same code and Python environment; retain the
repository revision and dependency lockfile alongside archived runs.

## Review against STATENT before use

First obtain the target registry export from the existing
`analysis.education_institutions` stage. Its output under
`<analysis_path>/education_institutions/` includes `institutions.csv`,
`establishments.csv` and `compatibility.csv`. These must come from the intended
STATENT extract. The review command does not rebuild a registry or run Synpp.

```bash
uv run python -m analysis.education_reference_update review \
  --reference-path /data/education_references/candidate_2024_25 \
  --registry-path /data/analysis/education_institutions \
  --previous-registry-path /data/education_references/previous_review/registry \
  --output-path /data/education_references/review_2024_25 \
  --academic-year 2024/25
```

On the first review omit `--previous-registry-path`. The report then explicitly
states that historical identity/location drift cannot be assessed. Each report
archives its target registry under `registry/` for the next review.

Checks include:

- Duplicate keys, missing evidence, missing enterprises/sites/BFS categories,
  incompatible institution types and invalid eligibility rules.
- Aggregate-provider BFS categories that cannot identify one enterprise, and
  proposed identities that must not silently become verified.
- Distance between D anchors and linked, type-compatible establishments, plus
  the closest alternative enterprise for inspection. Default
  `--review-distance 500` is in metres and controls this check only; it is not
  a campus boundary, assignment radius or automatic linking rule.
- New/removed sites and changes in enterprise IDs, type, local/enterprise NOGA
  and coordinates relative to the previous registry. Every relevant coordinate
  movement is exposed, even below 500 m. Identifier stability alone is not proof
  of identity. Unchanged IDs do not certify unchanged real-world ownership/use.
- Selected-year C schema, counts and agreement with its frozen official cube.
  Legacy files lacking the raw cube receive a review notice.

Inspect `report.html`, `review_issues.csv`, `spatial_review.csv`,
`registry_changes.csv` and `review_manifest.json`. Structural errors produce
CLI exit code **2** and status `blocked`. Review notices produce exit code **0**
but status `requires_human_review`; exit zero does not certify a reference bundle.
Missing input files or an unusable official snapshot can fail before a report
is written. No issue is automatically repaired and no input file is rewritten.

For A/B/D corrections, retain the original bundle, edit a separate candidate,
record the evidence and dates, then run review into another new output directory.
Do not simply rename a nearby STATENT site or enlarge the radius to clear a flag.

## Activate only after review

No production configuration was changed by implementing these tools. When a
candidate has been reviewed, explicitly point the existing parameters at that
version:

```yaml
education_enrolment_year: "2024/25"
education_enrolment_path: /data/education_references/candidate_2024_25
education_institution_links: /data/education_references/candidate_2024_25/education_institution_links.csv
education_site_eligibility: /data/education_references/candidate_2024_25/education_site_eligibility.csv
education_higher_vocational_reference: /data/education_references/candidate_2024_25/education_higher_vocational_enrolment.csv
education_reference_sites: /data/education_references/candidate_2024_25/education_reference_sites.csv
```

The separate file parameters do not automatically follow a change to
`education_enrolment_path`. This tool adds **no new production parameters**.
After deliberate activation, rerun the affected pipeline stages and inspect
enrolment/campus/assignment diagnostics before accepting the new model results.

## Validation of this implementation

On 21 October 2026, an isolated download and offline replay gave:

| Academic year | UH | FH/PH | Higher vocational |
| --- | ---: | ---: | ---: |
| 2023/24 | 169,249 | 107,196 | 60,854 |
| 2024/25 | 172,242 | 109,092 | 60,377 |

All three 2023/24 C groups matched the existing reference exactly. A/B/D were
unchanged, and A/B/C/D were byte-identical between download and offline replay.
Review against the cached 2022 registry (1,101 establishments) produced zero
structural errors and seven review notices: five already-proposed institution
identities, one missing historical registry baseline, and ZHAW's Zurich
Lagerstrasse anchor approximately 2.14 km from its nearest linked establishment.
That anchor requires investigation; it was not remapped. Neither the candidate
2024/25 data nor any changed assignment was activated.
