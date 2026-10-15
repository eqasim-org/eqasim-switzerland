# BFS institution enrolment and STATENT links

`data.statent.education_enrolment` follows the existing `configure`/`execute`
structure. It reads local BFS JSON-stat2 snapshots and the institution mapping;
the pipeline itself makes no network requests. `analysis.education_enrolment`
downloads reproducible snapshots and reports enrolment and mapping coverage.
Neither stage changes student assignment or campus eligibility.

## Inputs and definitions

Sources:

- [UH enrolment by year, subject, level, educational origin and institution](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-1502040100_108/px-x-1502040100_108.px).
- [FH/PH enrolment by year, subject, level, sex and institution](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-1502040400_136/px-x-1502040400_136.px).

The download selects all categories of every non-year dimension. The reader
checks category completeness against the saved metadata before summing subjects
and educational-origin/sex categories. Null cells remain unknown, rather than
being treated as zero. The query, metadata, retrieval time and response SHA-256
are saved alongside each response. Snapshot category codes are table-specific;
labels are retained for review.

Snapshots cover 2021/22, 2022/23 and 2023/24, aligning with the structural-survey
years. The reference is **2023/24**, which reports ZHAW, ZHdK, HWZ and EHSM
separately following changes to the BFS reporting structure. STATENT remains
2022 and the OD observations remain pooled across 2021–2023. This setting is
explicit and does not automatically scale totals to the scenario's scaling year.

Raw Bachelor, Master, doctorate, legacy degrees and continuing education/other
categories remain separate in `enrolment` and `linked_enrolment`. The derived
`model_enrolment` and `linked_model_enrolment` tables align with the supplied
2023 structural-survey codebook (sheet 25): FH/PH diplomas join Bachelor/code 11;
university/ETH licentiates and diplomas join Master/code 12. Each institution's
categories are combined once before linking and FTE campus splitting. Strict
and legacy contributions and source labels are retained for review. Campus/type
shares use this same aligned reference in both numerator and denominator.
Continuing education stays excluded; no study-level prediction is changed.
In particular,
the BFS doctorate category is not proof of a separate habilitation enrolment
count, despite our shared internal survey label `doctorate_habilitation`.

These are institution enrolments, not student commuting trips, campus capacity,
current Swiss-resident counts or calibrated assignment targets. Educational
origin cannot identify present residence abroad. UH/FH/PH do not cover higher
vocational education.

## Institution mapping

`<data_path>/education_enrolment/education_institution_links.csv` contains source institution type,
BFS institution code, anonymous enterprise ID, name, evidence URL/note and
`proposed` or `verified` status. EPFL, UNIL and ZHAW are spatially reviewed
proposals (see `analysis/education_institution_matching.md`). Public campus
addresses support the location interpretation but do
not prove the identities of anonymous source enterprises; neither proposal is
marked verified.

Proposed links appear in the audit, but cannot populate
`verified_enrolled_students`. Unknown institutions and enterprise IDs, incompatible
types, duplicate links, multiple verified institutions for one enterprise and
multiple verified enterprises for one institution raise errors. Mapping an
institution to several legal enterprises needs an explicit allocation rule before
verification; its total must never be copied to each group. Verified links require
a name and evidence. Campus allocations are deferred.

The FH/PH table is not a complete list of individual providers. `Andere PH/Inst.`
and `Andere FH` are aggregates, and some named categories are umbrella institutions
(e.g. HES-SO and ZFH). Aggregates cannot be verified as one institution. Historic
categories remain in the source and can have zero enrolment in recent years.

## Reference-year results

| Type | BFS categories | Positive categories | All-level enrolment | Bachelor/master/doctorate enrolment |
| --- | ---: | ---: | ---: | ---: |
| UH | 17 | 14 | 169,249 | 158,880 |
| FH/PH | 19 | 14 | 107,196 | 97,786 |

There are no missing source cells in these snapshots. Of the 189 STATENT groups,
three have proposed links and none has a verified link. All 189 therefore remain
unresolved for automatic allocation. Counts by institution are calculated once,
independently of how many proposed links it has.

| EPFL, 2023/24 | Students |
| --- | ---: |
| Bachelor | 6,773 |
| Master | 3,986 |
| Doctorate | 2,306 |
| Continuing education/other | 572 |
| All levels | 13,637 |
| Current model levels only | 13,065 |

These EPFL counts cover the institution, not only Lausanne/Ecublens. They must
not be used directly as that campus's target.

The table above reports strict source degree categories. The aligned national
reference is 160,817 Bachelor (including 2,383 FH/PH diplomas), 72,884 Master
(including 1,718 UH legacy degrees), and 27,066 doctorate enrolments.
`institution_totals` preserves `enrolled_students` for all source levels and
`strict_degree_enrolled_students` separately; `model_level_enrolled_students`
uses the aligned definition. Doctoral and all-level enrolment are unchanged.

## Run and outputs

Default input directory: `<data_path>/education_enrolment`. Optional configuration
`education_enrolment_path` overrides it; `education_enrolment_year` selects the
reference year; `education_institution_links` selects the mapping file. Use an
absolute mapping path when running the pipeline outside the repository root.

```bash
python -m analysis.education_enrolment \
  --input-path /path/to/input/education_enrolment \
  --registry-path /path/to/education_institutions \
  --output-path /path/to/education_enrolment_audit \
  --download
```

Omit `--download` to reuse local snapshots. `--academic-year` chooses the reference
year; `--years` chooses the download years. The registry path must contain the
previous campus audit's `institutions.csv`. Standard pipeline execution uses
`data.statent.education_institutions` instead of this CSV.

Outputs: `enrolment.csv` (all downloaded years), `linked_enrolment.csv` (reference
year and candidate evidence), `institution_totals.csv`, `statent_links.csv`,
`coverage.csv`, `unmatched_bfs_institutions.csv`, `unmatched_statent_groups.csv`,
and a README. Unmatched reports retain proposed links until verified.

Validation checked source count conservation in all six table/year slices,
JSON-stat axis reordering and sparse values, missing-cell propagation, unsupported
year rejection, exclusion of proposed links from verified counts, and rejection
of duplicated verified allocation. The downloader and audit ran against live BFS
tables and the existing 189-group registry; no synthetic assignments were made.

Next: confirm institution identities or obtain an enterprise crosswalk, including
umbrella-provider structures. Then review study-level campus eligibility (including
doctoral research sites), reconcile totals with the simulated population, and
estimate campus shares using FTE where observed campus enrolment is unavailable.

## External reference files

Reference CSVs are not tracked in Git. Store `education_institution_links.csv`,
`education_reference_sites.csv`, `education_site_eligibility.csv` and
`education_higher_vocational_enrolment.csv` under `<data_path>/education_enrolment/`.
These defaults can be overridden by `education_institution_links`,
`education_reference_sites`, `education_site_eligibility` and
`education_higher_vocational_reference`. Standalone audit commands require explicit
reference paths through their `--mapping-path`, `--reference-path` and
`--eligibility-path` options where applicable.
