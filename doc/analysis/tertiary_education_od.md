# Pooled tertiary OD matrix and sample support

The new `data.od.tertiary_education_matrix` stage pools current education codes
9–13. It does not segment by UH/FH/PH/higher vocational type and does not discard
unresolved institution classifications. Reported foreign study and unusable or
foreign departure/destination zones are excluded using the shared education OD
preparation function.

Education origins use `DEPFORSCHOOLMUN` and `DEPFORSCHOOLQUARTER`, harmonised to
the pipeline's municipality and quarter IDs. Registered `RES_MUN`/`RES_QUARTER`
remain in the processed survey's `home_*` columns for population and work models.
Only the education OD preparation interface aliases actual departure IDs to its
existing `home_*` origin columns. Unknown departures are excluded without a
registered-residence fallback; reported foreign departure countries are also
excluded. The survey retains departure code, country, main-residence category,
origin status and a flag for departures from a different municipality.

The stage returns `(pdf_matrix, cdf_matrix)` in `data.spatial.zones` row/column
order, following the existing matrix interface. The initial empirical audit below
measured sample support before smoothing. Municipality/quarter rows now receive
[geographic and parent-origin smoothing](tertiary_education_od_smoothing.md).
The existing general education stage
continues to use its own-zone fallback. Tertiary destination assignment consumes
the pooled matrix and preserves campus calibration through controlled rounding.

## Reproduce the audit

```bash
python -m analysis.tertiary_education_od \
    --data-path /path/to/pipeline \
    --output-path /path/to/tertiary_education_od
```

The analysis also supports the existing `configure`/`execute` pipeline structure.
The direct command runs the actual survey reader, processed survey and spatial
stages. It uses the existing data-based municipality-type stage rather than
fitting the optional municipality-type model; this choice does not affect OD
zone IDs. Source municipalities are normalized to the repository's reference
geography and quarters use its existing quarter mapping.

The reviewed artifacts are at `/private/tmp/tertiary_education_od`, including the
compressed matrix, zone IDs, observed cells and origin/destination support tables.
No individual respondent records are exported.

## Usable observations

| Year | Usable domestic tertiary OD observations |
| --- | ---: |
| 2021 | 11,929 |
| 2022 | 11,768 |
| 2023 | 11,401 |
| Pooled | 35,098 |

Annual expansion weights are combined before row normalization. Their pooled
totals are not enrolment targets or one-year population totals. The codebook
interpretation for 2021–2022 remains the documented historical assumption.

## Origin support across all three years

| Origin level | Catalog origins | Zero observations | 1–4 observations | 5–9 observations | 10+ observations | Median observations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Municipality | 2,162 | 238 | 742 | 439 | 743 | 5 |
| Quarter | 239 | 8 | 32 | 22 | 177 | 23 |

These counts cover the complete municipality/quarter zone catalog. Zero-support
zones may include places without synthetic tertiary students, so not every zero
necessarily requires an operational fallback. Municipality origins include
respondents assigned to quarters within them; the two levels must not be added
as independent observations.

The weighted effective-sample diagnostic is `(sum w)^2 / sum(w^2)`. Among supported
origins, it is below five for 851 municipalities and 38 quarters. This diagnoses
weight concentration, not the full survey-design precision or a confidence
interval. Five observations is a reporting threshold, not an adopted smoothing
parameter.

## Observed cell support

| Origin level | Observed OD cells | Cells with one respondent | Cells with 1–4 respondents |
| --- | ---: | ---: | ---: |
| Municipality | 17,125 | 12,495 | 16,165 |
| Quarter | 3,409 | 2,250 | 3,118 |

Even after pooling, most observed cells have few respondents. Observed zeros
cannot generally be interpreted as impossible flows. Geographic and institution
compatibility must be distinguished from a lack of survey observations.

## Validation and next step

The real-data matrices have shape 8,856 by 8,856, matching the full zone catalog;
2,155 origin rows have observations. Supported probabilities are finite,
nonnegative and sum to one. Their CDFs are monotonic and end at one. In that
initial matrix, unsupported rows remained NaN and agreed with the support table.
The new smoothing supplies valid distributions for all domestic origins.
Municipality cell counts preserve the 35,098 usable observations.

Checks also covered exclusion of secondary and foreign study, inclusion of
unresolved tertiary institutions, object-typed zone IDs from the processed
survey, and the existing general matrix's empty-origin behavior. Its results
match the original implementation on domestic observations.

Sparse-origin backoff/smoothing is now implemented and validated by holding out
each survey year. The next step is institution/campus grouping and student
budgets before integration into destination choice. Unsupported tertiary students
are not automatically assigned to their home zone.
