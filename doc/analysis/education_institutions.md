# STATENT tertiary institution and campus registry

The new `data.statent.education_institutions` stage follows the existing
`configure`/`execute` structure. It returns establishment, institution, campus,
campus-zone and nominal programme-compatibility tables. Destination assignment
is not connected to these tables yet.

## Grouping and identifiers

Anonymous source enterprise IDs define institution groups. These are not yet
verified named universities: a legal enterprise can contain multiple providers
or institution types. Institution names remain empty and `name_verified=False`.

Within each enterprise, sites are grouped by DBSCAN with a configurable 500 m
link distance and minimum cluster size one. Separate enterprises never merge,
even at identical coordinates. Connected chains can form clusters wider than
500 m; extent exceeding twice the threshold and type conflicts flag review.
Cluster IDs use the smallest source establishment ID as an anchor, making them
independent of input row order for a fixed membership. They can change if source
membership or the clustering distance changes.

The existing row-based `enterprise_id` used for destinations remains unchanged.
Source enterprise/local IDs and every establishment's destination and municipal
zone links are retained. A campus centroid is descriptive, not a substitute for
its individual site locations or zones.

## Reviewed results

| Registry item | Count |
| --- | ---: |
| Establishments with local tertiary NOGA | 1,029 |
| Additional sites affiliated with a tertiary enterprise | 72 |
| Total establishments retained | 1,101 |
| Anonymous enterprise groups | 189 |
| Provisional campus clusters at 500 m | 359 |
| Nominally eligible teaching establishments | 1,028 |
| Clusters without eligible teaching establishments | 13 |
| Clusters crossing destination zones | 14 |
| Clusters crossing municipality zones | 3 |
| Clusters flagged for grouping review | 11 |

All retained sites map to destination and municipal zones using the existing
STATENT spatial functions. The direct audit imputes only the small candidate
subset, preserving the full reader's original destination IDs. The standard
STATENT stage still reads and imputes all establishments.

Threshold sensitivity:

| Link distance | Campus clusters | Grouping-review clusters |
| --- | ---: | ---: |
| 250 m | 418 | 14 |
| 500 m | 359 | 11 |
| 1,000 m | 328 | 8 |

These are provisional geographic clusters, not a validated count of physical
campuses. The 500 m setting requires inspection, especially the 11 review clusters.
Two enterprise groups contain multiple nominal institution types. One site has
conflicting local and enterprise tertiary types and is excluded from automatic
teaching eligibility pending review.

## Student compatibility

Nominal local-NOGA compatibility is:

| Local institution type | Current study level |
| --- | --- |
| UH | Bachelor, master, doctorate/habilitation |
| FH/PH | Bachelor, master |
| Higher vocational | Higher professional examination, Höhere Fachschule |

This does not establish which programmes a provider actually offers. Ancillary
sites remain in the registry but are not automatically selectable as teaching
destinations. The subsequent campus-weight step adds nominal doctoral compatibility
for research NOGA 721* sites in UH enterprises (enterprise NOGA 854201), with
affiliation assumptions retained. Hospitals, other ancillary sites and cooperative
FH sites still require additional evidence; lack of nominal compatibility is a review
condition, not proof that a reported activity is impossible.

Municipality destinations can select a site in one of their quarters. Quarter
destinations require an exact site-zone match. Multi-zone campuses retain all
site links; no single zone is forced on the campus.

Student capacities at establishment, campus and institution levels remain
missing. Employee and FTE totals are descriptive and are not converted into
student capacities. Verified institution/campus names and enrolment targets are
still needed for calibration.

## Observed and predicted zone coverage

All usable pooled tertiary OD observations remain in the audit:

| Survey year | Usable observations | Without a nominally compatible site in the reported zone |
| --- | ---: | ---: |
| 2021 | 11,929 | 2,101 |
| 2022 | 11,768 | 2,022 |
| 2023 | 11,401 | 2,329 |

The 2023 gap consists of 597 professional-examination cases, 663 Höhere
Fachschule cases, 714 bachelor cases, 284 master cases and 71 doctorate/habilitation
cases. Missing candidates may reflect NOGA limitations, shared/ancillary teaching
sites, programme-specific arrangements, location reporting or zone boundaries.
They are not silently removed or assigned to an incompatible establishment.

The audit also tests the smoothed destination probabilities for every domestic
origin. Median probability mass with a nominally compatible site is 81.19% for
bachelor/master, 57.86% for doctorate/habilitation, and 36.13% for the two higher
vocational levels. These medians across 2,401 origin rows are not population-
weighted student shares. They demonstrate why programme-aware zone checking or
reselection is necessary after drawing from one pooled matrix.

## Reproduce and inspect

```bash
python -m analysis.education_institutions \
    --data-path /path/to/pipeline \
    --output-path /path/to/education_institutions
```

The analysis supports the standard pipeline stage interface as well. The direct
command uses the same smoothing estimator without expanding a dense matrix.
Artifacts are at `/private/tmp/education_institutions`: registry CSVs, cluster
sensitivity, zone/programme candidate lists, observed and smoothed coverage,
and `campuses.html`, an interactive review map. Institution names and capacities
are explicitly unverified. The swisstopo national-map tiles require an internet
connection (https://docs.geo.admin.ch/visualize-data/xyz.html).

Checks verified unique source IDs, unchanged destination IDs for all 1,101
sites, stable clusters under input reordering, separate co-located enterprises,
distant-site separation, ancillary/conflict handling, valid zones and unset
capacities. Establishment counts are conserved across institutional, campus and
campus-zone aggregations; OD counts are conserved in compatibility summaries.

## Next step

Review the flagged clusters and missing-provider zones, link enterprise groups
to verified institution/campus identities, and expand eligibility only where
programme evidence supports it. Then add enrolment targets and define compatible
zone reselection before connecting the assignment stage.
