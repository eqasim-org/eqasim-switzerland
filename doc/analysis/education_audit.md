# Education input audit

Audit of the structural survey extracts for 2021–2023 and STATENT 2022 on
`feature/education-destination-calibration`. This step adds analysis only; it does
not change student classification, destination selection or activity chains.

## Reproduce

From the repository root, using the project's Python environment:

```bash
python -m analysis.education_audit \
    --data-path /path/to/pipeline \
    --output-path /path/to/education_audit
```

The module also follows the existing `configure(context)` / `execute(context)`
stage structure. As a pipeline stage, it writes to
`analysis_path/education_audit`. The direct command avoids rebuilding the spatial
stages. It reads the actual structural survey reader and audits employment before
the processed survey's existing apprenticeship correction.

The reviewed run is available at `/private/tmp/education_audit`. Its CSV files
contain aggregates rather than individual respondent records. All seven survey
cross-tabs preserve annual respondent counts and expanded weight totals.

## Coverage and code definitions

| Survey year | Respondents |
| --- | ---: |
| 2021 | 282,992 |
| 2022 | 286,499 |
| 2023 | 279,164 |
| Total records audited | 848,655 |

Annual weights must not simply be summed and interpreted as one year's
population. Counts below describe these extracts, not exact national enrolment.
Unweighted coverage describes the available observations; it is not a weighted
estimate of the proportion of students attending each institution type.

The institution mapping is supported by the official
[BFS NOGA 2008 classification](https://www.kubb-tool.bfs.admin.ch/fr/noga/2008/8542):

| School NOGA | Institution type |
| --- | --- |
| 854201 | University / ETH (UH) |
| 854202 | University of applied sciences / teacher education (FH/PH) |
| 854203 | Higher vocational education |

FH and PH cannot be separated with this NOGA code alone. Institutional mapping is
needed. The broader prefix `854` also includes post-secondary non-tertiary
education (`8541`), so it should not become the new definition of tertiary study.

The supplied [2023 BFS codebook](../../../../../../Users/balacm/Downloads/do-d-40-se_user-code-2023 (2).xlsx>)
confirms the education code list on sheet 25:

| Code | Education |
| --- | --- |
| 1 | None |
| 2 | Compulsory school attended but not completed |
| 3 | Compulsory school |
| 4 | One-year training / bridging programme |
| 5 | Fachmittelschule / Diplommittelschule |
| 6 | Apprenticeship / vocational school / commercial school |
| 7 | Gymnasiale Maturität / teacher training seminar |
| 8 | Berufs- / Fachmaturität |
| 9 | Professional examination / higher professional examination |
| 10 | Höhere Fachschule (HF) |
| 11 | Bachelor, including FH/PH diplomas |
| 12 | Master, including university/ETH licentiate or diploma |
| 13 | Doctorate / habilitation |

The workbook contains code lists rather than variable-by-variable definitions.
The observed ongoing-education values match this list. Historical applicability
to the 2021–2022 releases has not been independently verified; their observed
values are consistent with the same list. Code 13 must not be interpreted as
exclusively PhD enrolment because its label also includes habilitation. Current
education and highest completed education remain separate variables.

## School institution coverage

For respondents with current education codes 11–13:

| Year | Respondents | Recognized local tertiary NOGA | Recognized enterprise tertiary NOGA | Conflicting recognized types |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 10,295 | 73.73% | 76.08% | 2 |
| 2022 | 10,153 | 73.57% | 77.31% | 4 |
| 2023 | 9,944 | 72.09% | 75.76% | 0 |

The two coverage percentages are separate, not additive. A conflict means both
codes identify one of the three tertiary types but identify different types.
Agreement is a consistency check, not independent confirmation of correctness:
the local and enterprise codes may share register information.

In 2023, 384 respondents had another local NOGA but a recognized tertiary
enterprise NOGA; 360 of these were in a UH enterprise. Another 1,107 had missing
codes at both levels. Thus an enterprise-level fallback may be useful even when
the local code is present but describes an ancillary activity. It should retain
its evidence source and uncertainty rather than overwrite the raw local code.

Conversely, 326 respondents in codes 11–13 had a local higher-vocational NOGA.
This mismatch between degree-study level and institution type needs review
rather than automatic reassignment to higher vocational education.

## Employment and attendance

`IN_EDUCATION` and `CURRACTIVITY_STUDENT` agree for every audited record. They
provide no additional distinction in these extracts; the agreement does not
establish two independent measurements of participation.

| Year | Code 13 respondents | Weighted employed share within code 13 |
| --- | ---: | ---: |
| 2021 | 853 | 74.87% |
| 2022 | 774 | 77.75% |
| 2023 | 766 | 76.49% |

The pattern supports retaining employment and study as separate, overlapping
attributes. It does not prove that every employed student works at their
education campus. The employer NOGA
cross-tab is available for examining that relationship. Highest completed
education must not be substituted for current education.

School frequency exceeds five in 4,163, 4,320 and 4,095 records respectively.
Sheet 46 names its grouped frequency variable "Anzahl Pendelbewegungen pro Woche"
(number of commuting movements per week): category 1 is up to five, category 2 is
six to ten, and category 3 is more than ten. This supports interpreting the
frequency as movements rather than attendance days. The code list does not
establish whether the raw frequency counts one-way or return journeys. That
definition must be checked before deriving daily attendance probabilities.

Sheet 51 confirms the detailed school commuting codes:

| Code | Meaning |
| --- | --- |
| 0 | Does not commute |
| 1 | Commutes within the residential municipality |
| 2 | Commutes within the residential canton |
| 3 | Commutes within Switzerland |
| 4 | Departs from Switzerland and attends education abroad |
| 5 | Departs from abroad and attends education in Switzerland |
| 6 | Commutes abroad |

`COMMUTERTOSCHOOL_DETAIL` remains a diagnostic field, not a mandatory condition
for enrolment or an automatically trusted filter. The commuting departure
location need not be the registered home address: sheet 45 distinguishes
departure from the residential address and another address. Therefore comparing
the code directly with `RES_MUN` is not a complete consistency test.

Sheet S confirms -7 = not assignable, -8 = not applicable, -9 = no information,
and -6 = not assignable according to current boundaries. These statuses must not
be collapsed into "does not commute". The audit's grouped missing-code coverage
does not replace these source-specific meanings.

The official [BFS commuting definitions](https://dam-api.bfs.admin.ch/hub/api/dam/assets/31525916/master)
distinguish education participation from leaving one's residential building to
attend an education location. They also allow a person to count as both a work
and an education commuter. The supplied workbook provides the numeric commuting
codes; neither source establishes their measurement accuracy for each respondent.

## STATENT grouping

| NOGA | Establishments | Distinct anonymous enterprises | Employees |
| --- | ---: | ---: | ---: |
| 854201 | 743 | 42 | 57,105 |
| 854202 | 167 | 59 | 31,280 |
| 854203 | 119 | 90 | 6,495 |

These enterprise groups are not counts of named universities. Establishments
must be grouped into institutions and campuses before interpreting destination
totals. Employment is useful for reviewing site size, not a measured student
capacity.

## Proposed classification rules for the next step

1. Preserve current education level separately from institution type. Use the
   supplied 2023 education code list for labels, with explicit provenance and a
   recorded historical assumption if applying it to 2021–2022. Retain
   doctorate/habilitation as the source category rather than silently narrowing it
   to PhD students.
2. Where local NOGA gives a recognized tertiary type, retain it as local evidence.
   Flag disagreement with a recognized enterprise type for review.
3. Where local NOGA does not identify a tertiary type but enterprise NOGA does,
   retain an enterprise-based candidate type and its evidence source. Degree-level
   compatibility and institution mapping still need checking.
4. Leave unresolved cases explicit. Later imputation should use survey weights,
   age, employment and region, with calibrated totals, rather than assigning all
   students over 16 to tertiary institutions.
5. Keep employment independent of student type. After campus mapping, coordinate
   work and education locations only where the evidence supports co-location.
6. Keep enrolment assignment separate from daily attendance; do not exclude
   students based solely on missing detailed commuting status.

## Implemented survey classification

The existing processed survey stage now calls `classify_education` and retains:

- `study_level`: the current-education label, preserving doctorate/habilitation.
- `institution_type`: UH, FH_PH, higher_vocational or unresolved.
- `institution_type_source`: establishment, enterprise or unresolved.
- `institution_type_conflict`: two recognized tertiary NOGA types disagree.
- `study_institution_conflict`: reported level and institution type require review.
- `education_participation_conflict`: the education flags or level disagree about participation.
- `education_codebook_assumed`: the record's year differs from the supplied 2023 codebook.
- `education_od_group`: UH, other_tertiary, unresolved or not_applicable.

The institution label retains establishment evidence even when it conflicts;
the OD group remains unresolved for conflicting tertiary observations. FH/PH and
higher vocational education are combined only in `other_tertiary`. Codes 1–8
are outside these tertiary OD groups. Other levels with unresolved institutions
remain explicit; no survey records are dropped or imputed by this helper.

The age and employment classification tables describe raw survey employment,
before the processed stage's existing apprenticeship correction. They are
diagnostics, not a rerun of the full spatial pipeline. Student classification in
the synthetic population and destination assignment have not been changed.

For current education codes 9–13, the resulting unweighted respondent counts are:

| Year | UH OD group | Other tertiary OD group | Unresolved |
| --- | ---: | ---: | ---: |
| 2021 | 4,750 | 3,171 | 5,423 |
| 2022 | 4,790 | 3,185 | 4,923 |
| 2023 | 4,528 | 3,044 | 4,903 |

The unresolved column includes missing institution types and review conflicts;
it is not a group to discard. There are respectively 563, 510 and 553
study-level/institution conflicts within these tertiary levels. The three
classification cross-tabs preserve annual respondent and expanded weight totals.

The next step is to measure OD sample support and missing institution information
for the broad groups, then specify weighted imputation for unresolved cases
before estimating destination flows.

## Examination of unresolved cases

The 4,903 unresolved tertiary cases in 2023 break down as follows. Reasons are
mutually exclusive for this summary, with participation conflicts taking priority
over institution conflicts and study/institution conflicts. Individual flags
remain separate in the classification output.

| Current education | Total unresolved | Missing institution type | Study/institution conflict | Participation conflict |
| --- | ---: | ---: | ---: | ---: |
| Professional / higher professional examination (9) | 1,056 | 1,018 | 35 | 3 |
| Höhere Fachschule (10) | 1,112 | 936 | 174 | 2 |
| Bachelor (11) | 1,635 | 1,440 | 193 | 2 |
| Master (12) | 842 | 714 | 128 | 0 |
| Doctorate/habilitation (13) | 258 | 229 | 23 | 6 |

For broad study-family assignment, current education supplies more direct
evidence than school NOGA in codes 9–10. These programmes belong to higher
vocational study even where a provider is coded as a secondary school, training
provider or public administration. The present conservative classifier requires
recognized school NOGA and consequently leaves 2,168 such cases unresolved.
This is a limitation of the current OD grouping rule, not evidence that all
2,168 respondents' study families are unknowable.

Doctorate/habilitation likewise supports the academic university study family.
It does not identify the physical research/teaching site: cooperative programmes,
hospital appointments and other hosts must remain possible. Study-family OD
classification and observed host-institution classification should therefore be
distinguished before updating the conservative rules.

Bachelor and master codes do not distinguish UH from FH/PH. The 2023 extract has
no direct university/FH/PH programme-type or named-school identifier among the
inspected fields. Highest completed education and its aggregates cannot replace
the current institution type. Employment, occupation and age can support weighted
imputation, but do not determine the school attended.

Of all unresolved cases, 840 report a foreign school country, 3,829 have a valid
school municipality, and 234 have neither. The foreign cases include 715 of the
2,477 unresolved bachelor/master cases. Study abroad must be handled separately
from missing institution type at a Swiss destination, rather than filled with a
Swiss campus. A negative country code alone is not proof of study in Switzerland.

Exploratory checks of additional 2023 source fields found:

- 1,672 unresolved bachelor/master respondents have a valid school municipality.
  For 387, resolved survey respondents in that municipality occupy only one
  broad OD group; 1,112 occupy both, and 173 have no resolved counterpart.
- School quarters provide finer information: 1,031 have a valid quarter, of which
  528 have one observed group among resolved respondents, 352 have both and 151
  have no resolved counterpart. A single observed group is an imputation prior,
  not proof that only one institution type exists in that quarter.
- Matching municipality and positive school-establishment size gives a unique
  observed group for 252 cases; using enterprise size gives 308. These are not
  validated school identities. Size and legal form should only support a checked
  institution linkage or statistical model, not function as replacement IDs.
- Public-law legal form code 24 appears in both resolved groups (4,345 UH and
  2,331 other tertiary cases), so legal form alone cannot separate them.

Recommended next revision: use current education to assign the broad programme
family where it is explicit, preserve host-institution evidence and review flags,
separate foreign study, then impute UH versus other tertiary for remaining
bachelor/master cases using geography and calibrated survey information.
This examination does not itself change production classification rules.

## Revised OD design

Use one pooled tertiary OD sample for current education codes 9–13 rather than
separate UH and other-tertiary matrices. Keep classification fields as evidence
for compatible institution and establishment selection within the destination
zone, not as a requirement to enter the pooled sample. Missing school NOGA must
not remove an otherwise usable tertiary OD observation.

The existing education matrix stage now excludes positive reported foreign
school-country codes other than Switzerland (8100), in addition to its existing
foreign-zone exclusions. Negative special country codes are not treated as
foreign. This excludes observations from domestic OD estimation; it does not
remove people from the synthetic population. The 840 previously discussed cases
are the foreign subset of unresolved tertiary cases, not necessarily all foreign
study observations.

The existing matrix still covers education generally. A separate pooled tertiary
stage has now been prepared and audited; see [tertiary_education_od.md](tertiary_education_od.md).
Neither matrix is yet connected to the current education destination assignment.
Primary and secondary school flows are not combined with the new tertiary flows.

Establishment choice should account for institution/campus grouping, student
compatibility and enrolment-based destination budgets. If a sampled zone has no
compatible institution, zone selection must be revisited. Choosing a building
within a fixed zone cannot correct a wrong university-zone total; campus controls
may therefore need to constrain zone allocation as well.
