# Campus weights and source sensitivity checks

`data.statent.education_campus_weights` and `analysis.education_campus_weights`
follow the existing `configure`/`execute` structure. The data stage returns site
eligibility, campus shares/estimated enrolment and institution-level conservation
summaries. Production uses pooled tertiary enrolment and teaching-only FTE shares;
the education target and balancing stages consume these weights. The audit also
retains detailed programme tables to inspect source assumptions. Those detailed
tables do not assign synthetic Bachelor/Master/doctoral groups.

`pooled_campus_weights.csv` and `pooled_site_eligibility.csv` describe production.
The detailed eligibility rules and numerical examples below describe the source
sensitivity audit, not the current pooled assignment.

## Eligibility

The existing registry now also records `eligible_doctoral_site`. Conflict-free UH
teaching establishments remain nominally compatible with bachelor/master/doctoral
study. Research establishments with local NOGA **721*** and enterprise NOGA
**854201** are additionally compatible with doctoral study. This adds six
research establishments nationally, including four in the candidate EPFL group.
Their evidence is `affiliated_uh_research_nominal_compatibility`.

Other ancillary sites and conflicts remain outside automatic compatibility.
FH/PH research cooperation does not on its own make that institution a recipient
of BFS university doctoral enrolment. Undergraduate/master research internships
and projects are possible, but this initial regular-destination rule does not
estimate placement frequency.
[EPFL bachelor laboratory placements](https://www.epfl.ch/education/bachelor/enhanced-learning-programs/summer-in-the-lab-internship-program/),
[Sion master projects and doctoral research](https://www.epfl.ch/schools/sb/research/isic/valais/)

Programme coverage is not proven by NOGA. Reviewed rules are now stored in
`data/statent/education_site_eligibility.csv`: Sion is excluded from the regular
bachelor split but remains placement-eligible. Its master eligibility is retained
for possible thesis destinations. The remote ZHAW branch is excluded from regular
bachelor/master splits while remaining placement-eligible. These are explicit
modelling assumptions, not proof that teaching is absent or observed participation
frequencies. Other sites retain nominal eligibility and review flags.
[ZHAW research branch](https://www.zhaw.ch/en/lsfm/institutes-centres/iunr/tourism-and-sustainable-development)

## Weighting

For each linked institution and study level:

`campus share = sum(FTE of eligible campus sites) / sum(FTE of eligible institution sites)`

`estimated enrolled students = BFS level enrolment * campus share`

Only eligible establishment FTE contributes; using whole-campus FTE could include
incompatible ancillary employment. All selected campuses appear, with zero share
where no site is eligible for that level. Missing/non-finite/negative eligible FTE,
zero eligible institution FTE or missing enrolment produce an unresolved estimate;
there is no hidden uniform/headcount fallback. A second linked enterprise cannot
duplicate a BFS institution total: it requires an explicit split before weighting.

Fractional estimates are preserved. They are not measured capacity, daily arrivals,
domestic-resident commuter targets or downsampled population counts. Both proposed
and verified identity links can be inspected, but proposed links remain proposed.
FTE includes postdocs, other researchers, technicians and support staff and can
misrepresent the distribution of students. Even a verified identity does not
validate FTE as enrolment. All estimates have `ready_for_assignment=False` and
base `student_capacity` remains unset.

## EPFL example, 2023/24

| Candidate site | Doctoral share | Estimated doctoral enrolment |
| --- | ---: | ---: |
| Lausanne/Ecublens | 88.77% | 2,046.93 |
| Sion | 2.39% | 55.16 |
| Neuchâtel | 2.88% | 66.36 |
| Geneva | 4.86% | 111.99 |
| Fribourg | 0.66% | 15.24 |
| Villigen-area site requiring review | 0.45% | 10.32 |

The estimate conserves the institution's 2,306 doctoral enrolments. Reviewed
regular bachelor weights now put all 6,773 EPFL bachelor enrolments at the main
campus; this is a provisional institutional split, not a resident commuting target.
Master weights retain 97.38% main campus and 2.62% Sion. The nominal NOGA-only
baseline is retained separately: it previously put about 177.74 bachelor students
in Sion. Evidence of occasional placements is not converted into a regular campus
student total.

UNIL and ZHAW receive their own provisional weights. Unsupported-location flags
are retained and do not remove sites: the official reference list is incomplete.
All eight original pilot institution/level totals were conserved. The updated
reference now includes FH/PH diplomas in Bachelor and UH licentiates/diplomas in
Master before campus splitting, consistent with survey codes 11–12. Continuing
education and higher vocational education remain outside this reference.

## Run and checks

```bash
python -m analysis.education_campus_weights \
  --registry-path /path/to/education_institutions \
  --enrolment-path /path/to/input/education_enrolment \
  --output-path /path/to/education_campus_weights
```

The CLI rebuilds the current registry from already-imputed site coordinates, so
previous audit CSVs can be reused without rerunning spatial imputation. Default
reference year is 2023/24. Outputs include `site_eligibility.csv`,
`campus_weights.csv`, `allocation_summary.csv`, `study_compatibility.csv`,
`teaching_only_baseline.csv`, `nominal_noga_baseline.csv`,
`doctoral_research_sensitivity.csv` and a README. `education_site_eligibility`
configures the reviewed rule file; the CLI offers `--eligibility-path`.

Checks passed for conservation of all eight enrolment totals, inclusion of
doctoral research sites, unchanged bachelor/master weights against the baseline,
missing/zero eligible FTE remaining unresolved, rejection of duplicated allocation,
and preservation of unset capacities and disconnected assignment.

For the connected assignment procedure, see
`tertiary_education_assignment.md`. The source sensitivity checks above remain
available to review eligibility assumptions; their degree-specific shares are
not production campus targets.
