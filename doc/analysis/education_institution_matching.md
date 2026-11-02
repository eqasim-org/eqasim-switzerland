# Review of EPFL, UNIL and ZHAW candidates

The analysis follows the existing `configure`/`execute` structure and uses the
STATENT registry and BFS enrolment stage. It does not change student assignment,
campus capacity or study-level eligibility.

`<data_path>/education_enrolment/education_reference_sites.csv` contains 14 documented location
references, official sources, LV95 coordinates, coordinate sources, notes and
review date. Addresses were checked on institution pages and geocoded with the
swisstopo address service. Coordinates use its projected geometry, avoiding the
reversed `x`/`y` attribute names. EPFL Lausanne uses an approximate official
campus-map view centre. These are anchors, not complete campus boundaries or a
historical inventory of buildings in 2022.

The audit measures each reference's distance to the nearest retained site of
every enterprise with the same nominal institution type. It ranks candidates by
references within 500 m, then nearest distance and source ID. Shared locations
retain all alternatives. Sensitivity is reported at 250/500/1000 m. A connected
campus can be supported by an anchor even if some of its sites are farther away.
Individual distances remain visible; unsupported campuses are flagged for review,
not excluded. No rank automatically verifies an anonymous identity.

| Candidate | Enterprise | Supported references | Rank | Sites | Clusters |
| --- | --- | ---: | ---: | ---: | ---: |
| EPFL | 806544543221 | 5 / 5 | 1 | 6 | 6 |
| UNIL | 883878979156 | 3 / 3 | 1 | 62 | 6 |
| ZHAW | 838828978571 | 5 / 6 | 1 | 12 | 5 |

EPFL supports Lausanne/Ecublens, Geneva, Neuchâtel, Fribourg and Sion; its
Villigen-area research site remains outside the references. Shared-location
alternatives exist at several anchors.
[EPFL locations](https://www.epfl.ch/campus/visitors/coming-to-epfl/)

UNIL supports Dorigny, Bugnon and Epalinges. Three other single-site clusters
remain outside the anchors, including one near Sion. The source URLs for all
three references are retained in the reference CSV.
[UNIL Bugnon facility](https://cif.unil.ch/dorigny-bookings/)

ZHAW supports Winterthur Technikum/Gertrudstrasse, Zürich Toni-Areal, Wädenswil
Grüental and Reidbach. Its published Lagerstrasse address has no candidate-group
site within 500 m (nearest approximately 2.14 km); other FH/PH groups occur nearby.
The gap could reflect changed locations or another enterprise and needs review.
[ZHAW locations](https://www.zhaw.ch/en/about-us/visitor-information/location-plans),
[Lagerstrasse address](https://www.zhaw.ch/de/engineering/studium/bachelorstudium/infotag-zum-bachelorstudium)

The small remote ZHAW candidate site is consistent with its documented Wergenstein
research presence. It remains flagged because an exact building reference has not
been checked. This public description does not verify the anonymous establishment.
[ZHAW IUNR locations](https://www.zhaw.ch/en/lsfm/institutes-centres/iunr/about-us/locations)

All three links remain **proposed**, with the spatial review recorded in their
evidence notes. Public locations provide strong geographic support but no definitive
anonymous-enterprise crosswalk. Other nearby enterprises do not inherit student
totals. No BFS total is multiplied across enterprises or campuses.

```bash
python -m analysis.education_institution_matching \
  --registry-path /path/to/education_institutions \
  --enrolment-path /path/to/input/education_enrolment \
  --output-path /path/to/education_institution_matching
```

Outputs include reviewed links, ranked candidates, all reference distances,
site/campus review flags, references, sensitivity, README and `campuses.html`.
Map labels show candidate names and status. Official reference points can be
enabled in the layer selector. Colours still describe institution type/grouping
review, not name confidence; unreviewed groups remain unnamed.

Validation checked stable rankings after establishment reordering, all three
candidates rank first, shared-location alternatives remain visible, and links
remain proposed. All 1,460 registry markers and 14 references were preserved in
the generated map. The existing campus map and enrolment audit were regenerated;
the third proposed link leaves BFS reference totals unchanged.

Next: review remaining campus locations and study-level eligibility, especially
doctoral research sites, before estimating FTE-weighted campus shares. Retain
identity and eligibility assumptions alongside each estimate.
