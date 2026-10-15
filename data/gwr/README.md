Public GWR residence reference
=============================

`residences.py` follows the existing configure/execute/validate stage structure.
`gwr_residence_candidates` can override `<data_path>/gwr/residence_candidates.csv`.
The content hash participates in cache validation; the pipeline does not fetch
live buildings or depend on temporary audit paths.

The local public snapshot contains 4,774 unique existing GWR buildings:
4,712 communal-residence buildings (class 1130), 38 further buildings with explicit
student-housing names, and 24 further matches to FMEL's published addresses.
Overlapping evidence is deduplicated by EGID. Source date: 13 October 2026.
The 85 ambiguous name candidates from the map audit are not included.

Names and operator-address matches are evidence for residence candidates, not
student status or building occupancy. Address ranges and mixed-use buildings
do not establish that every resident belongs to the operator. GWR class 1122
alone never qualifies: Bourdonnette's seven apartment-class buildings qualify
through their explicit names and independently published FMEL addresses.

Reproducibility/source provenance is retained in evidence, GBEZ, operator
residence, verification URL, address, construction year and GEXPDAT columns.
An existing 2026 building need not have had the same use in the 2023 population
year. Missing names and unlinked operator addresses remain coverage limitations.

Attribution: Bundesamt für Statistik, Eidgenössisches Gebäude- und Wohnungsregister
(GWR), public export: https://public.madd.bfs.admin.ch/ch.zip
Access documentation: https://www.gwr.admin.ch/de/data/supply/public_content.html
Operator addresses: https://www.fmel.ch/nos-maisons
Bourdonnette verification: https://www.fmel.ch/maison/bourdonnette

Construction audit scripts and review outputs are recorded in
/private/tmp/gwr_named_residences_audit/README.md; pipeline execution needs only
the external CSV. No person identities or residential population locations are
included in this public building reference.

Reference CSVs are external inputs and are not tracked in Git. Copy the GWR
snapshot to `<data_path>/gwr/residence_candidates.csv` before running.
