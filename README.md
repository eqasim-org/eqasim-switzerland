This repository contains all the scripts that are used to create the
IVT Switzerland / Zurich MATSim scenario. It uses a custom build pipeline
with `python` modules that call each other in the sense of incremental builds.

The starting point is `run.py`, where some configuration options can be set. Right
now it is not very configurable, but should become more so in the future.

# Deployment

No deployment yet, still work in progress. Later new updates will be automatically
deployed to NAS.

# Raw data

The raw data that is used in the process can be found on either of our servers
(pikelot, ifalik, nama) under:

```
/nas/ivtmatsim/scenario/raw/raw
```

**Microcensus Transport and Mobility**
- Content: `microcensus/` contains the Mikrozensus Verkehr und Mobilität in CSV
format with 60'000 daily trips of Swiss residents.
- Year: 2015 (published 2017)
- Contract: Rahmenvertrag with BfS

**STATPOP**
- Content: `statpop/` contains the Registererhebung (STATPOP) with socio-demographic
information on around 8M Swiss residents.
- Year: 2012
- Contract: Until end of 2018

**Structural Survey**
- Content: `structural_survey/` contains the Strukturerhebung with socio-demographic
and work and household related information about ~20% of the Swiss population in each
data set.
- Year: 2010, 2011, 2012
- Contract: Until end of 2018

**Municipality Borders**
- Content: `municipality_borders/` contains the shape files for Swiss municipalities
for different years.
- Year: 2008 - 2018
- Contract: [Open data][1]

**Statistical Quarters**
- Content: `statistical_quarter_borders/` contains the borders of the Statistische Quartiere,
which further divide large cities into smaller pieces. The 2017 data set fits exactly into
the 2018 municipality shape file.
- Year: 2017
- Contract: [Open Data][2]

**Spatial Structure**
- Content: `spatial_structure_2018.xlsx` is an Excel sheet with all kinds of spatial
classifications for all municipalities in 2018
- Year: 2018
- Contract: [Open Data][3]

**Municipality Type**
- Content: `municipality_types/` contains a shape file from BfS that assigns a Gemeindetyp
(municipality type) to each municipality.
- Year: 2014
- Contract: [Open Data][3]

**Country Codes**
- Content: `country_codes_2018.xlsx` contains the official BfS country codes
- Year: 2018
- Contract [Open Data][4]

**STATENT**
- Content: `statent/` contains the enterprise register for Switzerland with coordinates,
number of employees and classifications of the enterprises.
- Year: 2014
- Contract: Until end of 2018?

**OSM**
- Content: `osm/` contains a snapshot of the OSM database for Switzerland
from [geofabrik][5]. Originally, the format is bz2, but pt2matsim can only work
with gz. Therefore, it has been repackaged (see `utils/repackage_osm.sh`)!
- State: 7 Oct 2018
- Contract: [Open Data][5]

**HAFAS**
- Content: `hafas/` contains the official SBB HAFAS schedule for Switzerland.
- State: 17 Sep 2018
- Contract: [Open Data][6]

[1]: https://www.bfs.admin.ch/bfs/de/home/dienstleistungen/geostat/geodaten-bundesstatistik/administrative-grenzen/generalisierte-gemeindegrenzen.assetdetail.5247306.html

[2]: https://www.bfs.admin.ch/bfs/de/home/dienstleistungen/geostat/geodaten-bundesstatistik/administrative-grenzen/quartiergrenzen-schweizer-staedte.html

[3]: https://www.bfs.admin.ch/bfs/de/home/statistiken/querschnittsthemen/raeumliche-analysen/raeumliche-gliederungen/raeumliche-typologien.assetdetail.4542638.html

[4]: https://www.bfs.admin.ch/bfs/de/home/grundlagen/stgb.assetdetail.6166613.html
[5]: https://download.geofabrik.de/europe/switzerland.html
[6]: https://opendata.swiss/en/dataset/fahrplanentwurf-2018-hrdf
