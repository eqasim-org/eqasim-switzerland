**Make sure you are on a develop branch when you are using this repository as it contains the most recent improvements tot he pipeline!! The master branch will be soon updated to reflect these important updates!!**

This repository contains all the scripts that are used to create the
IVT Switzerland / Zurich MATSim scenario. It uses a custom build pipeline
with `python` modules that call each other in the sense of incremental builds.

A more flexible version is being made public at [https://github.com/eqasim-org/synpp](https://github.com/eqasim-org/synpp). The documentation is more thorough, and may be helpful.

# Installation

Two bash scripts which set up everything that is needed to run the pipeline on our servers, as well as a requirements.txt file, can be found in `environment`:

- `setup.sh [path]` downloads Miniconda3, creates a Python virtual environment, installs OpenJDK and Maven. A path needs to be passed, which defines the directory in which the environment will be setup. Make sure you call this script with `bash`!
- `activate.sh [path]` activates the environment when the script is *source*'d. The path to the environment needs to be supplied.

Example:
- `bash environment/setup.sh myenv`
- `source environment/activate.sh myenv`

To clean, simply delete the environment directory (here `myenv`).

# Run

Once you have set up your environment, all dependencies should have been installed, including synpp. At this point, all you need to do is adjust the config file (**DO NOT MODIFY** `config_gitlab.yml`, as this is the one that is used for gitlab testing) to run the stages you required, and then:

`python3 -m synpp config.yml`

# Generating pipeline flowchart

To generate a json file which represents the graph of the pipeline, you need to specify the output path in the config file.
Simply add `flowchart_path: /path/to/flowchart.json` under the "General pipeline settings".
Then, the flowchart json will be saved at this path next time you run the pipeline until the last specified stage.
To only generate the flowchart without running the pipeline, add `dryrun: true` under the "General pipeline settings".
Setting `dryrun: false` will run the full pipeline.

To convert the json file into an image, you will need to use 
[Pipeline Data Flow Plotter](https://gitlab.ethz.ch/ivt-vpl/populations/pipeline-data-flow-plotter).
The full syntax is:

`python3 visualize_pipeline.py -j /path/to/flowchart.json -o /path/to/flowchart.png -g`

Type `python3 visualize_pipeline.py -h` for further explanations.



# Output

To create a full scenario, add the `matsim.final` stage to the config file. The
configuration options `output_path` and `output_id` must be set. The option
`output_path` must point to an existing directory, where results of the pipeline
will be saved. For a specific run, the scenario output will be written to the
folder `output_path/output_id`, i.e. the subfolder "output_id" *may be overwritten*
if it exists already.

# Deployment

No deployment yet, still work in progress. Later new updates will be automatically
deployed to NAS.

# Setting up and running on Windows

We recommend to run the pipeline on a Linux server, mainly because for large scenarios around 100GB of memory are needed. However, it is possible to run the pipeline locally on a Windows machine. This can either be done by setting up an environment step by step (see `environment/setup.sh` for the neccessary steps). Alternatively, the whole pipeline can run in  a Virtual Machine (VM). The procedure would be as follows:

- Install VirtualBox
- Install a lightweight Linux system in the VM, for instance Ubuntu Server
- Clone the pipeline repository and follow the setup instructions as above
- Download the data into the VM or mount a folder of the local file system in the VM to access all the necessary files for the pipeline

Depending on how the VM is configured (memory, cores, etc.) the pipeline will have a certain performance. However, it is probably much lower than running it in a real Linux environment. It only provides a solution for locally testing small cases, like creating a 0.1% sample population.

(TODO: More detailed explanation will follow)

# Docker

*This was EXPERIMENTAL. The information may be outdated*

Alternatively, the pipeline is available as a dockerized application. To create
the Docker container, call `docker build -t chpop .` in the project directory.

The pipeline can then be run using `docker run -v [data path]:/data -v [cache path]:/cache chpop /cache/config_docker.yml`. This assumes that the raw data is located at `[data path]` and that the output path is at `[cache path]`. The directories will be mounted in the docker container at `/data` and `/cache`, respectively. To use the docker container, a config file must be provided in
one of the mounted directories, e.g. in `/cache/config_docker.yml`. An example config
file is given in `config_docker.yml`. Note that also there the paths must be adjusted accordingly.

# Raw data

The raw data that is used in the process can be found on either of our servers
(pikelot, ifalik, nama) under:

```
/nas/ivtmatsim/scenarios/switzerland/data OR /nas/ivtmatsim/scenario/raw/raw
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

**ÖV Güteklasse**
- Content: `ov_guteklasse/` contains the shape files of ARE for the "ÖV Güteklasse",
which is a spatial classification of public transport level of service.
- State: 20 Mar 2018
- Contract: [Open Data][7]

**ARE Gemeindetypologie**
- Content: `municipality_types` contains the ARE Gemeindetypologie which assigns a certain spatial type to each municipality in Switzerland.
- State: 26 Feb 2019
- Contract: [Open Data][8]

**Projections**
- Households:
    - Content: `projections/households` contains data for household sizes per canton from 2012-2017 and projections of household sizes per canton in 2017 and 2045.
    All projections are according to the BfS reference scenario.
    - State: 1 Apr 2019
    - Contract:
        - Past data: [Open Data][9]
        - Projections: [Open Data][10]
- Population:
    - Content: `projections/population` contains data of population per canton, nationality, gender and age from 2010-2017 and projections from 2015 to 2045.
    All projections are according to the BfS reference scenario.
    - State: 1 Apr 2019
    - Contract:
        - Past data: [Open Data][11]
        - Projections: [Open Data][12]
- Freight:
    - Content: `projections/are/freight` contains projections for freight traffic from 2010 to 2040.
    All projections are according to the ARE Transport Outlook 2040 reference scenario.
    - State: 20 Oct 2016
    - Contract: [Open Data][15]

**NUTS**
- Content: `nuts_borders` contains the borders of the Nomenclature of Territorial Units for Statistics (NUTS) country
subdivisions.
- State: 2016, 2013, 2010, 2006 & 2003
- Contract: [Open Data][13]

**Postal codes**
- Content: `postal_codes` contains shapefiles for postcodes in Switzerland.
- State: 1 Apr 2019
- Contract: [Open Data][14]

**Freight**
- GTE:
    - Content: `freight/gte` contains data from GTE survey which examines freight travel for freight vehicles registered in Switzerland.
    - State: 2017
    - Contract: BFS contract until ?
- GQGV:
    - Content: `freight/gqgv` contains data from GQGV survey which examines freight travel for freight vehicles registered abroad.
    - State: 2014
    - Contract: BFS contract until ?
- Departure times:
    - Content: `freight/departure_times.csv` contains data on the probability of a freight vehicle departing within a certain time bin.
    - State: 2008
    - Contract: [Open Data][16]



[1]: https://www.bfs.admin.ch/bfs/de/home/dienstleistungen/geostat/geodaten-bundesstatistik/administrative-grenzen/generalisierte-gemeindegrenzen.assetdetail.5247306.html

[2]: https://www.bfs.admin.ch/bfs/de/home/dienstleistungen/geostat/geodaten-bundesstatistik/administrative-grenzen/quartiergrenzen-schweizer-staedte.html

[3]: https://www.bfs.admin.ch/bfs/de/home/statistiken/querschnittsthemen/raeumliche-analysen/raeumliche-gliederungen/raeumliche-typologien.assetdetail.4542638.html

[4]: https://www.bfs.admin.ch/bfs/de/home/grundlagen/stgb.assetdetail.6166613.html
[5]: https://download.geofabrik.de/europe/switzerland.html
[6]: https://opendata.swiss/en/dataset/fahrplanentwurf-2018-hrdf
[7]: https://opendata.swiss/de/dataset/ov-guteklassen-are
[8]: https://opendata.swiss/de/dataset/gemeindetypologie-are
[9]: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/daten.assetdetail.6106027.html
[10]: https://www.bfs.admin.ch/bfs/de/home/statistiken/kataloge-datenbanken/tabellen.assetdetail.3882982.html
[11]: https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/stand-entwicklung/bevoelkerung.assetdetail.5887433.html
[12]: https://www.bfs.admin.ch/bfs/de/home/statistiken/bevoelkerung/zukuenftige-entwicklung/kantonale-szenarien.assetdetail.255402.html
[13]: https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
[14]: https://www.cadastre.ch/en/services/service/plz.html
[15]: https://www.are.admin.ch/are/en/home/transport-and-infrastructure/data/transport-perspectives.html
[16]: https://trimis.ec.europa.eu/sites/default/files/project/documents/20150826_232657_83989_SVI_1999_328.pdf
