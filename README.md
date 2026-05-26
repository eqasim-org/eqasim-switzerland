This repository contains the documentation and the code used to create the
Switzerland synthetic population and corresponding travel. It also provides stages that can be used to convert it and run a MATSim/eqasim agent-based simulations. Furthermore, it provides a visualization of the data at the end of the pipeline.

The pipeline uses the `synpp` Python package for stage chaining avaialble at [here](https://github.com/eqasim-org/synpp).

## Main reference

The main research reference for the synthetic population of Switzerland is:
> Dib, A., Sallard, A., and M. Balac (2026) [Agent-based transport model of Switzerland: A fully automated pipeline based on eqasim and MATSim](https://polybox.ethz.ch/index.php/s/eiNCcex4dDxfjpj), presented at the _26th Swiss Transportation Research Conference_, Ascona, Switzerland.


## Installation

Before using the pipeline one needs to have the Python environment set up. This can be done either by setting up the `conda` environment or a `python` environment.

For setting up the conda environment (not continuously tested):

Two bash scripts which set up everything that is needed to run the pipeline on Linux machines, as well as a requirements.txt file, can be found in `environment`:

- `setup.sh [path]` downloads Miniconda3, creates a Python virtual environment, installs OpenJDK and Maven. A path needs to be passed, which defines the directory in which the environment will be setup. Make sure you call this script with `bash`!
- `activate.sh [path]` activates the environment when the script is *source*'d. The path to the environment needs to be supplied.

Example:
- `bash environment/setup.sh myenv`
- `source environment/activate.sh myenv`

To clean, simply delete the environment directory (here `myenv`).

In case you are using a Mac machine there are minoconda paths within the `environment/setup.sh` file that you can use.

For settign up the python environment:
- Install `Python 3.10.13`
- Install packages in `euler_requirements.txt`
- How to do this in detail on our Euler server can be found [here](https://gitlab.ethz.ch/csfm/csfm-documentation/-/wikis/MATSim/Eqasim-on-Euler).

## Run

Once you have set up your environment, all dependencies should have been installed, including synpp. At this point, all you need to do is adjust the config file (**DO NOT MODIFY** `config.yml`) to run the stages you required, and then:

`python3 -m synpp config.yml`

## Generating pipeline flowchart

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



## Output

To create a full scenario, add the `matsim.simulation.run` stage to the config file. The
configuration option `output_path` must be set. The option
`output_path` must point to an existing directory, where results of the pipeline
will be saved. 

# Setting up and running on Windows

We recommend to run the pipeline on a Linux server, mainly because for large scenarios around 100GB of memory are needed. However, it is possible to run the pipeline locally on a Windows machine. This can either be done by setting up an environment step by step (see `environment/setup.sh` for the neccessary steps). Alternatively, the whole pipeline can run in  a Virtual Machine (VM). The procedure would be as follows:

- Install VirtualBox
- Install a lightweight Linux system in the VM, for instance Ubuntu Server
- Clone the pipeline repository and follow the setup instructions as above
- Download the data into the VM or mount a folder of the local file system in the VM to access all the necessary files for the pipeline

Depending on how the VM is configured (memory, cores, etc.) the pipeline will have a certain performance. However, it is probably much lower than running it in a real Linux environment. It only provides a solution for locally testing small cases, like creating a 0.1% sample population.

(TODO: More detailed explanation will follow)

## Raw data

The raw data that is used in the process can be found on our server
(Euler) under (only available to CSFM members):

```
/cluster/project/cmdp/ch_data/pipeline
```

**Microcensus Transport and Mobility**
- Content: `microcensus/` contains the Mikrozensus Verkehr und Mobilität in CSV
format with 60'000 daily trips of Swiss residents.
- The following files should be placed in the `microcensus/` directory: `etappen.csv`, `haushalte.csv`, `haushaltspersonen.csv`, `wege.csv`, `zielpersonen.csv`
- Year: 2015, 2021
- Contract: BfS

**STATPOP**
- Content: `statpop/` contains the Registererhebung (STATPOP) with socio-demographic
information on around 8M Swiss residents.
- The following files should be placed in the `statpop/` directory: `STATPOP_2023_HOUSEHOLD_CH_K.csv`, `STATPOP_2023_LINK_CH.csv`, `STATPOP_PP_2023_TEIL_1.csv`, `STATPOP_PP_2023_TEIL_2.csv`
- Year: 2023
- Contract: BfS

**Structural Survey**
- Content: `structural_survey/` contains the Strukturerhebung with socio-demographic
and work and household related information about ~20% of the Swiss population in each
data set.
- The following files should be placed in the `structural_survey/` directory: `se_zpers_2021_ch.csv`, `se_zpers_2022_ch.csv`, `se_zpers_2023_ch.csv`
- Year: 2021, 2022, 2023
- Contract: BfS

**STATENT**
- Content: `statent/` contains the enterprise register for Switzerland with coordinates,
number of employees and classifications of the enterprises.
- The following files should be placed in the `statent/` directory: `250221_STATENT_2022_LOC_17042025.csv`
- Year: 2022
- Contract: BfS

**Country Borders**
- Content: `spatial/country/` contains the shape file for Swiss border.
- Go to the link below and download the file `swissboundaries3d_2023-01_2056_5728.shp.zip`, unpack its contents and place `LANDESGEBIET` files to : `spatial/country/`
- Year: 2025
- Location: [Open data](https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d#swissBOUNDARIES3D---Download)

**Canton Borders**
- Content: `spatial/country/` contains the shape file for Cantonal borders.
- Go to the link below and download the file `swissboundaries3d_2023-01_2056_5728.shp.zip` (if you followed the previous step you already have this file), unpack its contents and place `KANTONSGEBIET` files to : `spatial/canton/`
- Year: 2025
- Location: [Open data](https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d#swissBOUNDARIES3D---Download)

**Municipality Borders**
- Content: `spatial/municipality/` contains the shape files for Swiss municipalities
for different years.
- Go to the link below and download the file `swissboundaries3d_2023-01_2056_5728.shp.zip` (if you followed the previous step you already have this file), `swissboundaries3d_2022-01_2056_5728.shp.zip`, and `swissboundaries3d_2021-01_2056_5728.shp.zip`, unpack them and place `HOHEITSGEBIET` files to corresponding years : `spatial/canton/2023`, `spatial/canton/2022`, `spatial/canton/2021`
- Year: 2021, 2022, 2023
- Location: [Open data](https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d#swissBOUNDARIES3D---Download)

**Statistical Quarters**
- Content: `spatial/statistical_quarter_borders/` contains the borders of the Statistische Quartiere,
which further divide large cities into smaller pieces. This file is unfortunately, no longer avaialble online. Therefore, we provide a version to download below.
- Download the file below, unpack it and place it within `spatial/statistical_quarter_borders/`
- Year: 2017
- Contract: Open data available in the `opendata` folder in this repository.

**NUTS**
- Content: `spatial/nuts_borders` contains the borders of the Nomenclature of Territorial Units for Statistics (NUTS) country
subdivisions.
- Download the data for two years 2021 and 2024 with the following attributes: Scale: 01M; FileFormat: SHP; coordinate system: EPSG:4326; GeometryType: Polygons(RG), and place the unpacked files into `spatial/nuts_borders`.
- State: 2021, 2024
- Contract: [Open Data](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics)

**ÖV Güteklasse**
- Content: `spatial/ov_guteklasse/` contains the shape files of ARE for the "ÖV Güteklasse",
which is a spatial classification of public transport level of service.
- Download the 2023 ov gueteklassen file, unzip it and place the `OeV_Gueteklassen_ARE.gpkg` file into `spatial/ov_guteklasse/`
- State: 2023
- Contract: [Open Data](https://data.geo.admin.ch/browser/index.html#/collections/ch.are.gueteklassen_oev?.language=en)

**Postal codes**
- Content: `spatial/postal_codes` contains shapefiles for postcodes in Switzerland.
- Download `ortschaftenverzeichnis_plz_2056.shp.zip` file, and unpack its contents into `spatial/postal_codes`
- State: 01.01.2024
- Contract: [Open Data](https://www.swisstopo.admin.ch/de/amtliches-ortschaftenverzeichnis#Download)

**Municipality Types**
- Content: `spatial/Raumgliederungen.xlsx` is an Excel sheet with all kinds of spatial
classifications for all municipalities on 01.01.2024
- Go to the link below, type 01.01.2024 as the date and select `Raum mit städtischem Charakter 2020` click on `Suche` and doenload the `xlsx` file provided at the bottom and place it in the `spatial` folder.
- Year: 01.01.2024
- Contract: [Open Data](https://www.agvchapp.bfs.admin.ch/de/typologies/query)

**Country Codes**
- Content: `spatial/be-b-00.04-sg-01.xlsx` contains the official BfS country codes
- Download the xlsx file available at the below link and add it to the `spatial` folder
- Year: 2024
- Contract [Open Data](https://www.bfs.admin.ch/bfs/de/home/grundlagen/stgb.assetdetail.32028071.html)


**OSM**
- Content: `osm/` contains a snapshot of the OSM database for Switzerland. 
- State: 2025
- Contract: [Open Data](https://download.geofabrik.de/europe/switzerland.html)

**HAFAS**
- Content: `hafas/` contains the official SBB HAFAS schedule for Switzerland.
- Use it only if you do not want to use gtfs below.
- State: 2025
- Contract: [Open Data](https://data.opentransportdata.swiss/dataset/timetable-54-2025-hrdf)

**GTFS**
- Content: `gtfs/` contains the official GTFS schedule for Switzerland.
- Download the file below and place it in  `gtfs/` folder, the code itself will unpack it.
- State: 2025
- Contract: [Open Data](https://data.opentransportdata.swiss/de/dataset/timetable-2025-gtfs2020)

**Freight**
- GTE:
    - Content: `freight/gte_2023` contains data from GTE survey which examines freight travel for freight vehicles registered in Switzerland.
    - Copy the data located in `Donnes/`, `journeych.csv`, `transport.csv`, `week.csv` into the `freight/gte_2023` folder
    - State: 2023
    - Contract: BfS
- GQGV:
    - Content: `freight/gqgv_2019` contains data from GQGV survey which examines freight travel for freight vehicles registered abroad.
    - State: 2019
    - Contract: BfS
- Departure times:
    - Content: `freight/departure_times.csv` contains data on the probability of a freight vehicle departing within a certain time bin. This data is not avaialble online. Please use the file provided below.
    - State: 2008
    - Contract: Open data available in the `opendata` folder in this repository.

**Projections are used currently only for the population and not freight; need an update to the code**
- Households:
    - Content: `projections/households` should contain data for number of households per canton from 2020-2050 (unfortunately it is no lonegr avaialble to download household size distribution per canton). All projections are according to the BfS reference scenario.
    - Download the `xlsx` file and place it in the `projections/households` folder
    - State: 2024
    - Contract:
        - Projections: [Open Data](https://www.bfs.admin.ch/bfs/de/home/statistiken/katalog.assetdetail.16344851.html)
- Population:
    - Content: `projections/population` contains data of population per canton, nationality, gender and age from 2024-2055. All projections are according to the BfS reference scenario.
    - On the webpage below select for Kanton: all except Schweiz; Staatsangehörigkeit (Kategorie): Schweiz and Ausland; Geschlecht: Mann and Frau; Alter: all except Total; Jahr: all; Beobachtungseinheit: Bevölkerungsstand am 1. Januar. Click on Weiter. On the left side in the dropdwon menu select `Ergebnis speichern asl... Excel`, and place the downloaded file in the `projections/population` folder
    - State: 2024
    - Contract:
        - Projections: [Open Data](https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-0104020000_101/px-x-0104020000_101/px-x-0104020000_101.px)
- Freight:
    - Content: `projections/are/freight` contains projections for freight traffic from 2010 to 2040.
    All projections are according to the ARE Transport Outlook 2050 reference scenario.
    - State: 2024
    - Contract: 

Finally your data folder should look something like this:
```
+--- statpop
|   +--- STATPOP_2023_LINK_CH.csv
|   +--- STATPOP_2023_HOUSEHOLD_CH_K.csv
|   +--- STATPOP_PP_2023_TEIL_1.csv
|   +--- STATPOP_PP_2023_TEIL_2.csv
+--- statent
|   +--- 250221_STATENT_2022_LOC_17042025.csv
+--- osm
|   +--- switzerland-latest-2025.osm.pbf
+--- freight
|   +--- GTE_2023
|   |   +--- journeych.csv
|   |   +--- transport.csv
|   |   +--- week.csv
|   +--- GQGV_2019
|   |   +--- GQGV_2019_Mikrodaten.csv
|   +--- departure_times.csv
+--- gtfs
|   +--- gtfs_fp2024_2024-11-11.zip
+--- hafas
+--- microcensus
|   +--- haushalte.csv
|   +--- zielpersonen.csv
|   +--- wege.csv
|   +--- etappen.csv
|   +--- haushaltspersonen.csv
+--- projections
|   +--- households
|   |   +--- su-d-01.03.03.01.xlsx
|   +--- population
|   |   +--- px-x-0104020000_101_20250808-151932.csv
+--- spatial
|   +--- municipality
|   |   +--- 2023
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.cpg
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.dbf
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.prj
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shp
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shx
|   |   +--- 2022
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.cpg
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.dbf
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.prj
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shp
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shx
|   |   +--- 2021
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.cpg
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.dbf
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.prj
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shp
|   |   |   +--- swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shx
|   +--- Raumgliederungen.xlsx
|   +--- nuts_borders
|   |   +--- NUTS_RG_01M_2024_4326.cpg
|   |   +--- NUTS_RG_01M_2024_4326.dbf
|   |   +--- NUTS_RG_01M_2024_4326.prj
|   |   +--- NUTS_RG_01M_2024_4326.shp
|   |   +--- NUTS_RG_01M_2024_4326.shx
|   |   +--- NUTS_RG_01M_2021_4326.cpg
|   |   +--- NUTS_RG_01M_2021_4326.dbf
|   |   +--- NUTS_RG_01M_2021_4326.prj
|   |   +--- NUTS_RG_01M_2021_4326.shp
|   |   +--- NUTS_RG_01M_2021_4326.shx
|   +--- ov_guteklasse
|   |   +--- OeV_Gueteklassen_ARE.gpkg
|   +--- canton
|   |   +--- swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.cpg
|   |   +--- swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.dbf
|   |   +--- swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.prj
|   |   +--- swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp
|   |   +--- swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shx
|   +--- statistical_quarter_borders
|   |   +--- quart17.dbf
|   |   +--- quart17.prj
|   |   +--- quart17.shp
|   |   +--- quart17.shx
|   +--- postal_codes
|   |   +--- AMTOVZ_ZIP.cpg
|   |   +--- AMTOVZ_ZIP.dbf
|   |   +--- AMTOVZ_ZIP.prj
|   |   +--- AMTOVZ_ZIP.shp
|   |   +--- AMTOVZ_ZIP.shx
|   +--- country
|   |   +--- swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.cpg
|   |   +--- swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.dbf
|   |   +--- swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.prj
|   |   +--- swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shp
|   |   +--- swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET.shx
|   +--- be-b-00.04-sg-01.xlsx
+--- structural_survey
|   +--- se_zpers_2021_ch.csv
|   +--- se_zpers_2022_ch.csv
|   +--- se_zpers_2023_ch.csv
```


