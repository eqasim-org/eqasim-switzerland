# eqasim Switzerland

This repository generates Switzerland's synthetic population and daily travel,
builds MATSim/eqasim scenarios, runs transport simulations, and analyzes the results.
The pipeline uses [Synpp](https://github.com/eqasim-org/synpp) to organize its stages.

See the [documentation index](doc/INDEX.md) for module guides and changelogs.

## Main reference

The main research reference for the synthetic population of Switzerland is:
> Dib, A., Sallard, A., and M. Balac (2026) [Agent-based transport model of Switzerland: A fully automated pipeline based on eqasim and MATSim](https://polybox.ethz.ch/index.php/s/eiNCcex4dDxfjpj), presented at the _26th Swiss Transportation Research Conference_, Ascona, Switzerland.

> Dib, A., and M. Balac (2026) [Agent-Based Transport Model Calibration via Online Stochastic Batch Optimization](https://www.research-collection.ethz.ch/entities/publication/62e03c14-62a2-4b7f-b16e-402bac39b5eb), presented at the World Conference on transport research (WCTR 2026), Toulouse, France.

## Preparing the environment

Use Python 3.13 or newer and `uv` for dependency management. MATSim requires a
Java 25 JDK; building eqasim from source also requires Maven 3.x and access to its
dependency repositories. Install the Python dependencies from the repository root:

```bash
cd /project/code
uv sync
```

In particular, the dependencies contain the [synpp](https://github.com/eqasim-org/synpp) package, which is the computational backbone of the pipeline.

`pyproject.toml` and `uv.lock` pin Synpp to commit
[`638adc5`](https://github.com/Dib-AEK/synpp/commit/638adc5ea847eef97a8637c8620ea636b436ae8a)
of [Dib-AEK/synpp](https://github.com/Dib-AEK/synpp/tree/fix/shared-dependency-traversal),
based on v1.7.3. The fork resolves shared dependencies once per stage, avoiding
the repeated traversal that stalled mode-choice initialization. `uv sync` installs
the corrected package directly; no local package patch is needed.

## Executing the pipeline

The beginner [configuration template](config.yml) contains the main inputs and
scenario choices, each with a short comment. Advanced defaults are declared
directly in each stage. Explicit YAML values override these defaults.


To run the pipeline, go to the project directory and call the `synpp` execution script:

```bash
cd /project/code
uv run -m synpp config.yml
```

Fill in the template's paths and input filenames before running it. The selected
`run` stages determine whether population synthesis, simulation, or analysis is
executed. The working cluster configuration is run with
`uv run -m synpp config_dib_10.yml`; private configurations are not included in Git.
On the cluster, Maven needs explicit proxy settings. The local launchers enable
these with `MAVEN_USE_ENV_PROXY=true` after loading `eth_proxy`. Alternatively,
set `eqasim_path` to an existing, valid eqasim JAR.

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

Use `matsim.output` in the configuration's `run` list to build, simulate and
export a complete scenario to `<output_path>/<output_id>`. Set `output_path` to
your export folder. Scenario inputs and simulation results are copied there;
the upstream caches remain available. `matsim.simulation.run` alone runs the
prepared scenario without exporting it. Use `synthesis.output` to export the
synthetic population separately.

# Setting up and running on Windows

We recommend running the pipeline on a Linux server because full-country scenarios can require substantial memory. However, it is possible to run the pipeline locally on a Windows machine. The whole pipeline can run in a Virtual Machine (VM). The procedure would be as follows:

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
format describing Swiss residents' daily travel.
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
    - Content: `projections/households/households_2025.csv` contains private-household counts from the BFS reference scenario **AM-00-2025**, for 2025–2055. Use the code-based SDMX CSV export from [DF_SCEN_MENAGES](https://stats.swiss/vis?lc=en&df%5bds%5d=disseminate&df%5bid%5d=DF_SCEN_MENAGES&df%5bag%5d=CH1.SCEN&df%5bvs%5d=1.0.0).
    - Select `SCENARIO=AM-00-2025`, all cantons, `SIZE_HOUSEHOLD=_T`, `NUMBER_HOUSEHOLD_PERSON=MEN`, `FREQ=A`, `TIME_HORIZ=P`, and the desired years. The reader also accepts exports containing the national total and other measures, but selects only private-household counts for the 26 cantons. The old AM-00-2020 workbook is no longer used.
    - Recovered residents' synthetic singleton households are excluded from this private-household margin. They remain in the total-person controls and in household replication after fitting.
    - State: 2025
    - Contract:
        - Projections: [Open Data](https://www.bfs.admin.ch/asset/en/DF_SCEN_MENAGES)
- Population:
    - Content: `projections/population/population_year_end_2025.csv` contains population by canton, nationality, sex and single age for 2024–2055 from reference scenario **AR-00-2025**. The suffix identifies the projection vintage; `scaling_year` selects the target year.
    - Select Kanton: all except Schweiz; Staatsangehörigkeit (Kategorie): Schweiz and Ausland; Geschlecht: Mann and Frau; Alter: ages 0 through 100+ without Total; Jahr: all; Beobachtungseinheit: **Bevölkerungsstand am 31. Dezember**. Save as a semicolon-delimited CSV with German labels (Latin-1 encoding) at the path above.
    - `scaling_year: 2025` now means **31 December 2025**. January stocks use age attained during the coming year and cannot be substituted directly. The reader rejects a January-only extract; it preserves actual STATPOP ages and constructs year-end control groups instead.
    - State: 2025
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
|   |   +--- households_2025.csv
|   +--- population
|   |   +--- population_year_end_2025.csv
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


