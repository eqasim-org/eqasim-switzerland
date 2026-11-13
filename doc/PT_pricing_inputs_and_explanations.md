# Important note
The processing stages depend on PDF libraries that are only available through Ubuntu shell scripts and accessed in Python via the subprocess library. These libraries are not provided on Euler, and I could not find a workaround to make them available there. Existing Python PDF-to-text libraries were tested but did not meet the required quality standards. For now, I therefore suggest treating these stages as “locally runnable only.” In practice, this should not be a major limitation: the zone assignment and SBB network construction are expected to remain stable over time, so once the inputs are finalized, it should not be necessary to re-run these stages for every pipeline execution. Simply copying the prepared input files will be sufficient.

# Input files and folder structure
Folder path: ch_data/pipeline/pt_pricing. 

Most of the files are published by PT operators on the [Alliance Swiss Pass website](https://www.allianceswisspass.ch/de/tarife-vorschriften/uebersicht). However, in some cases, the information provided is not sufficient to create the input. A solution was to use the archive built by Sebastian Hörl and Joe Molloy in 2018 when they worked on the [ch-pt-pricing library](https://github.com/joemolloy/ch-pt-pricing/tree/main/ch_pt_utils) back in 2018. These data sets are available in the shared IVT servers (let's hope no one gets rid of them...) at /nas/ivtmatsim/pt_data.

### output
This subfolder contains the results of the processing stages. They are ready to be used within eqasim-java.

### T601
The T601_f.pdf file is the French version of the T601 tarif document from [[Alliance Swiss Pass](https://www.allianceswisspass.ch/de/tarife-vorschriften/uebersicht)]. This document provides the main pricing algorithm used, among others, by SBB.

### T603
This document, also from [Alliance Swiss Pass](https://www.allianceswisspass.ch/de/tarife-vorschriften/uebersicht)], is the reference for finding out which distance is charged by SBB for each origin-destination pair, as it often doesn't exactly correspond to the "real" network distance. Two versions are necessary here:
- the 2025 release (T603_2025.pdf) only contains the most important distances in the SBB network (pages 16 to 24).
- the 20218 release (T603_2018.pdf) however contains all "distance triangles" that are required to reconstruct the graph representing the SBB network and the priced distances (pages 20 to 63).

### T651
Each local PT operator publishes one PDF that contains two major pieces of information:
- the pricing rules applied in the local network
- the "stops registry" which is often a table mapping each PT stop name with the zone(s) it belongs to in the local network. Sometimes the registry is missing and only a map of the local network is provided.

For Mobilis (operating in the Canton Vaud), the T651 PDF document only contains a network map, but Martin Repoux from TL shared an excel file (flph_Liste des arrêts Mobilis _dès le 15.12.2024 modif dès le 01.06.2025.xlsx) that can be used as a stop registry. Myriam Bris, from TPG (operating in Geneva), also shared a csv file mapping TPG stop names with their GTFS id. This csv was used to simplify the cleaning process for the TPG (Unireso) stops. 

Otherwise, most of the PDFs were extracted from [Alliance Swiss Pass](https://www.allianceswisspass.ch/de/tarife-vorschriften/uebersicht/Tarife-der-Verbuende). Some exploration of the individual websites might be required to find the appropriate documents.

The canton of Zurich has released an online map of the cantonal PT zones at [Geo ZH](https://geo.zh.ch/maps?x=2682588&y=1253620&scale=269371&basemap=arelkbackgroundzh). This map, downloaded as a shp file  in ch_data/pipeline/pt_pricing/t651/ZVVzonenplan/Tarifzonen_des_offentlichen_Verkehrs_-OGD/ZVV_TARIFZONEN_F.shp. Thus, the canton of Zurich (and the canton of Geneva because it is one single zone), are the only two cantons whose division into PT zones is sure. For the other cantons/authorities, uncertainties remain because of the unaccuracy of the PDF data extraction and cleaning phase, which often relies on manually adding or reassigning stops.

If necessary, I can add more information about the stages leading from the raw PDFs to the final gtfs_zones.csv file.



# Zone polygons (data.pt_pricing.zones)
No official map of the zones exists for most fare networks. The stage `data.pt_pricing.zones` rebuilds one polygon per zone of each network from the municipalities of `tarifverbundkarte.geojson` ([SBB open data: Tarifverbundkarte](https://data.sbb.ch/explore/dataset/tarifverbundkarte/information/)), the networks they belong to, and the zones of the stops (`gtfs_zones.csv` of `data.pt_pricing.pt_pricing`, located with the stops of `data.gtfs.cleaned`):
- a municipality belongs to the zone of at least 90 % of its stops; otherwise it is split by nearest stop (Bern, Lausanne...);
- a municipality whose stops are all on a zone border (Wichtrach: Libero 626 and 710) is split between these zones;
- a municipality without stops takes the zone of the neighbour with which it shares the longest border.

The French zones of Unireso come from `unireso_france_zones.csv` (insee, commune, zone, optionally network) and the commune outlines of `communes_france.gpkg` (insee, name, geometry). A commune listed in several zones is split by its nearest stops.

The French zones only come from that list: a Swiss stop given one of them is not counted for its municipality. `gtfs_zones.csv` gave Unireso zone 230 to the stops of Collonges VS, named like Collonges in the Ain; `uniresoFR.assign_zones_to_gtfs` now leaves out the stops inside the Swiss cantons, so regenerated inputs no longer have them.

The stage returns a GeoDataFrame (network, zone, geometry, municipalities, name; EPSG:2056) and writes `pt_zones.gpkg` and an interactive map, `pt_zones.html`, in its cache folder. The map has one layer per network on an Esri background (Z-Pass layers hidden at first), a layer of the places in several regional networks (dashed outline), and a layer of the stops at zone or network borders (white dots, with all their zones on click). A click anywhere lists every zone of every network at that place, including the hidden layers, and outlines them. The input files are set by `pt_zones_municipalities_file`, `pt_zones_foreign_communes_file` and `pt_zones_foreign_zones_file` (relative to `data_path`, by default in `pt_pricing/`). `synthesis.population.models.subscription_zones` uses these polygons, for the zones of the homes and destinations and of the ride tracks (`utils/gtfs_router/tracks.py`).

# Files passed to MATSim (data.pt_pricing.pt_pricing)
The stage `data.pt_pricing.pt_pricing` returns `(SBB_all_distances.csv, gtfs_zones.csv, pricingDescription.xml, {name: Léman Pass table})`. The MATSim preparation copies these files next to the config. `matsim.config.tools.PT_PRICING_FILES` lists them with their parameters in the `ptZones` module:

| Parameter | File |
| --- | --- |
| `ptZonesFilePath` | `gtfs_zones.csv`: the zones of the stops. The stage adds a row for each feed stop that Python and Java would not find by its id or its station, with the zones of its station (`pt_pricing_rekey_stops`, default `always`; existing rows are kept). This includes the SNCF ids of stations (`StopPoint:OCETrain:::TER-85171421` is Lancy-Bachet, `8517142`). It also adds the feed stops in the French communes of Unireso, with the zone of their commune. |
| `sbbDistancesPath` | `SBB_all_distances.csv` |
| `pricingDescriptionPath` | `pricingDescription.xml` |
| `lemanPassProductsPath` | `lemanpass_products.csv` |
| `lemanPassStationsPath` | `lemanpass_stations.csv` |

The Léman Pass tables hold the single tickets only; subscriptions are not passed to MATSim. They come from the tpg fare table (`tpg_<date>_Fare-system_Geneva.xlsx` in `pt_pricing_lemanpass_path`, by default `pt_pricing/output`). Without that table they are written with their header only, which means no Léman Pass. They are UTF-8, comma-separated CSV files with a header.

- `lemanpass_products.csv`: one row per product (trip type of the fare table), 1,102 in the 2025 table.
  - `product_id`, and `label`, the trip type, for reading only.
  - `pieces`, `end_a`, `end_b` and `cover`: lists of places separated by `|`. A place is `station:<name>` (a station of `lemanpass_stations.csv`) or `zone:<number>` (a Unireso zone, as in `gtfs_zones.csv`). `pieces` are the pieces of the product. `end_a` and `end_b` are the places its two ends hold, with the stations inside its zones already listed. `cover` is only filled for a product made of zones only (`Zone 10 + Zone 230`): every place it covers.
  - The price in CHF for each traveller and class: `adult_2`, `adult_1`, `child_2`, `child_1` (6-15), `youth_2`, `youth_1` (16-25), `reduced_2` and `reduced_1` (with the half-fare card; only 30 products have it). A price is empty when the table has none; when it has two, only the lower is kept.
- `lemanpass_stations.csv`: one row per rail stop of a Léman Pass station: `stop_id` (the GTFS and transit schedule id), `station`, `zone` (its Unireso zone) and `country` (`FR` or `SW`).

A journey needs the Léman Pass only if one of its places is abroad: a station with country `FR`, or a zone other than 10.

How a journey is matched (as in `utils/gtfs_router/lemanpass.py`):

1. **Train part.** It runs from the first to the last rail ride that starts and ends at a stop of `lemanpass_stations.csv`.
2. **First and last places.**
   - With a train part, the first places are its boarding station plus the Unireso zones of the rides before it. The last places are its alighting station plus the zones of the rides after it.
   - With buses and trams only, they are the zones of the first and last stops (the lowest zone for a stop on a border).
3. **Abroad.** The journey needs the Léman Pass only if one of these places, or a zone it crosses, is abroad.
4. **Fitting products.** A product fits when both of these hold:
   - The zones of the rides inside the train part are `piece`s. With buses and trams only, every zone of the rides must be a `piece`.
   - `end_a` holds the first places and `end_b` the last ones, or the reverse; or its `cover` holds all of them.
5. **Price.** The cheapest fitting product counts, over the traveller's columns: `adult`, plus `child` from 6 to 15, `youth` from 16 to 25 and `reduced` with the half-fare card, in the traveller's class. A single ticket covers the train part and the Unireso rides right before and after it. It is kept when it is cheaper than the usual tickets for those rides. With the Léman Pass, Unireso only sells Zone 10 tickets: a usual way of paying that has a Unireso ticket with another zone (abroad) is dropped, even when it is cheaper. It is kept only if no Léman Pass ticket fits the journey. The Java model applies the same rule.
