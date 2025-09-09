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


