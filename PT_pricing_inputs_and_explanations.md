# Important note
The processings stages heavily rely on PDF processing libraries only available in ubuntu shell scripts. They are accessible in Python via the subprocess library. However, they are not available in Euler as they are not provided in the standard modules. I could not find any solution to still work with them from Euler, and all pdf-to-text Python library do not achieve the required quality standards. Until someone finds a better solution, I would thus suggest to keep these stages as "only locally runable" stages. Anyways, the zone assignment process and the SBB network construction should be quite stable over time, so once we are happy with the inputs, it should not be necessary to re-run the stages each time the pipeline is run. Copying the provided input files should be enough.

# Input files and folder structure
Folder path: ch_data/pipeline/pt_pricing. 

Most of the files are regularly updated by the public transport operators and are published on the [Alliance Swiss Pass website](https://www.allianceswisspass.ch/de/tarife-vorschriften/uebersicht). However, in some cases, the information provided is not sufficient to create the input. A solution was to use the archive built by Sebastian Hörl and Joe Molloy in 2018 when they worked on the [ch-pt-pricing library](https://github.com/joemolloy/ch-pt-pricing/tree/main/ch_pt_utils) back in 2018. These data sets are available in the shared IVT servers (let's hope no one gets rid of them...) at /nas/ivtmatsim/pt_data.

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

For Mobilis (operating in the Canton Vaud), the T651 PDF document only contains a network map, but Martin Repoux from TL shared an excel file (flph_Liste des arrêts Mobilis _dès le 15.12.2024 modif dès le 01.06.2025.xlsx) that can be used as a stop registry.

Most of the PDFs were extracted from [Alliance Swiss Pass](https://www.allianceswisspass.ch/de/tarife-vorschriften/uebersicht/Tarife-der-Verbuende).


