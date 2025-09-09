import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [31, 53]

def create_stations(pdf_path, temp_dir):

    stations = []

    for page in range(PAGES[0], PAGES[1] + 1):

        page_pdf = Path(temp_dir) / f"{page}.pdf"
        page_txt = Path(temp_dir) / f"{page}.txt"

        subprocess.run([
            "pdf-stapler", "sel", str(pdf_path), str(page), str(page_pdf)
        ], check=True)

        subprocess.run([
            "pdftotext", "-layout", str(page_pdf), str(page_txt)
        ], check=True)

        if page < 53:
            with open(page_txt, encoding = "utf-8") as f:
                for line in f:
                    line  = line.strip()
                    zone_pattern = r"(?:\d+\s*(?:/\s*\d+)*)"
                    m = re.match(rf"^\s*(.+?)\s+({zone_pattern})\s+\S+\s+(.+?)\s+({zone_pattern})\s+\S+$", line)

                    if not m:
                        continue

                    stop1_name, stop1_zones, stop2_name, stop2_zones = m.groups()

                    if "          " in stop1_name:
                        stop1_name = stop1_name.split("          ")[-1]
                    if "          " in stop2_name:
                        stop2_name = stop2_name.split("          ")[-1]

                    stop1_zones = [int(z) for z in stop1_zones.replace(" ", "").split("/")]
                    stop2_zones = [int(z) for z in stop2_zones.replace(" ", "").split("/")]

                    if stop1_name == "Wiggen, Dürrenbach":
                        stop1_zones = [145]

                    if stop2_name == "Wiggen, Dürrenbach":
                        stop2_zones = [145]

                    stations.append((stop1_name, stop1_zones, "Bern", "Libero"))
                    stations.append((stop2_name, stop2_zones, "Bern", "Libero"))

        else:
            with open(page_txt, encoding = "utf-8") as f:
                for line in f:
                    line = line.strip()
                    m = re.match(r"^\s*(.+?)\s+([\d/]+)\s+[A-Z0-9]+$", line)

                    if not m:
                        continue

                    stop1_name, stop1_zones = m.groups()

                    stop1_zones = [int(z) for z in stop1_zones.replace(" ", "").split("/")]

                    if stop1_name == "Wiggen, Dürrenbach":
                        stop1_zones = [145]

                    stations.append((stop1_name, stop1_zones, "Bern", "Libero"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Thörishaus Station", [112], "Bern", "Libero"))
    stations.append(("Thörishaus Station, Bahnhof", [112], "Bern", "Libero"))
    stations.append(("Oberbottigen, Chäs und Brot", [112], "Bern", "Libero"))
    stations.append(("Oberbottigen, Flühli", [112], "Bern", "Libero"))
    stations.append(("Oberbottigen, Dorf", [112], "Bern", "Libero"))
    stations.append(("Brünnemoos b. Rosshäusern", [177], "Bern", "Libero"))
    stations.append(("Zilacher b. Rosshäusern", [177], "Bern", "Libero"))
    stations.append(("Kleingümmenen", [698], "Bern", "Libero"))
    stations.append(("Gümmenen, Dorf", [698], "Bern", "Libero"))
    stations.append(("Gümmenen, Stutz", [177], "Bern", "Libero"))
    stations.append(("Mauss", [177], "Bern", "Libero"))
    stations.append(("Trüllern bei Mühleberg", [177], "Bern", "Libero"))
    stations.append(("Allenlüften, Dorf", [177], "Bern", "Libero"))
    stations.append(("Mühleberg, Dällenbach", [177], "Bern", "Libero"))
    stations.append(("Mühleberg, Post", [177], "Bern", "Libero"))
    stations.append(("Dotzigen, alte Käserei", [311], "Bern", "Libero"))
    stations.append(("Busswil BE", [311], "Bern", "Libero"))
    stations.append(("Aarberg, Scheueracker", [310], "Bern", "Libero"))
    stations.append(("Bargen BE, Gemeindeverwaltung", [310], "Bern", "Libero"))
    stations.append(("Bargen BE, Bahnhof", [310], "Bern", "Libero"))
    stations.append(("Kallnach, Bahnhof", [177], "Bern", "Libero"))
    stations.append(("Kallnach, Mitteldorf", [177], "Bern", "Libero"))
    stations.append(("Fräschels, Dorf", [56], "Bern", "Libero"))
    stations.append(("Gampelen, Bahnhof", [313], "Bern", "Libero"))
    stations.append(("Mullen, Dorf", [313], "Bern", "Libero"))
    stations.append(("Evilard, La Lisière", [301], "Bern", "Libero"))
    stations.append(("Magglingen,Zum Alten Schweizer", [315], "Bern", "Libero"))
    stations.append(("Magglingen, Alte Sporthalle", [315], "Bern", "Libero"))
    stations.append(("Magglingen, End der Welt", [315], "Bern", "Libero"))
    stations.append(("La Heutte, Boccalino", [321], "Bern", "Libero"))
    stations.append(("Sonceboz-Sombeval", [321], "Bern", "Libero"))
    stations.append(("Sonceboz-Sombeval, gare", [321], "Bern", "Libero"))
    stations.append(("Tramelan, Dessous", [351], "Bern", "Libero"))
    stations.append(("Pontenet, gare", [342], "Bern", "Libero"))
    stations.append(("Court, église", [343], "Bern", "Libero"))
    stations.append(("Oberdorf SO, Bahnhof", [201], "Bern", "Libero"))
    stations.append(("Lommiswil, Dorfstrasse/Im Holz", [201], "Bern", "Libero"))
    stations.append(("Hellsau, Feuerwehrmagazin", [217], "Bern", "Libero"))
    stations.append(("Höchstetten, Käsereistrasse", [217], "Bern", "Libero"))
    stations.append(("Höchstetten, Dorfstrasse", [217], "Bern", "Libero"))
    stations.append(("Willadingen, Moosgasse", [217], "Bern", "Libero"))
    stations.append(("Willadingen, Dorf", [217], "Bern", "Libero"))
    stations.append(("Wynigen", [152], "Bern", "Libero"))
    stations.append(("Kestenholz, Dörfli", [281], "Bern", "Libero"))
    stations.append(("Murgenthal, Fahracker", [191], "Bern", "Libero"))
    stations.append(("Murgenthal, Post", [191], "Bern", "Libero"))
    stations.append(("Murgenthal, Bahnhof", [191], "Bern", "Libero"))
    stations.append(("St. Urban, Bahnhof", [191], "Bern", "Libero"))
    stations.append(("Gutenburg, Badstrasse", [192], "Bern", "Libero"))
    stations.append(("Madiswil, Bahnhof", [192], "Bern", "Libero"))
    stations.append(("Lindenholz, Bahnhof", [194], "Bern", "Libero"))
    stations.append(("Huttwil Sportzentrum", [180], "Bern", "Libero"))
    stations.append(("Gondiswil, Haltestelle", [180], "Bern", "Libero"))
    stations.append(("Häusermoos", [157], "Bern", "Libero"))
    stations.append(("Mussachen", [157, 181], "Bern", "Libero"))
    stations.append(("Schafhausen i.E., Thunstrasse", [154,156], "Bern", "Libero"))
    stations.append(("Goldbach BE, Sonnhalde", [154,156], "Bern", "Libero"))
    stations.append(("Trubschachen, Kröschenbrunnen", [145], "Bern", "Libero"))
    stations.append(("Schangnau, Käserei", [447], "Bern", "Libero"))
    stations.append(("Brenzikofen, Bahnhof", [701, 711], "Bern", "Libero"))
    stations.append(("Kirchdorf BE, Post", [626], "Bern", "Libero"))
    stations.append(("Kaufdorf, Moosstrasse", [626, 126], "Bern", "Libero"))
    stations.append(("Lohnstorf, Dorf", [626], "Bern", "Libero"))
    stations.append(("Kalchstätten, Eigen", [628], "Bern", "Libero"))
    stations.append(("Vorderfultigen, Kuhweid", [126],  "Bern", "Libero"))
    stations.append(("Hinterfultigen, Post", [126],  "Bern", "Libero"))
    stations.append(("Riedbach, Bahnhof", [101, 112],  "Bern", "Libero"))
    stations.append(("Riedbach", [101, 112],  "Bern", "Libero"))
    stations.append(("Heggidorn", [177],  "Bern", "Libero"))
    stations.append(("Oberwohlen BE", [101],  "Bern", "Libero"))
    stations.append(("Illiswil", [113],  "Bern", "Libero"))
    stations.append(("Möriswil, Abzw.", [113],  "Bern", "Libero"))
    stations.append(("Ortschwaben, Schützenrain", [113],  "Bern", "Libero"))
    stations.append(("Ortschwaben, Weissenstein Abzw", [113],  "Bern", "Libero"))
    stations.append(("Ortschwaben, Weissenstein Abzw", [113],  "Bern", "Libero"))
    stations.append(("Ortschwaben, Aetzikofen Abzw.", [113],  "Bern", "Libero"))
    stations.append(("Grächwil", [113],  "Bern", "Libero"))
    stations.append(("Kallnach, Mitteldorf", [177],  "Bern", "Libero"))
    stations.append(("Kallnach, Bahnhof", [177],  "Bern", "Libero"))
    stations.append(("Lüterswil, Mehrzweckhalle", [229],  "Bern", "Libero"))
    stations.append(("Grenchen, Lingeriz 91", [250],  "Bern", "Libero"))
    stations.append(("Biel/Bienne, Am Stutz", [300],  "Bern", "Libero"))
    stations.append(("Tüscherz, Bahnhof", [301],  "Bern", "Libero"))
    stations.append(("Oberlindach, Käserei", [101],  "Bern", "Libero"))
    stations.append(("Utzigen", [115],  "Bern", "Libero"))
    stations.append(("Wikartswil, Dorf", [126],  "Bern", "Libero"))
    stations.append(("Richigen, Graben", [126],  "Bern", "Libero"))
    stations.append(("Metzgerhüsi", [126],  "Bern", "Libero"))
    stations.append(("Ried bei Worb", [126],  "Bern", "Libero"))
    stations.append(("Bigenthal, Dorfstrasse", [146, 156],  "Bern", "Libero"))
    stations.append(("Rohrbach b. Riggisberg,Schulh.", [626],  "Bern", "Libero"))
    stations.append(("Latterbach, Burgholz", [840],  "Bern", "Libero"))    
    stations.append(("Latterbach, Dorf", [840],  "Bern", "Libero")) 
    stations.append(("Erlenbach i.S., Marktplatz", [840],  "Bern", "Libero"))
    stations.append(("Erlenbach i.S., Stockhornbahn", [840],  "Bern", "Libero"))
    stations.append(("Därstetten, Bahnhof", [841],  "Bern", "Libero"))
    stations.append(("Weissenburg, Dorf", [841],  "Bern", "Libero"))
    stations.append(("Oberwil i.S., Bahnhof", [841, 842],  "Bern", "Libero"))
    stations.append(("Enge im Simmental", [842],  "Bern", "Libero"))
    stations.append(("Enge i.S., Abzweigung Bahnhof", [842],  "Bern", "Libero"))
    stations.append(("Grubenwald, Hauptstrasse", [842, 843],  "Bern", "Libero"))
    stations.append(("Lauenen b. Gstaad, Bochte", [846],  "Bern", "Libero"))
    stations.append(("St. Stephan, Bawald", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Kirche", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Ried", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Bleiki", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Lengenbrand", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Nageldach", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Grodey", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, Bahnhof", [844],  "Bern", "Libero"))
    stations.append(("St. Stephan, altes Moosschulh.", [844],  "Bern", "Libero"))
    stations.append(("Matten i.S., Dorf", [844],  "Bern", "Libero"))
    stations.append(("Matten i.S., Färmelbach", [844],  "Bern", "Libero"))
    stations.append(("Matten i.S., Stocken", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Rufeli", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Gütsch", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Äussere Bleiken", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Schadauli", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Lischmatte", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Honeggli", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Stein", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Erlebnisbad", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Talstation Betelberg", [844],  "Bern", "Libero"))
    stations.append(("Lenk, Krummenbach", [844],  "Bern", "Libero"))
    stations.append(("Adelboden, Schermtanne", [835],  "Bern", "Libero"))
    stations.append(("Adelboden, Gilbach, des Alpes", [835],  "Bern", "Libero"))
    stations.append(("Zweilütschinen, Bahnhof", [820],  "Bern", "Libero"))
    stations.append(("Lütschental, Gemeindehaus", [820],  "Bern", "Libero"))
    stations.append(("Burglauenen, Bahnhof", [821],  "Bern", "Libero"))
    stations.append(("Schwendi BE,Grindelwaldstrasse", [821],  "Bern", "Libero"))
    stations.append(("Obermoos", [151],  "Bern", "Libero"))
    stations.append(("Schwanden i.E., Nesselgraben", [156],  "Bern", "Libero"))
    stations.append(("Rüderswil, Kirche", [156],  "Bern", "Libero"))
    stations.append(("Zollbrück, Sekundarschulhaus", [141],  "Bern", "Libero"))
    stations.append(("Emmenmatt, Moosbad", [146],  "Bern", "Libero"))
    stations.append(("Häusernmoos", [157],  "Bern", "Libero"))
    stations.append(("Rohrbach, Hauptstrasse", [194],  "Bern", "Libero"))
    stations.append(("Kleindietwil, Post", [194],  "Bern", "Libero"))
    stations.append(("Kleindietwil, Hauptstrasse", [194],  "Bern", "Libero"))
    stations.append(("Kühlewil, Heim", [116],  "Bern", "Libero"))
    stations.append(("Ursellen", [130],  "Bern", "Libero"))
    stations.append(("Stalden i.E., Thunstrasse", [130],  "Bern", "Libero"))
    stations.append(("Hüswil, Bahnhof", [180], "Bern", "Libero"))
    stations.append(("Gondiswil, Gemeindehaus", [180], "Bern", "Libero"))
    stations.append(("Aeschlen ob Gunten, Dorf", [701], "Bern", "Libero"))
    stations.append(("Schwanden i. E., Nesselgraben", [156], "Bern", "Libero"))	
    stations.append(("Gammenthal", [157], "Bern", "Libero"))	

    return stations