import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [58, 74]


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

        with open(page_txt, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                m = re.match(r"^\s*([A-ZÄÖÜÉÈÀ'\-,\s]+?)\s+((?:\d+\s*)+?)\s{2,}\d+\s*$", line)

                if not m:
                    continue

                stop_name, zones_raw = m.groups()
                stop_name  = stop_name.title()
                zones = [int(z) for z in zones_raw.split()]

                if stop_name == "Goldau, Waage":
                    zones = [674]

                stop_name = stop_name.replace("Küssnacht Am Rigi", "Küssnacht am Rigi")

                stations.append((stop_name, zones, "Schwyz", "TVSZ"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Immensee, Bahnhof", [676, 675], "Schwyz", "TVSZ"))
    stations.append(("Brunnen, See/Schiffstation", [670], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Grünenwald", [685], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Selgis", [685], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Hesingen", [685, 686], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Mühlestuden", [686], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Kappelmatt", [686], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Seilb. Illgau", [686], "Schwyz", "TVSZ"))
    stations.append(("Ried (Muotathal), Vord. Brücke", [686], "Schwyz", "TVSZ"))
    stations.append(("Muotathal, hintere Brücke", [686], "Schwyz", "TVSZ"))
    stations.append(("Bisisthal, vorder Seeberg", [687], "Schwyz", "TVSZ"))
    stations.append(("Bisisthal, Sahli Seilbahnstat.", [688], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Allerheiligen", [670], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Dorf", [670], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Rotenfluebahn", [670], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Bol", [670], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Stalden", [670], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Aufiberg/Gruobi", [685], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Windstock", [685], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Chaisten", [685], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Meinradsrank", [685], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Lauenen", [685], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Handgruobi", [685], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Gründel/St Karl", [686], "Schwyz", "TVSZ"))
    stations.append(("Rickenbach SZ, Oberberg", [684], "Schwyz", "TVSZ"))
    stations.append(("Hoch-Ybrig, Talst. Weglosen", [683], "Schwyz", "TVSZ"))
    stations.append(("Hoch-Ybrig, Talst. Laucheren", [683], "Schwyz", "TVSZ"))
    stations.append(("Unteriberg, vord.Schmalzgruben", [683], "Schwyz", "TVSZ"))
    stations.append(("Unteriberg, hint.Schmalzgruben", [683], "Schwyz", "TVSZ"))
    stations.append(("Studen SZ, Dörfli", [682], "Schwyz", "TVSZ"))
    stations.append(("Studen SZ, Adelmatt", [682], "Schwyz", "TVSZ"))
    stations.append(("Studen SZ, Ochsenboden", [682], "Schwyz", "TVSZ"))
    stations.append(("Egg SZ, Postplatz", [679], "Schwyz", "TVSZ"))
    stations.append(("Egg SZ, Eintracht", [679], "Schwyz", "TVSZ"))
    stations.append(("Egg SZ, Langrüti", [679], "Schwyz", "TVSZ"))
    stations.append(("Egg SZ, Roblosen", [679], "Schwyz", "TVSZ"))
    stations.append(("Schindellegi-Feusisberg, Bhf.", [680], "Schwyz", "TVSZ"))
    stations.append(("Einsiedeln, Kornhausstrasse", [679], "Schwyz", "TVSZ"))
    stations.append(("Trachslau, altes Schulhaus", [679], "Schwyz", "TVSZ"))
    stations.append(("Trachslau, alte Säge", [679], "Schwyz", "TVSZ"))
    stations.append(("Brunni SZ, Rest. Brunni", [685], "Schwyz", "TVSZ"))
    stations.append(("Brunni SZ, Talstation LBH", [685], "Schwyz", "TVSZ"))
    stations.append(("Steinen, Rest. Löwen", [670], "Schwyz", "TVSZ"))
    stations.append(("Goldau, Schützenhaus", [674, 675], "Schwyz", "TVSZ"))

    return stations