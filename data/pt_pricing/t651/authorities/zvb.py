import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [42, 52]

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
                m = re.match(r"^\s*([A-ZÄÖÜÉÈÀ\s'\-]+?)\s+[A-Z]+\s+(\d+(?:/\d+)*)", line)

                if not m:
                    continue

                stop1_name, stop1_zones = m.groups()
                stop1_name  = stop1_name.title()
                stop1_zones = [int(z) for z in stop1_zones.replace(" ", "").split("/")]

                name_parts = stop1_name.split(" ")
                if name_parts[0][-1] != ",":
                    name_parts[0] = name_parts[0] + ","
                    stop1_name = " ".join(name_parts)

                stop1_name = stop1_name.replace("Küssnacht Am Rigi", "Küssnacht am Rigi")

                stations.append((stop1_name, stop1_zones, "Zug", "ZVB"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Cham, Bahnhof", [622], "Zug", "ZVB"))
    stations.append(("Cham", [622], "Zug", "ZVB"))
    stations.append(("Cham Alpenblick", [622], "Zug", "ZVB"))
    stations.append(("Steinhausen Rigiblick", [623], "Zug", "ZVB"))
    stations.append(("Steinhausen, Zugerland EKZ", [623], "Zug", "ZVB"))
    stations.append(("Steinhausen", [623], "Zug", "ZVB"))
    stations.append(("Steinhausen, Hinterberg Bhf", [623], "Zug", "ZVB"))
    stations.append(("Blickensdorf, H. Waldmann-Str", [623], "Zug", "ZVB"))
    stations.append(("Baar, Bachtalerhöhe", [623], "Zug", "ZVB"))
    stations.append(("Blickensdorf, Dorf", [623], "Zug", "ZVB"))
    stations.append(("Baar, Blegistrasse", [623], "Zug", "ZVB"))
    stations.append(("Inwil bei Baar, Huobhof", [623], "Zug", "ZVB"))
    stations.append(("Allenwinden, Inkenberg", [623], "Zug", "ZVB"))
    stations.append(("Allenwinden, St. Meinrad", [623], "Zug", "ZVB"))
    stations.append(("Allenwinden, Dorf", [623], "Zug", "ZVB"))
    stations.append(("Allenwinden, Egg", [623], "Zug", "ZVB"))
    stations.append(("Neuägeri, Schmittli", [623, 625], "Zug", "ZVB"))
    stations.append(("Neuägeri, Alte Post", [625], "Zug", "ZVB"))
    stations.append(("Neuägeri, Rössli", [625], "Zug", "ZVB"))
    stations.append(("Edlibach, Nidfuren", [625], "Zug", "ZVB"))
    stations.append(("Edlibach, Hündlital", [625], "Zug", "ZVB"))
    stations.append(("Edlibach, Sonnhalde", [625], "Zug", "ZVB"))
    stations.append(("Menzingen,Institut/Bernardapl.", [625], "Zug", "ZVB"))
    stations.append(("Schönegg", [610, 611], "Zug", "ZVB"))
    stations.append(("Zugerberg", [613], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug, Stolzengraben", [610], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug, Kreuz", [610], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug, Rigiblick", [610], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug, Räbmatt", [610], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug,Klinik Zugersee", [610], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug, Räbmatt", [610], "Zug", "ZVB"))
    stations.append(("Oberwil b. Zug, Murpfli", [610], "Zug", "ZVB"))
    stations.append(("Walchwil, Hörndli", [625], "Zug", "ZVB"))
    stations.append(("Walchwil Hörndli", [625], "Zug", "ZVB"))
    stations.append(("Walchwil, St. Adrian", [625], "Zug", "ZVB"))
    stations.append(("Morgarten, Haselmatt", [625], "Zug", "ZVB"))
    stations.append(("Morgarten, Hotel", [625], "Zug", "ZVB"))
    stations.append(("Morgarten, Denkmal", [625], "Zug", "ZVB"))
    stations.append(("Morgarten, Sydefade", [625], "Zug", "ZVB"))
    stations.append(("Sattel, Bahnhofstrasse", [636, 637], "Zug", "ZVB"))
    stations.append(("Arth-Goldau", [637, 638], "Zug", "ZVB"))
    stations.append(("Arth-Goldau, Bahnhof", [637, 638], "Zug", "ZVB"))


    return stations