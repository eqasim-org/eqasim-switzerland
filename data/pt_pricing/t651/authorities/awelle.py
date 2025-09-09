import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [61, 97]

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
                m = re.match(r"^\s*(.+?)\s+(\d+(?:/\d+)*)\s+", line)

                if not m:
                    continue

                stop1_name, stop1_zones = m.groups()

                stop1_zones = [int(z) for z in stop1_zones.replace(" ", "").split("/")]

                stations.append((stop1_name, stop1_zones, "Aarau", "Awelle"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Safenwil, Bahnhof", [521, 522], "Aarau", "Awelle"))
    stations.append(("Egliswil, Engelgasse", [512], "Aarau", "Awelle"))
    stations.append(("Mühlau, Dorf", [535], "Aarau", "Awelle"))
    stations.append(("Zufikon, Belvédère Bahnhof", [574], "Aarau", "Awelle"))
    stations.append(("Eiken, Volg", [590], "Aarau", "Awelle"))
    stations.append(("Tecknau, Bahnhof", [524], "Aarau", "Awelle"))
    stations.append(("Mühlau, Dorf", [535], "Aarau", "Awelle"))
    stations.append(("Hendschiken, Bahnhof", [530], "Aarau", "Awelle"))
    stations.append(("Baden, Pinte", [570], "Aarau", "Awelle"))
    stations.append(("Baden, Rüteli", [570], "Aarau", "Awelle"))
    stations.append(("Dättwil AG, Kantonsspital", [570], "Aarau", "Awelle"))
    stations.append(("Baden, Segelhof", [570], "Aarau", "Awelle"))
    stations.append(("Baden, Rütihof Moosstrasse", [571], "Aarau", "Awelle"))
    stations.append(("Baden, Rütihof Haberacher", [571], "Aarau", "Awelle"))
    stations.append(("Baden, Rütihof Bohnacker", [571], "Aarau", "Awelle"))
    stations.append(("Klingnau, Dorfstrasse", [562], "Aarau", "Awelle"))
    stations.append(("Rietheim, Bahnhof", [563], "Aarau", "Awelle"))
    stations.append(("Spreitenbach, Glattler", [572], "Aarau", "Awelle"))


    return stations