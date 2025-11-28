import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [53, 70]

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

        if page < 70:
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

                    stations.append((stop1_name, stop1_zones, "Luzern", "Passepartout"))
                    stations.append((stop2_name, stop2_zones, "Luzern", "Passepartout"))

        else:
            with open(page_txt, encoding = "utf-8") as f:
                for line in f:
                    line = line.strip()
                    m = re.match(r"^\s*(.+?)\s+([\d/]+)\s+[A-Z0-9]+$", line)

                    if not m:
                        continue

                    stop1_name, stop1_zones = m.groups()

                    stop1_zones = [int(z) for z in stop1_zones.replace(" ", "").split("/")]

                    stations.append((stop1_name, stop1_zones, "Luzern", "Passepartout"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Waldibrücke", [10, 28], "Luzern", "Passepartout"))
    stations.append(("Waldibrücke, Bahnhof", [10, 28], "Luzern", "Passepartout"))
    stations.append(("Emmenbrücke, Strassenkreuz", [10, 26], "Luzern", "Passepartout"))
    stations.append(("Stächenrain", [10, 23, 26], "Luzern", "Passepartout"))
    stations.append(("Grosswangen, Ed. Huber-Strasse", [45], "Luzern", "Passepartout"))
    stations.append(("Vordemwald, Pflegeheim Sennhof", [76], "Luzern", "Passepartout"))

    return stations