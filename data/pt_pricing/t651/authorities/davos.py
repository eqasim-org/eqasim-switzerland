import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [33, 33]

def create_stations(pdf_path, temp_dir):
    stations = []

    zone_pattern = r"(?:\d+\s*(?:/\s*\d+)*)"

    pattern_two = re.compile(
        rf"^\s*({zone_pattern})\s+(.+?)\s+({zone_pattern})\s+(.+?)$"
    )

    pattern_one = re.compile(
        rf"^\s*({zone_pattern})\s+(.+?)$"
    )

    for page in range(PAGES[0], PAGES[1] + 1):

        page_pdf = Path(temp_dir) / f"{page}.pdf"
        page_txt =  Path(temp_dir) / f"{page}.txt"

        subprocess.run([
            "pdf-stapler", "sel", str(pdf_path), str(page), str(page_pdf)
        ], check=True)

        subprocess.run([
            "pdftotext", "-layout", str(page_pdf), str(page_txt)
        ], check=True)

        with open(page_txt, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                
                m2 = pattern_two.match(line)
                if m2 and not "Haltestellenverzeichnis" in line.strip():
                    zones1, stop1, zones2, stop2 = m2.groups()
                    zones1 = [int(z) for z in re.split(r"\s*/\s*", zones1.strip()) if z]
                    zones2 = [int(z) for z in re.split(r"\s*/\s*", zones2.strip()) if z]
                    stations.append((stop1, zones1, "Davos", "Davos"))
                    stations.append((stop2, zones2, "Davos", "Davos"))
                    continue

                m1 = pattern_one.match(line)
                if m1 and not "Haltestellenverzeichnis" in line.strip():
                    zones, stop = m1.groups()
                    zones = [int(z) for z in re.split(r"\s*/\s*", zones.strip()) if z]
                    stations.append((stop, zones, "Davos", "Davos"))                

    shutil.rmtree( Path(temp_dir))

    stations.append(("Davos Laret, Landhaus", [410], "Davos", "Davos"))
    stations.append(("Davos Wolfgang", [410, 400], "Davos", "Davos"))
    stations.append(("Davos Wolfgang (Bus)", [410, 400], "Davos", "Davos"))
    stations.append(("Davos Wolfgang, Hochgebirgskl.", [400], "Davos", "Davos"))
    stations.append(("Davos Wolfgang, Seebüel.", [400], "Davos", "Davos"))
    stations.append(("Wägerhus/Abzw. Jöriseen", [434], "Davos", "Davos"))
    stations.append(("Davos Clavadel,Abzw. Wanderweg", [450], "Davos", "Davos"))
    stations.append(("Davos Wiesen, Valdanna", [422], "Davos", "Davos"))

    return stations
