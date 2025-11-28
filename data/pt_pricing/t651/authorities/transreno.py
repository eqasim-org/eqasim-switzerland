import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [24, 30]

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

        zone_pattern = r"(?:\d+\s*(?:/\s*\d+)*)"

        with open(page_txt, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                m = re.match(rf"^\s*(.+?)\s+({zone_pattern})\s*$", line)

                if not m:
                    continue

                stop_name, stop_zones = m.groups()

                stop_zones = [int(z) for z in re.split(r"\s*/\s*", stop_zones.strip()) if z]

                stations.append((stop_name, stop_zones, "Chur", "Transreno"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Maladers, Sax", [150], "Chur", "Transreno"))

    return stations