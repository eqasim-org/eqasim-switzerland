import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [37, 51]

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
                line  = line.strip()
                m = re.match(r"^\s*(\d+)\s+(\d+)\s+(.*)$", line)

                if not m:
                    continue

                _, _, rest = m.groups()
                tokens = rest.split()

                split_index = None
                for i in range(len(tokens)):
                    if all(t.isdigit() for t in tokens[i:]): 
                        split_index = i
                        break

                if split_index is not None:
                    station_name = " ".join(tokens[:split_index])
                    zones = [int(z) for z in tokens[split_index:]]
                else:
                    station_name = " ".join(tokens)
                    zones = []

                stations.append((station_name, zones, "Neuchâtel", "OndeVerte"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Vaumarcus, giratoire", [15], "Neuchâtel", "OndeVerte"))
    stations.append(("Travers, Pont de la Presta", [33], "Neuchâtel", "OndeVerte"))
    stations.append(("Travers, gare", [30], "Neuchâtel", "OndeVerte"))
    stations.append(("Noiraigue, gare", [30], "Neuchâtel", "OndeVerte"))
    stations.append(("Champ-du-Moulin, gare", [11, 30], "Neuchâtel", "OndeVerte"))
    stations.append(("Les Taillères, restaurant", [32], "Neuchâtel", "OndeVerte"))
    stations.append(("Les Geneveys-s.C., Mont-Racine", [30], "Neuchâtel", "OndeVerte"))
    stations.append(("Boudevilliers, Bottes", [11, 30], "Neuchâtel", "OndeVerte"))
    stations.append(("Boudevilliers, Malvilliers", [11, 30], "Neuchâtel", "OndeVerte"))
    stations.append(("Thielle, Wavre", [11], "Neuchâtel", "OndeVerte"))

    return stations