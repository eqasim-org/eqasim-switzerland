import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [33, 43]

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
                
                m = re.match(r"^\s*(.+?)\s+(\d+(?:/\d+)*)\s+[A-Z]+$", line)
        
                if not m:
                    continue

                stop_name, zones_raw = m.groups()
                stop_name = stop_name.strip()
                zones = [int(z) for z in zones_raw.replace(" ", "").split("/")]

                stations.append((stop_name, zones, "Jura", "CTJU"))  


    shutil.rmtree(Path(temp_dir))

    stations.append(("Boncourt, bif. Stand", [22], "Jura", "CTJU")) 
    stations.append(("Boncourt, poste", [22], "Jura", "CTJU")) 
    stations.append(("Boncourt, piscine", [22], "Jura", "CTJU")) 
    stations.append(("Delle, gare", [22], "Jura", "CTJU")) 
    stations.append(("Bure, Caserne", [22], "Jura", "CTJU")) 
    stations.append(("Courchavon, route de la Gare", [20], "Jura", "CTJU")) 
    stations.append(("Vendlincourt, bif. gare", [23], "Jura", "CTJU")) 
    stations.append(("Miécourt", [24], "Jura", "CTJU")) 
    stations.append(("Le Boéchet, gare", [42], "Jura", "CTJU")) 
    stations.append(("Les Bois, gare", [42], "Jura", "CTJU")) 
    stations.append(("Mont-Tramelan, Ferme-Hourier", [41], "Jura", "CTJU")) 
    stations.append(("Mont-Tramelan, La Paule", [41], "Jura", "CTJU")) 
    stations.append(("Les Genevez JU, ch.de Tramelan", [40], "Jura", "CTJU")) 

    return stations