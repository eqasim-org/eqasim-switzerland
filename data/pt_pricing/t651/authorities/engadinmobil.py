import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [32, 34]

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
                m = re.match(
                    r"^\s*\d+\s+(.+?)\s+(\d+(?:/\d+)*)\s+[A-ZÄÖÜÉÈÀ'’\-]+(?:/[A-ZÄÖÜÉÈÀ'’\-]+)*(?:\s+[A-ZÄÖÜÉÈÀ'’\-]+)*(?:\s+x)?\s*$",
                    line
                )

                if not m:
                    continue

                stop_name, zones_raw = m.groups()
                stop_name = stop_name.strip()
                
                zones = [int(z) for z in zones_raw.replace(" ", "").split("/")]

                stations.append((stop_name, zones, "Engadin", "EngadinMobil"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("S-chanf, Zielgelände ESM", [42], "Engadin", "EngadinMobil"))
    stations.append(("S-chanf, Marathon", [42], "Engadin", "EngadinMobil"))
    stations.append(("S-chanf, Punt da Crap", [42], "Engadin", "EngadinMobil"))
    stations.append(("La Punt Chamues-ch,Abzw Albula", [41], "Engadin", "EngadinMobil"))
    stations.append(("La Punt Chamues-ch", [41], "Engadin", "EngadinMobil"))
    stations.append(("La Punt Chamues-ch, Bahnhof", [41], "Engadin", "EngadinMobil"))
    stations.append(("Celerina, Rosatsch", [10], "Engadin", "EngadinMobil"))
    stations.append(("Celerina, Vietta Saluver", [10], "Engadin", "EngadinMobil"))
    stations.append(("Celerina, Pradatsch Suot", [10], "Engadin", "EngadinMobil"))
    stations.append(("Celerina, Cresta Run", [10], "Engadin", "EngadinMobil"))
    stations.append(("Silvaplana, Vallun", [11], "Engadin", "EngadinMobil"))
    stations.append(("Silvaplana, Baselgia", [11], "Engadin", "EngadinMobil"))
    stations.append(("Silvaplana, Plazza dal Güglia", [11], "Engadin", "EngadinMobil"))
    stations.append(("Silvaplana, Plazza dal Mastrel", [11], "Engadin", "EngadinMobil"))
    stations.append(("Silvaplana, Munteratsch", [11], "Engadin", "EngadinMobil"))
    stations.append(("Silvaplana, Mandra", [11], "Engadin", "EngadinMobil"))
    stations.append(("Surlej, Cristins", [11], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Maria, Föglias", [11], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Baselgia, Margna", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Baselgia, Randolina", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Baselgia, Muttals", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Baselgia, Silserhof", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Maria, Alpenrose", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Maria, Chesa Fonio", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Maria,Chesa Cumünela", [12], "Engadin", "EngadinMobil"))
    stations.append(("Sils/Segl Maria, Edelweiss", [12], "Engadin", "EngadinMobil"))


    return stations
