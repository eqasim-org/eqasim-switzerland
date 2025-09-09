import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [36, 58]

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
                m = re.match(r"^\s*(.+?)\s+([\d\s/]+)\s*$", line)

                if not m:
                    continue

                station_name = m.group(1).strip()
                zones_raw    = m.group(2).strip()

                station_name = station_name.replace(" - ", "-")
                station_name = station_name.replace(" – ", "-")
                station_name = station_name.replace("Charmey (Gruyère)", "-")

                if not "2018" in station_name:
                    zones = [int(z) for z in zones_raw.replace("/", " ").split()]

                    if station_name == "La Roche FR, La Berra":
                        zones = [26]
                    if station_name == "Allières":
                        zones = [23, 24]
                    if station_name == "Les Cases":
                        zones = [24]
                    stations.append((station_name, zones, "Fribourg", "Frimobil"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Ecublend-Rue, gare", [46], "Fribourg", "Frimobil"))
    stations.append(("Ecuvillens, Aérodrome", [11], "Fribourg", "Frimobil"))
    stations.append(("Semsales, gare", [41], "Fribourg", "Frimobil"))
    stations.append(("Lessoc, gare", [22], "Fribourg", "Frimobil"))
    stations.append(("Jaun, Bergbahnen", [27], "Fribourg", "Frimobil"))
    stations.append(("Jaunpass, Restaurant", [27], "Fribourg", "Frimobil"))
    stations.append(("Gruyères, Bike Revolution", [20], "Fribourg", "Frimobil"))
    stations.append(("Enney, zone d'activités", [21], "Fribourg", "Frimobil"))
    stations.append(("Vuadens, Les Kâ", [30], "Fribourg", "Frimobil"))
    stations.append(("Vuadens, Le Maupas", [30], "Fribourg", "Frimobil"))
    stations.append(("Vaulruz, village", [28], "Fribourg", "Frimobil"))
    stations.append(("Vaulruz, village", [28], "Fribourg", "Frimobil"))
    stations.append(("Epagny, centre", [20, 30], "Fribourg", "Frimobil"))
    stations.append(("Epagny, les Gottes", [20, 30], "Fribourg", "Frimobil"))
    stations.append(("Broc-Chocolaterie", [31], "Fribourg", "Frimobil"))
    stations.append(("Riaz, CO", [30], "Fribourg", "Frimobil"))
    stations.append(("Gumefens, village", [32], "Fribourg", "Frimobil"))
    stations.append(("Gumefens, lac", [32], "Fribourg", "Frimobil"))
    stations.append(("Rueyres-St-Laurent,En Borgogne", [37], "Fribourg", "Frimobil"))
    stations.append(("Posat, village", [33], "Fribourg", "Frimobil"))
    stations.append(("St. Silvester, Fifermoos", [17], "Fribourg", "Frimobil"))
    stations.append(("St. Silvester, Chrache", [17], "Fribourg", "Frimobil"))
    stations.append(("St. Silvester, Plenefy", [17], "Fribourg", "Frimobil"))
    stations.append(("Cottens FR, gare", [35, 36], "Fribourg", "Frimobil"))
    stations.append(("Cottens FR", [35, 36], "Fribourg", "Frimobil"))
    stations.append(("Neyruz FR", [35], "Fribourg", "Frimobil"))
    stations.append(("Neyruz FR, village", [35], "Fribourg", "Frimobil"))
    stations.append(("Rosé, gare", [11, 35], "Fribourg", "Frimobil"))
    stations.append(("Corserey, village", [82], "Fribourg", "Frimobil"))
    stations.append(("Mannens, école", [82], "Fribourg", "Frimobil"))
    stations.append(("Montagny-la-Ville, école", [82], "Fribourg", "Frimobil"))
    stations.append(("Léchelles, village", [82, 83], "Fribourg", "Frimobil"))
    stations.append(("Grolley, gare", [11, 83], "Fribourg", "Frimobil"))
    stations.append(("Formangueires, Moulin", [11], "Fribourg", "Frimobil"))
    stations.append(("Lossy, école", [11], "Fribourg", "Frimobil"))
    stations.append(("La Corbaz, ancienne école", [11], "Fribourg", "Frimobil"))
    stations.append(("Pensier, Buffet de la Gare", [11, 53], "Fribourg", "Frimobil"))
    stations.append(("Courtepin, gare", [52, 53], "Fribourg", "Frimobil"))
    stations.append(("Courtepin, gare", [52, 53], "Fribourg", "Frimobil"))
    stations.append(("Courtepin, gare", [52, 53], "Fribourg", "Frimobil"))
    stations.append(("Courtepin, gare", [52, 53], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Tennishalle", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Bahnhof", [11, 12], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Zelg", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Ganstrichweg", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Blumenrain", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Am Bach", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Chännelmatt", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Leimacker", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Käsereistrasse", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Brugerastrasse", [11], "Fribourg", "Frimobil"))
    stations.append(("Düdingen, Haslerastrasse", [11], "Fribourg", "Frimobil"))
    stations.append(("Wünnewil, Dorf", [13], "Fribourg", "Frimobil"))
    stations.append(("Wünnewil, Bahnhof", [13, 14], "Fribourg", "Frimobil"))
    stations.append(("Wünnewil, Felsenegg", [13, 14], "Fribourg", "Frimobil"))
    stations.append(("Flamatt, Bahnhof", [14], "Fribourg", "Frimobil"))
    stations.append(("Muntelier-Löwenberg, Station", [50, 54, 56], "Fribourg", "Frimobil"))
    stations.append(("Muntelier, Expodrom", [50, 54, 56], "Fribourg", "Frimobil"))
    stations.append(("Murten, Löwenberg Einkaufsztr.", [50, 54, 56], "Fribourg", "Frimobil"))
    stations.append(("Murten, Löwenberg Stöckli", [50, 54, 56], "Fribourg", "Frimobil"))
    stations.append(("Laupen BE, Schützenstrasse", [13], "Fribourg", "Frimobil"))
    stations.append(("Laupen BE, Betagtenzentrum", [13], "Fribourg", "Frimobil"))
    stations.append(("Laupen BE, Poly-Areal", [13], "Fribourg", "Frimobil"))
    stations.append(("Laupen BE, Bahnhof", [13], "Fribourg", "Frimobil"))
    stations.append(("Laupen BE", [13], "Fribourg", "Frimobil"))
    stations.append(("Neuenegg, Louelemoos", [14], "Fribourg", "Frimobil"))
    stations.append(("Neuenegg", [14], "Fribourg", "Frimobil"))
    stations.append(("Neuenegg, Bahnhof", [14], "Fribourg", "Frimobil"))
    stations.append(("Flamatt, Oberflammat", [14], "Fribourg", "Frimobil"))
    stations.append(("Sugiez, gare", [54], "Fribourg", "Frimobil"))
    stations.append(("Cressier FR, gare", [52], "Fribourg", "Frimobil"))
    stations.append(("Heitenried, St. Michael", [15], "Fribourg", "Frimobil"))
    stations.append(("La Roche FR, Montsoflo", [26], "Fribourg", "Frimobil"))
    stations.append(("Ecublens-Rue, gare", [46], "Fribourg", "Frimobil"))
  
    return stations