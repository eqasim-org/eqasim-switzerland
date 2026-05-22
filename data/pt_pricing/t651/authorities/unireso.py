import subprocess
from pathlib import Path
import re
import geopandas as gpd
import pandas as pd
import shutil

PAGES = [40, 61]

def clean(name):
    name_map = {
        "Eaux-Vives (lac)": "Genève-Eaux-Vives (lac)",
        "Genève-Eaux-Vives/Vadier": "Genève-Eaux-Vives, gare/Vadier",
        "Molard (lac)": "Genève-Molard (lac)",
        "Pâquis (lac)": "Genève-Pâquis (lac)",
        "Arena-Halle 7": "Grand-Saconnex, Arena-Halle 7",
        "Athénaz-Passeiry": "Athenaz, Passeiry",
        "Athénaz-Village": "Athenaz, village",
        "Avusy-Les-Quoattes": "Avusy, Les Quoattes",
        "Avusy Moulin-de-la-Grave": "Avusy, Moulin de la Grave",
        "Bardonnex": "Bardonnex, village",
        "Bel-Air": "Genève, Bel-Air",
        "BIT": "Genève, OMS-BIT",
        "Blanché": "Grand-Saconnex, CS Blanché",
        "Centenaire": "Plan-les-Ouates, Champ-Filles",
        "CS Sous-Moulin": "Thônex, Sous-Moulin",
        "CERN": "Meyrin, CERN",
        "Certoux": "Certoux, village",
        "Champ-Claude": "Vernier, Champ-Claude",
        "Champvigny": "Satigny, Champvigny",
        "Châtelaine": "Vernier, Châtelaine",
        "Châtelaine-Pt-Bois": "Vernier, Châtelaine",
        "Chêne-Bougeries": "Chêne-Bougeries, village",
        "Claparède": "Chêne-Bougeries,Coll.Claparède",
        "CO Bois-Caran": "Collonge-Bellerive, Bois-Caran",
        "Conches": "Chêne-Bougeries, Conches école",
        "Crêts-de-Morillon": "Grand-Saconnex, Crêts-Morillon",
        "Dardagny-les Tilleuls": "Dardagny, Les Tilleuls",
        "Eaumorte-Croisée": "Cartigny, Eaumorte croisée",
        "Ecole-Médecine": "Genève, Ecole-de-Médecine",
        "Essertines": "Dardagny, Essertines",
        "Florissant": "Chêne-Bougeries, C.-Florissant",
        "Fret": "Grand-Saconnex, Fret",
        "Genève-Champel-Gare / Hôpital": "Genève-Champel, gare/Hôpital",
        "Genève-Champel-Gare / Peschier": "Genève-Champel, gare/Peschier",
        "Genève-Eaux-Vives-Gare / Bloch": "Genève-Eaux-Vives, gare/Bloch",
        "Genève-Eaux-Vives/Vadier": "Genève-Eaux-Vives, gare/Vadier",
        "Genthod-Le Haut": "Genthod, Les Hauts",
        "Gradelle": "Cologny, Gradelle",
        "Grand-Donzel": "Vessy, Grand-Donzel",
        "Grand-Lancy-Place du 1er août": "Grand-Lancy, Place du 1er-Août",
        "Grand-Pré": "Genève, Grand-Pré",
        "Goulard": "Goulart",
        "Grand-Saconnex-Douane": "Grand Saconnex, Crêts-Morillon",
        "P+R P47": "Grand-Saconnex, Aéroport-P47",
        "Hameau de Chèvres": "Bernex, Chèvres",
        "Horloge Fleurie": "Genève-Jardin-Anglais (lac)",
        "Lancy-Pont-Rouge- Gare / Etoile": "Lancy-Pont-Rouge, gare/Etoile",
        "Léonard-Sismondi": "Chêne-Bougeries, L.-Sismondi",
        "Les Bruyères": "Calas",
        "Les Esserts": "Petit-Lancy, Les Esserts",
        "Les Tilleuls": "Dardagny, Les Tilleuls",
        "Malval-Centre nature": "Dardagny, Malval Centre Nature",
        "Moillesulaz": "Thônex, Moillesulaz",
        "Monniaz-Hameau": "Jussy, Monniaz",
        "Morglas": "Vernier, Delay",
        "Moulins-de-Drize": "Plan-les-Ouates, Moulins-Drize",
        "Onex-Salle Communale": "Onex, Salle communale",
        "P+R Bernex": "Bernex, P+R",
        "Petit-Palais": "Genève, Petit-Palais",
        "Peupliers": "Vessy, Marsillon",
        "Place du Vengeron": "Chambésy, Plage du Vengeron",
        "Pont-Sierne": "Veyrier, Pont de Sierne",
        "Prairie": "Genève, Prairie",
        "Route de Lullier": "Jussy, Lullier",
        "Route de Presinge": "Presinge, Cara-Douane",
        "Route de Saint-Maurice": "Collonge-Bellerive,Pré-d'Orsat",
        "Servette": "Genève, Servette",
        "Sous-Moulin": "Thônex, Sous-Moulin",
        "Tourbillon": "Plan-les-Ouates, ZIPLO",
        "UIT": "Genève, Collège Sismondi",
        "Vieux-Bureau": "Meyrin, gare",
        "Voirets": "Grand-Lancy, Curé-Baud",
        "Vuillonnex": "Bernex, Vuillonnex"
    }
    return name_map.get(name, name)


def create_stations(pdf_path, temp_path):
    stations  = []
    lake_mode = False

    for page in range(PAGES[0], PAGES[1] + 1):

        page_pdf = Path(temp_path) / f"{page}.pdf"
        page_txt = Path(temp_path) / f"{page}.txt"

        subprocess.run([
            "pdf-stapler", "sel", str(pdf_path), str(page), str(page_pdf)
        ], check=True)

        subprocess.run([
            "pdftotext", "-layout", str(page_pdf), str(page_txt)
        ], check=True)

        with open(page_txt, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                match = re.match(r'^([\w+ ,/.-]+)\s+([0-9]{2})$', line)
                if match:
                    station_name = match.group(1).strip()
                    zone         = [int(match.group(2))]

                    if lake_mode and station_name != "De-Chateaubriand" and station_name != "Genève-Plage":
                        station_name = station_name + " (lac)"

                    station_name = station_name.replace(" - ", "-")
                    station_name = clean(station_name)

                    stations.append((station_name, zone, "Genève", "Unireso"))
                else:
                    if "Embarcadères" in line:
                        lake_mode = True
                    if "Arrêts tpg" in line:
                        lake_mode = False

    shutil.rmtree(Path(temp_path))

    return stations


def match_stops(gtfs_stops, stops, tpg_stops):
    stops = pd.merge(stops, tpg_stops, how = "left", left_on = "stop", right_on="stop")
    stops = pd.merge(stops, gtfs_stops[["stop_id", "stop_name", "geometry"]], 
                        how = "left", on="stop_id")
    
    matched     = stops[stops["geometry"].notna()]
    not_matched = stops[stops["geometry"].isna()]

    if len(not_matched) > 0:
        not_matched = not_matched[["stop", "zones", "local network", "tarif network"]]
        not_matched = pd.merge(not_matched, gtfs_stops[["stop_id", "stop_name", "geometry"]], 
                            left_on = "stop", right_on = "stop_name",
                        how = "left")

    matched     = pd.concat([matched, not_matched[not_matched["geometry"].notna()]])

    return matched


def read_geneva_canton(canton_df):
    return canton_df[canton_df["canton_id"]==25].geometry.values[0]


def import_zones(spatial_zones, canton_df):
    ge_canton = read_geneva_canton(canton_df)
    if ge_canton.geom_type == "MultiPolygon":
        ge_canton = max(ge_canton.geoms, key=lambda p: p.area)
    else:
        ge_canton = ge_canton 

    spatial_zones["Unireso"] = {"Unireso:10": ge_canton}
    return spatial_zones