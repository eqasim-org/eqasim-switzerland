import re
import pandas as pd

def create_stations(excel_path):
    stations = []

    df         = pd.read_excel(excel_path)[["UIC", "Nom OFT", "Zone no"]]
    df.columns = ["gtfs_id", "stop_name", "zones"]

    df["zones"] = df["zones"].astype(str)
    df["zones"] = df["zones"].apply(lambda x: re.split(r"[,\s]+", x.strip()))
    df["zones"] = df["zones"].apply(lambda lst: [int(z) for z in lst if z.strip() != ""])

    for _, row in df.iterrows():
        zones = row["zones"]
        if 106107 in zones:
            zones = [106, 107]
        if 156157 in zones:
            zones = [156, 157]
        stations.append((row["stop_name"], zones, "Vaud", "Mobilis"))

    stations.append(("Fey, village", [51, 52], "Vaud", "Mobilis"))
    stations.append(("Jouxtens-Mézery, gare", [12], "Vaud", "Mobilis"))
    stations.append(("Vufflens-la-Ville,grande salle", [15], "Vaud", "Mobilis"))
    stations.append(("Vouvry, Les Barges", [88], "Vaud", "Mobilis"))
    stations.append(("Corbeyvier, Festival Celtique", [144], "Vaud", "Mobilis"))
    stations.append(("Caux, gare", [75], "Vaud", "Mobilis"))
    stations.append(("Les Echets, gare", [76], "Vaud", "Mobilis"))
    stations.append(("Haut-de-Caux, gare", [76], "Vaud", "Mobilis"))
    stations.append(("Allières, gare", [182, 183], "Vaud", "Mobilis"))
    stations.append(("Les Sciernes, gare", [183], "Vaud", "Mobilis"))
    stations.append(("Montbovon, gare", [183], "Vaud", "Mobilis"))
    stations.append(("La Tine, village", [184, 185], "Vaud", "Mobilis"))
    stations.append(("Rossinière, Hôtel de Ville", [185], "Vaud", "Mobilis"))
    stations.append(("La Chaudanne-Les M., gare", [185, 186], "Vaud", "Mobilis"))
    stations.append(("La Lécherette,cont.de la Borne", [172], "Vaud", "Mobilis"))
    stations.append(("Les Diablerets, Les Bovets", [147], "Vaud", "Mobilis"))
    stations.append(("Les Diablerets, La Faverge", [147], "Vaud", "Mobilis"))
    stations.append(("Les Diablerets, Mon Abri", [170], "Vaud", "Mobilis"))
    stations.append(("Noville, La Mounie", [77], "Vaud", "Mobilis"))
    stations.append(("Glion, chemin du Tremblex", [75], "Vaud", "Mobilis"))
    stations.append(("Glion, chemin de Maulever", [74], "Vaud", "Mobilis"))
    stations.append(("Chamby-Musée", [74], "Vaud", "Mobilis"))
    stations.append(("Carrouge VD, Gustave Roud", [61, 65], "Vaud", "Mobilis"))
    stations.append(("Corcelles-Nord, gare", [100], "Vaud", "Mobilis"))
    stations.append(("Domdidier, gare", [131], "Vaud", "Mobilis"))
    stations.append(("Ste-Croix, Ma Retraite", [120], "Vaud", "Mobilis"))
    stations.append(("Ste-Croix, Crêt-Junod", [120], "Vaud", "Mobilis"))
    stations.append(("Baulmes, poste", [47], "Vaud", "Mobilis"))
    stations.append(("Essert-Pittet, gare", [42], "Vaud", "Mobilis"))
    stations.append(("Villars-Tiercelin, M. Villars", [57,58], "Vaud", "Mobilis"))
    stations.append(("Poliez-le-Grand, Bois-la-Croix", [50, 51, 57], "Vaud", "Mobilis"))
    stations.append(("Bioley-Magnoux, Moulin agr.", [55], "Vaud", "Mobilis"))
    stations.append(("Bavois, gare", [43], "Vaud", "Mobilis"))
    stations.append(("Assens, gare", [50], "Vaud", "Mobilis"))
    stations.append(("Les Ripes, gare", [16], "Vaud", "Mobilis"))
    stations.append(("Cossonay-Ville, Tannaz", [39], "Vaud", "Mobilis"))
    stations.append(("Romanel-sur-Morges,Z.I. Moulin", [15], "Vaud", "Mobilis"))
    stations.append(("Monnaz, Trésy", [31], "Vaud", "Mobilis"))
    stations.append(("Pampigny, Laiterie", [37], "Vaud", "Mobilis"))
    stations.append(("St-Saphorin (Lavaux), gare", [64], "Vaud", "Mobilis"))
    stations.append(("Château d'Hauteville, gare", [72], "Vaud", "Mobilis"))
    stations.append(("La Baume, chemin des Roches", [78], "Vaud", "Mobilis"))
    stations.append(("Chardonne, funi", [72], "Vaud", "Mobilis"))
    stations.append(("Mont-Pèlerin, funi", [78], "Vaud", "Mobilis"))
    stations.append(("Villette (Lavaux), gare", [19], "Vaud", "Mobilis"))
    stations.append(("Gland, Lignière", [23], "Vaud", "Mobilis"))
    stations.append(("Coppet, gare", [22], "Vaud", "Mobilis"))
    stations.append(("Essertes, collège", [62, 65], "Vaud", "Mobilis"))
    stations.append(("Ecublens-Rue, gare", [61], "Vaud", "Mobilis"))

    return stations
