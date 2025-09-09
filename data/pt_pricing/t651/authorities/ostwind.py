import subprocess
from pathlib import Path
import re
import shutil

import warnings
warnings.filterwarnings("ignore")

PAGES      = [42, 129]

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

                stations.append((stop1_name, stop1_zones, "St. Gallen", "Ostwind"))

    shutil.rmtree(Path(temp_dir))

    stations.append(("Lottstetten, Nack Ortsmitte", [837], "St. Gallen", "Ostwind"))
    stations.append(("Lottstetten, Abzw.Nacker Mühle", [837], "St. Gallen", "Ostwind"))
    stations.append(("Balm (Kr WT), Ortseingang", [837], "St. Gallen", "Ostwind"))
    stations.append(("Balm (Kr WT), Ortseingang", [837], "St. Gallen", "Ostwind"))
    stations.append(("Altenburg, Grosse Breite", [820], "St. Gallen", "Ostwind"))
    stations.append(("Jestetten, Osterfingerstrasse", [820], "St. Gallen", "Ostwind"))
    stations.append(("Detighofen-Berwangen, Ort", [838], "St. Gallen", "Ostwind"))
    stations.append(("Bühl (D), Notburgastrasse", [848], "St. Gallen", "Ostwind"))
    stations.append(("Bühl (D), Säge", [848], "St. Gallen", "Ostwind"))
    stations.append(("Bühl (D), Dorf", [848], "St. Gallen", "Ostwind"))
    stations.append(("Weisweil, Nägelehof", [848], "St. Gallen", "Ostwind"))
    stations.append(("Erzingen (Baden), Bahnhof", [848, 840], "St. Gallen", "Ostwind"))
    stations.append(("Büsingen, Stemmer/Goethestr.", [822], "St. Gallen", "Ostwind"))
    stations.append(("Langwiesen, Hauptstrasse/Bhf", [820], "St. Gallen", "Ostwind"))
    stations.append(("Uhwiesen, Wassergasse", [821], "St. Gallen", "Ostwind"))
    stations.append(("St. Katharinental, Bahnhof", [835], "St. Gallen", "Ostwind"))
    stations.append(("Gailingen, Friedrichsheim", [834], "St. Gallen", "Ostwind"))
    stations.append(("Randegg (Hegau), Oberdorf", [834], "St. Gallen", "Ostwind"))
    stations.append(("Randegg (Hegau), Ortsmitte", [834], "St. Gallen", "Ostwind"))
    stations.append(("Murbach (D), Zum Grenzstein", [834], "St. Gallen", "Ostwind"))
    stations.append(("Gailingen, Jugendwerk", [834], "St. Gallen", "Ostwind"))
    stations.append(("Etzwilen, Bahnhof", [845], "St. Gallen", "Ostwind"))
    stations.append(("Etzwilen, Hauptstrasse", [845], "St. Gallen", "Ostwind"))
    stations.append(("Mammern, Bahnhofstrasse", [845, 953], "St. Gallen", "Ostwind"))
    stations.append(("Dettighofen, Linde", [953], "St. Gallen", "Ostwind"))
    stations.append(("Lamperswil TG, Illharterstr.", [958], "St. Gallen", "Ostwind"))
    stations.append(("Wigoltingen, Fabrikstrasse", [923], "St. Gallen", "Ostwind"))
    stations.append(("Oppikon, Bahnhof", [924], "St. Gallen", "Ostwind"))
    stations.append(("Märwil, Hauptstrasse/Bhf", [919], "St. Gallen", "Ostwind"))
    stations.append(("Bürglen TG", [924], "St. Gallen", "Ostwind"))
    stations.append(("Bürglen TG, Bahnhof", [924], "St. Gallen", "Ostwind"))
    stations.append(("Berg TG, Hauptstrasse", [925], "St. Gallen", "Ostwind"))
    stations.append(("Schönholzerswilen,Gemeindehaus", [925], "St. Gallen", "Ostwind"))
    stations.append(("Kradolf, Bahnhof", [925], "St. Gallen", "Ostwind"))
    stations.append(("Dozwil, Pärkli", [226], "St. Gallen", "Ostwind"))
    stations.append(("Steinebrunn, Bahnhof", [230], "St. Gallen", "Ostwind"))
    stations.append(("Stachen, Museum Momö", [228, 230], "St. Gallen", "Ostwind"))
    stations.append(("Goldach, Rietli", [231], "St. Gallen", "Ostwind"))
    stations.append(("St. Margrethen SG, Nebengraben", [234], "St. Gallen", "Ostwind"))
    stations.append(("Walzenhausen, Almendsberg", [240], "St. Gallen", "Ostwind"))
    stations.append(("Diepoldsau, Schweizer Zoll", [235], "St. Gallen", "Ostwind"))
    stations.append(("Kreuzstrasse, Bahnhof", [236], "St. Gallen", "Ostwind"))
    stations.append(("Stoss AR, Bahnhof", [236, 244], "St. Gallen", "Ostwind"))
    stations.append(("Oberriet SG", [236], "St. Gallen", "Ostwind"))
    stations.append(("Zweibrücken, Bahnhof", [245], "St. Gallen", "Ostwind"))
    stations.append(("Gais, Bahnhof", [245, 244], "St. Gallen", "Ostwind"))
    stations.append(("Hebrig, Bahnhof", [244], "St. Gallen", "Ostwind"))
    stations.append(("Schachen (Gais), Bahnhof", [244], "St. Gallen", "Ostwind"))
    stations.append(("Rietli, Bahnhof", [244], "St. Gallen", "Ostwind"))
    stations.append(("Stoss AR, Bahnhof", [244, 236], "St. Gallen", "Ostwind"))
    stations.append(("Appenzell, Meistersrüte", [245, 247], "St. Gallen", "Ostwind"))
    stations.append(("Weissbad, Bahnhof", [248], "St. Gallen", "Ostwind"))
    stations.append(("Schwende, Bahnhof", [248], "St. Gallen", "Ostwind"))
    stations.append(("Wasserauen, Bahnhof", [248], "St. Gallen", "Ostwind"))
    stations.append(("Büchel bei Rüthi (Rheintal)", [237], "St. Gallen", "Ostwind"))
    stations.append(("Sennwald, Hof", [238], "St. Gallen", "Ostwind"))
    stations.append(("Gamprin, Badäl", [307], "St. Gallen", "Ostwind"))
    stations.append(("Nofels, GH Bad Nofels", [307], "St. Gallen", "Ostwind"))
    stations.append(("Nofels, Bergäcker", [307], "St. Gallen", "Ostwind"))
    stations.append(("Nofels, Oberer Hasenbach", [307], "St. Gallen", "Ostwind"))
    stations.append(("Nofels, Kirche", [307], "St. Gallen", "Ostwind"))
    stations.append(("Schaan, Hilti", [301], "St. Gallen", "Ostwind"))
    stations.append(("Forst (FL), Hilti", [301], "St. Gallen", "Ostwind"))
    stations.append(("Sevelen, Schild", [381], "St. Gallen", "Ostwind"))
    stations.append(("Vaduz, Schulzentrum", [301], "St. Gallen", "Ostwind"))
    stations.append(("Vaduz, Freibad", [301], "St. Gallen", "Ostwind"))
    stations.append(("Vaduz, Technopark", [301], "St. Gallen", "Ostwind"))
    stations.append(("Vaduz, Altenbach", [301], "St. Gallen", "Ostwind"))
    stations.append(("Vaduz, Wuhrstrasse", [301], "St. Gallen", "Ostwind"))
    stations.append(("Vaduz, Werkbetrieb", [301], "St. Gallen", "Ostwind"))
    stations.append(("Malbun, Jugendhaus", [305], "St. Gallen", "Ostwind"))
    stations.append(("Gaflei, Klinik", [305], "St. Gallen", "Ostwind"))
    stations.append(("Masescha, Kapelle", [305], "St. Gallen", "Ostwind"))
    stations.append(("Triesen, Alte Post", [303], "St. Gallen", "Ostwind"))
    stations.append(("Balzers, Palduinstrasse", [303], "St. Gallen", "Ostwind"))
    stations.append(("Balzers, St. Katrinabrunna", [303], "St. Gallen", "Ostwind"))
    stations.append(("Balzers, Unterm Stein", [303], "St. Gallen", "Ostwind"))
    stations.append(("Balzers, Mariahilf", [303], "St. Gallen", "Ostwind"))
    stations.append(("Trübbach, Dorf", [381], "St. Gallen", "Ostwind"))
    stations.append(("Azmoos, Feld", [381], "St. Gallen", "Ostwind"))
    stations.append(("Schwendi i. W., Mühleboden", [384], "St. Gallen", "Ostwind"))
    stations.append(("Schwendi i. W., Fischzucht", [384], "St. Gallen", "Ostwind"))
    stations.append(("Flums, Schulhaus Hochwiese", [386], "St. Gallen", "Ostwind"))
    stations.append(("Flumserberg, Ruslen", [387], "St. Gallen", "Ostwind"))
    stations.append(("Flumserberg, Bergheim", [387], "St. Gallen", "Ostwind"))
    stations.append(("Weesen, Restaurant Bahnhof", [991], "St. Gallen", "Ostwind"))
    stations.append(("Ziegelbrücke, Post", [991, 901], "St. Gallen", "Ostwind"))
    stations.append(("Niederurnen, Badstrasse", [901], "St. Gallen", "Ostwind"))
    stations.append(("Schwanden GL, Däniberg P+R", [903, 904, 911], "St. Gallen", "Ostwind"))
    stations.append(("Schwanden GL", [903, 904, 911], "St. Gallen", "Ostwind"))
    stations.append(("Schwanden GL, Bahnhof", [903, 904, 911], "St. Gallen", "Ostwind"))
    stations.append(("Nidfurn, Abzw. Bahnhof", [904], "St. Gallen", "Ostwind"))
    stations.append(("Leuggelbach, Abzw. Haslen", [904], "St. Gallen", "Ostwind"))
    stations.append(("Luchsingen, Dorfplatz", [904], "St. Gallen", "Ostwind"))
    stations.append(("Luchsingen-Hätzingen, Bahnhof", [904], "St. Gallen", "Ostwind"))
    stations.append(("Hätzingen, Feuerwehr", [904], "St. Gallen", "Ostwind"))
    stations.append(("Rüti GL, Bahnhof", [904], "St. Gallen", "Ostwind"))
    stations.append(("Linthal Braunwaldbahn (Talst.)", [904], "St. Gallen", "Ostwind"))
    stations.append(("Bazenheid, altes Zeughaus", [915], "St. Gallen", "Ostwind"))
    stations.append(("Bettwiesen, Bahnhof", [915], "St. Gallen", "Ostwind"))
    stations.append(("Tägerschen, Bahnhof", [919], "St. Gallen", "Ostwind"))
    stations.append(("Tägerschen, Wilerstrasse", [919], "St. Gallen", "Ostwind"))
    stations.append(("Wängi, Rosental (Bus)", [917], "St. Gallen", "Ostwind"))
    stations.append(("Wängi, Froheggstrasse", [917], "St. Gallen", "Ostwind"))
    stations.append(("St. Gallen, Industriestrasse", [210, 211], "St. Gallen", "Ostwind"))
    stations.append(("St. Gallen Winkeln, Bhf. Süd", [210, 211], "St. Gallen", "Ostwind"))
    stations.append(("St. Gallen, Winkeln", [210, 211], "St. Gallen", "Ostwind"))
    stations.append(("Flawil, Reithalle", [214], "St. Gallen", "Ostwind"))
    stations.append(("Schachen (Herisau), Bahnhof", [213], "St. Gallen", "Ostwind"))
    stations.append(("Waldstatt, Bahnhof", [213], "St. Gallen", "Ostwind"))
    stations.append(("Mogelsberg, Bahnhof", [273], "St. Gallen", "Ostwind"))
    stations.append(("Mogelsberg, Dorf", [273, 975], "St. Gallen", "Ostwind"))
    stations.append(("Gontenbad, Bahnhof", [249], "St. Gallen", "Ostwind"))
    stations.append(("Gonten, Bahnhof", [249], "St. Gallen", "Ostwind"))
    stations.append(("Jakobsbad, Bahnhof", [249], "St. Gallen", "Ostwind"))
    stations.append(("Neu St. Johann, Klosterkirche", [364], "St. Gallen", "Ostwind"))
    stations.append(("Ebnat-Kappel, Mitteldorf", [366, 974], "St. Gallen", "Ostwind"))
    stations.append(("Alt St. Johann, Steinbruch", [362], "St. Gallen", "Ostwind"))
    stations.append(("Rüdlingen, Steinenkreuz", [847], "St. Gallen", "Ostwind"))

    return stations