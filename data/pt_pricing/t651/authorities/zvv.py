import subprocess
from pathlib import Path
import re
import shutil
import geopandas as gpd
from shapely.ops import unary_union

PAGES      = [2, 34]

def clean(name):
    if "Alte Post" in name and name != "Wildberg, Alte Post":
        return name.replace("Alte Post", "alte Post")
    if "Affoltern am Albis" in name and name != "Affoltern am Albis":
        return name.replace("Affoltern am Albis", "Affoltern a.A.")
    if "Wangen" in name and not "Brüttisellen" in name and not "Neuwisen" in name:
        return name.replace("Wangen", "Wangen b D'dorf")
    name_map = {
        "Bassersdorf, Bahnhof Süd": "Bassersdorf, Bahnhof",
        "Bauma, Giess. Wolfensberger": "Bauma, Giesserei",
        "Bauma, Seewadel": "Bauma, Bahnhof",
        "Breite b. N., Grünenwaldstr.": "Breite b. N'dorf,Grünenwaldstr",
        "Breite b. N., Sternen": "Breite b. Nürensdorf, Sternen",
        "Breite b. N., Winterthurerstr.": "Breite b. N'dorf, W'thurerstr.",
        "Brüttisellen, Obere Wangenstr.": "Brüttisellen, Ob. Wangenstr.",
        "Buch am Irchel, Gemeindehaus": "Buch am Irchel, Unterbuch",
        "Bülach, Glasi": "Bülach, Soliboden",
        "Dietlikon, Haldensteig": "Dietlikon, Rebackerweg",
        "Dietlikon, Pappelstrasse": "Dietlikon, Brandbachstrasse",
        "Dübendorf, Sportanlage Heerenschürli": "Dübendorf, Sport Heerenschürli",
        "Elgg, Heurüti": "Hofstetten ZH, Dorf",
        "Elgg, Schwimmbad": "Elgg, Schloss",
        "Elgg, Torweiher": "Elgg, Obergasse",
        "Glattfelden, Schwimmbad": "Glattfelden, Post",
        "Glattpark": "Glattpark, Glattpark",
        "Horgen, Panorama": "Horgen, Panorama/CS",
        "Horgen, Risi/DOW": "Horgen, Risi/Dow",
        "Horgen, Untere Mühle": "Horgen, untere Mühle",
        "Horgen, Zentrum Tödi": "Horgen Oberdorf",
        "Kilchberg ZH, Schulhaus Dorfstr.": "Kilchberg ZH,Schulhaus Dorfstr",
        "Kilchberg ZH, Obere Hornhalde": "Zürich, Obere Hornhalde",
        "Küsnacht ZH, Obere Heslibachstr.": "Küsnacht ZH, Ob. Heslibachstr.",
        "Langnau am Albis, Oberrenggstr.": "Langnau am Albis,Oberrenggstr.",
        "Oberwil bei Nürensdorf": "Nürensdorf, Oberwil",
        "Oetwil an der Limmat, Gässliacker": "Oetwil a.d.L., Gässliacker",
        "Oetwil an der Limmat, Schweizäcker": "Oetwil a.d.L., Schweizäcker",
        "Pfäffikon SZ, Huob": "Pfäffikon SZ, Tertianum",
        "Pfäffikon SZ, Talstrasse": "Pfäffikon SZ, Seedamm-Center",
        "Pfäffikon ZH, Im Spitz": "Pfäffikon ZH, im Spitz",
        "Regensdorf, Spittelhölzli": "Regensdorf, Allmend",
        "Regensdorf, Sportanlage Wisacher": "Regensdorf, Strassenverkehrs.",
        "Rheinau, Klosterplatz": "Rheinau, Unterstadt",
        "Riedt bei Neerach, Riedacher": "Riedt b. Neerach, Riedacher",
        "Riedt bei Neerach, Storchen": "Riedt b. Neerach, Storchen",
        "Ringlikon, Langwis": "Ringlikon, Langwies",
        "Rümlang, Oberdorf": "Rümlang, Heuelstrasse",
        "Spreitenbach, Ikea": "Spreitenbach, IKEA",
        "Spreitenbach, Raiacker": "Spreitenbach, Geeracher",
        "Steg im Tösstal": "Steg im Tösstal, Bahnhof",
        "Thalwil, Oberdorf": "Thalwil, Mühlebachplatz",
        "Thalwil, Schwandelstrasse": "Thalwil, Zentrum",
        "Turbenthal, Strandbad Bichelsee": "Turbenthal,Strandbad Bichelsee",
        "Uetikon am See": "Uetikon am See, Bahnhof",
        "Uetikon am See (See)": "Uetikon am See, Bahnhof",
        "Uster, Im Hölzli": "Uster, im Hölzli",
        "Waltikon, Station": "Zumikon, Waltikon",
        "Wangen, Neuwisen": "Brüttisellen, Neuwisen",
        "Watt": "Watt, Dorf",
        "Weisslingen, Lendikon": "Lendikon",
        "Wil ZH, Gemeindehaus": "Wil ZH, Dorf",
        "Winkel, Scheidweg": "Winkel, Seebüel",
        "Winterberg ZH, Kleinikon": "Kleinikon",
        "Winterthur, Grubenstrasse": "Winterthur, Grubenstr.",
        "Zollikerberg, Station/Quartiertreff": "Zollikerberg, Station",
        "Zweidlen, Riverside": "Zweidlen, Bahnhof",
        "Zürich, Bahnhof Oerlikon": "Zürich Oerlikon, Bahnhof",
        "Zürich, Bahnhof Oerlikon Nord": "Zürich Oerlikon, Bahnhof Nord",
        "Zürich, Bahnhof Oerlikon Ost": "Zürich Oerlikon, Bahnhof Ost",
        "Zürich, Bahnhof Hardbrücke": "Zürich Hardbrücke, Bahnhof",
        "Zürich, Bahnhof Affoltern": "Zürich Affoltern, Bahnhof",
        "Zürich, Bahnhof Altstetten": "Zürich Altstetten, Bahnhof",
        "Zürich, Bahnhof Altstetten Nord": "Zürich Altstetten, Bahnhof N",
        "Zürich, Bahnhof Enge": "Zürich Enge, Bahnhof",
        "Zürich, Bahnhof Selnau": "Zürich Selnau, Bahnhof",
        "Zürich, Bahnhof Enge/Bederstr.": "Zürich Enge, Bahnhof/Bederstr.",
        "Zürich, Bahnhof Leimbach": "Zürich Leimbach, Bahnhof",
        "Zürich, Bahnhof Stettbach": "Stettbach, Bahnhof",
        "Zürich, Bahnhof Tiefenbrunnen": "Zürich Tiefenbrunnen",
        "Zürich, Bahnhof Wiedikon": "Zürich Wiedikon, Bahnhof",
        "Zürich, Bahnhof Wipkingen": "Zürich Wipkingen, Bahnhof",
        "Zürich, Bahnhof Wollishofen": "Zürich Wollishofen, Bahnhof",
        "Zürich, Bergstation Dolderbahn": "Zürich, Dolder",
        "Zürich, Besenrainweg": "Zürich, Besenrainstrasse",
        "Zürich, Bhf. Wollishofen/Staubstr.": "Zürich Wollishofen,Bhf/Staubst",
        "Zürich, Bhf. Wollishofen/Werft": "Zürich Wollishofen, Bhf/Werft",
        "Zürich, Central Polybahn": "Zürich Central (Polybahn)",
        "Zürich, Sihlpost / HB": "Zürich, Sihlpost/HB",
        "Zürich, Polyterrasse ETH": "Zürich Polyterrasse",
        "Zürich, Kinderspital": "Zürich, Kindersp. (ab 2.11.24)",
        "Zürich, Kantonsschule Enge": "Zürich Enge, Bahnhof",
        "Zürich, Kalkbreite/Bhf.Wiedikon": "Zürich,Kalkbreite/Bhf.Wiedikon",
        "Zürich Limmatquai (See)": "Zürich Limmatquai",
        "Wald ZH, Abzweigung Oberholz": "Wald ZH, Abzw. Oberholz",
        "Adlikon bei Andelfingen": "Adlikon b. Andelfingen",
        "Adliswil, Baumgartenweg": "Adliswil, Bahnhof",
        "Niederglatt ZH,Altes Schulhaus": "Niederglatt ZH,altes Schulhaus",
        "Regensdorf, Althard/Bahnhof": "Regensdorf, Althard",
        "Spreitenbach, Asp": "Spreitenbach, Kreuzäcker",
        "Wasterkingen, Ausserdorfstrasse": "Wasterkingen,Ausserdorfstrasse",
        "Au ZH, Austrasse": "Au ZH, Aubrücke",
        "Bäch": "Bäch SZ, Bahnhof",
        "Feuerthalen, Bahnhof": "Feuerthalen",
        "Schöfflisdorf-O'weningen, Bahnhof": "Schöfflisdorf-Oberwen, Bahnhof",
        "Winterthur, Bahnhof Grüze Süd":"Winterthur Grüze, Bahnhof",
        "Winterthur, Bahnhof Wülflingen": "Winterthur Wülflingen, Bahnhof",
        "Winterthur, Bhf. Oberwinterthur": "Oberwinterthur, Bahnhof",
        "Freienbach, Bezirksschule": "Freienbach, Gehren",
        "Herrliberg-Feldmeilen, Bhf. West": "Herrliberg-Feldmeilen, Bahnhof",
        "Birchwil bei Nürensdorf": "Birchwil (Nürensdorf)",
        "Wermatswil, Chammerholzstr.":"Wermatswil, Chammerholzstrasse",
        "Weisslingen, Dettenried": "Weisslingen, Dorf",
        "Ellikon an der Thur, Dorf": "Ellikon a. d. Thur, Dorf",
        "Oberweningen, Dorf": "Oberweningen, Hüeblistrasse",
        "Egetswil": "Egetswil, Dorf",
        "Pfäffikon SZ, Eichenstrasse": "Pfäffikon SZ, Roggenacker",
        "Horgen, Fähre": "Horgen (See)",
        "Meilen, Fähre": "Meilen (See)",
    }
    return name_map.get(name, name)


def create_stations(pdf_path, temp_path):
    stations = []

    for page in range(PAGES[0], PAGES[1] + 1):
        #print(f"Converting page {page}...")

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
                line  = line.strip()
                match = re.match(r'^\s*(\S.*?)\s{2,}(\S.*?)\s{2,}(\S.*?)\s{2,}([\d\s/]+)$', line)
                if match:
                    result = match.groups()

                    station_name = result[1].strip()
                    local_netw   = result[2].strip()
                    zones        = result[3].strip()

                    station_name = clean(station_name)

                    if "/" in zones:
                        zones = [int(z) for z in zones.split("/")]
                    else:
                        zones = [int(zones)]

                    stations.append((station_name, zones, local_netw, "ZVV"))

    shutil.rmtree(Path(temp_path))

    return stations


def import_zones(spatial_zones, shp_path):
    zvv_zones = gpd.read_file(shp_path)[["ZONE", "geometry"]]
    zvv_zones = zvv_zones[zvv_zones["ZONE"] != "NULL"]
    zvv_zones = zvv_zones[zvv_zones["ZONE"].notna()]

    zvv_zones["ZONE"] = zvv_zones["ZONE"].astype(int).astype(str)
    zvv_zones = zvv_zones.groupby("ZONE").agg({"geometry": lambda g: unary_union(g)}).reset_index()

    spatial_zvv = {"ZVV:" + row["ZONE"]: row["geometry"] for _, row in zvv_zones.iterrows()}

    spatial_zones["ZVV"] = spatial_zvv
    return spatial_zones