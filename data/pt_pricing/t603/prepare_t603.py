import shutil
import subprocess
from pathlib import Path
import re
import pandas as pd
import data.pt_pricing.t603.utils as t603utils


FIRST_IS_NUMBER_PATTERN = re.compile(r"[0-9]+")
STATION_NAME_PATTERN    = re.compile(r"[\S ]+? [0-9X]{4,5}")
DIRECTION_PATTERN       = re.compile(r"\(Fahrtrichtung (.*?)\)")

PAGE_RANGE_2018         = [20, 65]


def process_2018_page(path2018, page, temp_path):
    print(f"Converting page {page}...")
    temp_path_path = Path(temp_path)

    page_pdf = temp_path_path / f"{page}.pdf"
    page_txt = temp_path_path / f"{page}.txt"

    # Extract single page as a new PDF
    subprocess.run([
        "pdf-stapler", "sel", str(path2018), str(page), str(page_pdf)
    ], check=True)

    # Convert PDF to text with layout
    subprocess.run([
        "pdftotext", "-layout", str(page_pdf), str(page_txt)
    ], check=True)

    with open(page_txt, "r") as f:
        lines = [re.sub(r"\s+", " ", line).strip() for line in f]

        # Find triangle id
        triangle_id = lines[0].split(" ")[-1]

        # Find first station in the lines of the page
        first_station_index = 0
        for index, line in enumerate(lines):
            if re.match(STATION_NAME_PATTERN, line):
                first_station_index = index
                break

        # This will be filled with the station information
        reading_stations = False
        stations = []

        # Read through the file
        for line in lines[first_station_index:]:
            # Some special lines
            if len(line) == 0:
                #print("Ending page %d with empty line" % page_number)
                break

            if "Streckenabonnemente" in line:
                print("Skipping 'Streckenabonnemente' on page %d" % page)
                continue

            # Some fixing
            line = t603utils.fix_line(line, page)

            # Read through the stations
            station_match = re.search(STATION_NAME_PATTERN, line)
            if station_match:
                reading_stations = True # From now on we always expect a station in the next line

                # Find the direction if the station has one
                current_direction = None
                direction_match = re.search(DIRECTION_PATTERN, line)

                if direction_match:
                    current_direction = direction_match.group(1).split(" ")[-1]

                # Cut the line
                line = line[:station_match.end(0)].split(" ")

                # Apply some fixes
                line = t603utils.fix_station_line(line, page)

                # Read station information
                stations.append(t603utils.read_station(line, current_direction))

            elif reading_stations:
                raise RuntimeError("Page %d contains line without a station" % page)
            
        # Some distances are direction-dependent
        forward_direction = None
        backward_direction = None

        if page == 24:
            forward_direction = "Visp"
            backward_direction = "Leuk"

        # Construct two lists of indices to traverse the route
        forward_indices = [
            index
            for index, station in enumerate(stations)
            if station["direction"] is None or station["direction"] == forward_direction
        ]

        backward_indices = [
            index
            for index, station in enumerate(stations)
            if station["direction"] is None or station["direction"] == backward_direction
        ]

        # Here we have read all stations
        matrix = []

        # First, go forward
        for destination_index in forward_indices:
            destination_station = stations[destination_index]

            for j in range(len(destination_station["distances"])):
                origin_index = forward_indices[j]
                origin_station = stations[origin_index]
                distance = destination_station["distances"][j]
                matrix.append((origin_station["id"], destination_station["id"], distance))

        # Second, go backward
        for origin_index in backward_indices:
            origin_station = stations[origin_index]

            for j in range(len(origin_station["distances"])):
                destination_index = backward_indices[j]
                destination_station = stations[destination_index]
                distance = origin_station["distances"][j]
                matrix.append((origin_station["id"], destination_station["id"], distance))

        for station in stations:
            matrix.append((station["id"], station["id"], 0.0))

        return { "matrix" : matrix, "id" : triangle_id, "stations" : stations }


def process_2025_pdf(pdf_2025, temp_path):
    t603_path        = Path(pdf_2025)
    target_directory = Path(temp_path)
    target_directory.mkdir(parents=True, exist_ok=True)

    page_range = [16, 25]
    distances  = []

    def is_valid_name(s):
        return re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ\s\-]+$", s)
    
    for page in range(page_range[0], page_range[1]): 
        print(f"Converting page {page}...")

        page_pdf = target_directory / f"{page}.pdf"
        page_txt = target_directory / f"{page}.txt"

        # Extract single page as a new PDF
        subprocess.run([
            "pdf-stapler", "sel", str(t603_path), str(page), str(page_pdf)
        ], check=True)

        # Convert PDF to text with layout
        subprocess.run([
            "pdftotext", "-layout", str(page_pdf), str(page_txt)
        ], check=True)

        with open(page_txt, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                match = re.match(r"^(\S+(?:\s\S+)*)\s+(\S+(?:[\s-]\S+)*?)\s+(?:via\s+\S+\s+)?(\d+)(?:\s+.*)?$", line)

                if match:
                    result      = match.groups()
                    origin      = result[0].strip()
                    destination = result[1].strip()
                    distance    = int(result[2])

                    if is_valid_name(origin) and is_valid_name(destination):
                        distances.append((origin, destination, distance))

    shutil.rmtree(target_directory)

    return distances


def merge_into_gtfs(gtfs_stops, distances2018, distances2025):
    gtfs_stops["stop_id"] = gtfs_stops["stop_id"].astype(int)

    distances2018 = distances2018.merge(gtfs_stops.copy().rename(columns={"stop_name" : "origin_name"}), how = "left", left_on = "origin_id", right_on = "stop_id")[
        ["triangle_id", "origin_id", "destination_id", "origin_name", "distance"]
    ]
    
    distances2018 = distances2018.merge(gtfs_stops.copy().rename(columns={"stop_name" : "destination_name"}), how = "left", left_on = "destination_id", right_on = "stop_id")[
        ["triangle_id", "origin_id", "destination_id", "origin_name", "destination_name", "distance"]
    ]

    distances2025 = pd.DataFrame.from_records(
        distances2025,
        columns = ["origin_name", "destination_name", "distance"]
    )

    distances2025 = distances2025.merge(gtfs_stops.copy().rename(columns={"stop_id" : "origin_id"}), how = "left", left_on = "origin_name", right_on = "stop_name")[
        ["origin_id", "destination_name", "origin_name", "distance"]
    ]
    
    distances2025 = distances2025.merge(gtfs_stops.copy().rename(columns={"stop_id" : "destination_id"}), how = "left", left_on = "destination_name", right_on = "stop_name")[
        ["origin_id", "destination_id", "origin_name", "destination_name", "distance"]
    ]

    distances2025["triangle_id"] = 0

    distances_all = pd.concat([distances2018, distances2025])

    return distances_all


def configure(context):
    context.config("data_path")
    context.config("gtfs_name")


def execute(context):
    data_path       = context.config("data_path")
    t603_path       = f"{data_path}/pt_pricing/t603"
    gtfs_name       = context.config("gtfs_name")
    gtfs_stops_path = f"{data_path}/gtfs/{gtfs_name}/stops.txt"

    t603_2018_pdf_path = f"{t603_path}/T603_2018.pdf"
    t603_2025_pdf_path = f"{t603_path}/T603_2025.pdf"

    temp_path = f"{context.path()}/temp/t603"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    triangles = []
    for page in list(range(PAGE_RANGE_2018[0], PAGE_RANGE_2018[1])):
        triangle = process_2018_page(t603_2018_pdf_path, page, temp_path)
        triangles.append(triangle)

    df_triangles = []

    for triangle in triangles:
        triangle_id = triangle["id"]

        for origin_id, destination_id, distance in triangle["matrix"]:
            if len(origin_id) <= 5:
                origin_id = int("85%05d" % int(origin_id))
            else:
                origin_id = int(origin_id)
            if len(destination_id) <= 5:
                destination_id = int("85%05d" % int(destination_id))
            else:
                destination_id = int(destination_id)
            df_triangles.append([triangle_id, origin_id, destination_id, distance])

    df_triangles = pd.DataFrame.from_records(
        df_triangles,
        columns = ["triangle_id", "origin_id", "destination_id", "distance"]
    )

    df_triangles = df_triangles[df_triangles["origin_id"] != df_triangles["destination_id"]]

    shutil.rmtree(temp_path)

    gtfs_stops     = t603utils.process_gtfs_stops(gtfs_stops_path)
    distances_2025 = process_2025_pdf(t603_2025_pdf_path, temp_path)

    all_distances = merge_into_gtfs(gtfs_stops, df_triangles, distances_2025)

    return all_distances