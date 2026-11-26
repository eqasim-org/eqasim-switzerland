"""Archive and read the FSO releases used by the apprentice workplace analysis."""

import argparse
import hashlib
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd


EMPLOYER = "35887278.px"
REGION = "36890545.xlsx"
AGE = "36890553.xlsx"
SHA256 = {
    EMPLOYER: "eb744012f374de2a837f0be7d026cf252feaeafe9f92be1db14fe8dab5a3dc67",
    REGION: "b6faffb8814c5bb91ba8d6c5f3a33cd18dc4cb1b0bea25f29ad29aa0fa909fb9",
    AGE: "c4b161eb8f303705f888653ad13db6dec172501a75ef22d2a76bc41990426a5c",
}
REGION_NAMES = ["Genferseeregion", "Espace Mittelland", "Nordwestschweiz", "Zürich",
                "Ostschweiz", "Zentralschweiz", "Tessin"]


def source_folder(data_path):
    return Path(data_path) / "apprentices" / "fso"


def check_file(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"FSO source {path} has SHA-256 {actual}, expected {expected}")


def verify_sources(data_path):
    folder = source_folder(data_path)
    for name, digest in SHA256.items():
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing {folder / name}. Run python -m "
                                    f"data.structural_survey.apprentice_references --data-path \"{data_path}\"")
        check_file(folder / name, digest)
    return folder


def download(data_path):
    folder = source_folder(data_path)
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in SHA256.items():
        path = folder / name
        if path.exists():
            check_file(path, digest)
            continue
        url = f"https://dam-api.bfs.admin.ch/hub/api/dam/assets/{path.stem}/master"
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with urlopen(url, timeout=90) as response:
                temporary.write_bytes(response.read())
            check_file(temporary, digest)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return folder


def employer_stock(path):
    """Read dual training across all fields, qualifications and sexes by employer canton."""
    header, data = path.read_text(encoding="iso-8859-15").split("\nDATA=", 1)
    axes = {name: re.findall(r'"([^"]+)"', codes)
            for name, codes in re.findall(r'^CODES\("([^"]+)"\)=(.*?);', header, re.MULTILINE | re.DOTALL)}
    dimensions = ["Ausbildungsfeld", "Lehrbetriebskanton", "Ausbildungstyp", "Ausbildungsform", "Geschlecht", "Jahr"]
    values = np.fromstring(data.rstrip().removesuffix(";"), dtype=int, sep=" ")
    cube = values.reshape(tuple(len(axes[key]) for key in dimensions))
    field, canton, qualification, form, sex, year = (axes[key] for key in dimensions)
    dual = cube[field.index("0"), :, qualification.index("0"), form.index("20"), sex.index("0"), :]
    frame = pd.DataFrame(dual, index=pd.Index(map(int, canton), name="canton"), columns=year)
    cantons = frame.loc[range(1, 27)]
    if not cantons.sum().equals(frame.loc[0]):
        raise ValueError("Dual-training canton counts do not sum to the Swiss total")
    return cantons


def residence_controls(folder):
    """Read 2023 annual-average SLFS estimates for the permanent resident population."""
    region = pd.read_excel(folder / REGION, sheet_name="Jahreswerte", header=4, nrows=38)
    age = pd.read_excel(folder / AGE, sheet_name="Alter-Jahreswerte", header=4, nrows=12)
    # Use the apprentice block of the regional table and the 15–24 block of the age table.
    region_labels = region.iloc[:, 0].str.strip()
    start = region_labels[region_labels.eq("Lehrlinge")].index.item() + 1
    block = region.iloc[start:start + 8].set_index(region_labels.iloc[start:start + 8])
    stocks = block.loc[REGION_NAMES, 2023].astype(float).to_numpy() * 1000
    national = float(block.loc["Schweiz", 2023])
    age_labels = age.iloc[:, 0].str.strip()
    start = age_labels[age_labels.eq("15-24 Jahre")].index.item()
    young = age.iloc[start:].set_index(age_labels.iloc[start:]).loc["Lehrlinge", 2023]
    if not np.isclose(stocks.sum(), national * 1000) or not 0 < young < national:
        raise ValueError("Inconsistent apprentice totals in the 2023 SLFS tables")
    return stocks, float(young / national)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True, help="Pipeline data_path")
    print(download(parser.parse_args().data_path))
