import re
import os
import pdfplumber
import pandas as pd


def configure(context):
    context.config("data_path")
    context.config("tolls_aprr_area_open_system_file", "2026- Systèmes Ouverts APRR-AREA.pdf")

def execute(context):
    pdf_file = os.path.join(context.config("data_path"), "tolls", context.config("tolls_aprr_area_open_system_file"))
    assert os.path.exists(pdf_file), f"File not found: {pdf_file}" # we do this mainly to check the data manually

    df = pd.DataFrame(rows, columns=COLS)

    # rename columns
    df = df.rename(columns={"Autoroute": "autoroute",
                            "N° Gare": "station_number",
                            "Nom gare": "name",
                            "Classe 1": "price",
                            "Classe 2": "price_class_2",
                            "Classe 3": "price_class_3",
                            "Classe 4": "price_class_4",
                            "Classe 5": "price_class_5",
                            "Kilomètres": "distance"})
    return df

                



#################### helper functions ####################
"""
IMPORTANT:

This file is different than the other pdf files. It doesn't contain text, but a picture, a photo of the table. Therefore, we cannot extract the data from it using pdfplumber.
"""

# fmt: off
rows = [
    # Autoroute, N Gare, Nom gare, Classe1, Classe2, Classe3, Classe4, Classe5, Km
    ("A71",  "09078", "GERZAT-VILLE",                 1.40,  1.70,  3.30,  3.40, 1.10, 18),
    ("A6",   "09121", "VILLEFRANCHE-VILLE",           2.80,  4.20,  5.60,  7.20, 1.80, 30),
    ("A432", "09153", "LA BOISSE",                    2.40,  3.90,  6.20,  8.00, 1.50, 24),
    ("A77",  "09403", "DORDIVES",                     6.00,  9.00, 14.10, 18.90, 3.70, 62),
    ("A77",  "09404", "LE TOURNEAU",                  3.80,  5.80,  8.90, 11.90, 2.50, 40),
    ("A77",  "09405", "MYENNES",                      4.20,  6.00,  8.40, 11.90, 2.80, 33),
    ("A466", "09420", "QUINCIEUX BARRIERE",           3.60,  5.60,  8.30, 11.70, 2.30, 40),
    ("A46",  "09421", "GENAY",                        1.70,  2.50,  4.20,  5.60, 1.00, 17),
    ("A46",  "09422", "MIONNAY",                      1.00,  1.50,  2.40,  3.30, 0.50, 9),
    ("TML*", "09429", "LUSSE",                        6.80, 10.70, 19.00, 31.90, 4.10, 11),
    ("A36",  "09431", "FONTAINE-LARIVIERE",           3.10,  4.90,  7.80, 10.90, 1.90, 33),
    ("A406", "09443", "CROTTET",                      0.80,  1.20,  1.80,  2.30, 0.40, 7),
    ("A719", "09465", "VICHY",                        1.20,  1.90,  2.90,  4.00, 0.70, 13),
    ("A410", "03016", "CRUSEILLES A410",              3.20,  4.90,  7.40,  9.40, 1.40, 25),
    ("A41N", "15017", "CRUSEILLES A41N",              8.00, 14.90, 22.10, 27.20, 4.40, 19),
    ("A41N", "15019", "COPPONEX",                     2.90,  5.40,  8.70, 11.10, 1.80, 11),
    ("A43",  "03027", "CHIGNIN BRETELLE",             1.00,  1.60,  2.70,  3.70, 0.50, 7),
    ("A51",  "03041", "LE CROZET",                    4.50,  7.10, 10.00, 12.80, 2.00, 31),
    ("A432", "03070", "SAINT EXUPERY",                2.30,  3.60,  5.70,  7.00, 1.10, 19),
    ("A43",  "03071", "CHESNES",                      2.30,  3.60,  5.70,  7.00, 1.10, 19),
]
# fmt: on

COLS = ["Autoroute", "N° Gare", "Nom gare", "Classe 1", "Classe 2",
        "Classe 3", "Classe 4", "Classe 5", "Kilomètres"]
