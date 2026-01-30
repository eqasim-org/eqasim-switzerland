import pandas as pd
import numpy as np
import geopandas as gpd

"""
Stage description
This stage reads and processes the SBB public transport subscription data at the postal code (PLZ) level.

Data sources
- Verbund (regional) subscriptions
    - Source: https://data.sbb.ch/explore/dataset/verbunde/table/?sort=verbund_communaute_comunita
    - Coverage: 2017-2024
    - Content: Subscription counts by tariff community (Verbund)
    - Limitation: he dataset covers most major PT operators except
      TNW (Tarifverbund Nordwestschweiz, the trinational region around Basel)
- GA and Halbtax subscriptions
    - Source: https://data.sbb.ch/explore/dataset/generalabo-halbtax-mit-bevolkerungsdaten/information/
    - Coverage: 2012-2024
    - Content: GA and Halbtax subscription counts by PLZ, resident population by PLZ (2012-2022)

TNW subscriptions in 2022 
TNW subscription data are not included in the SBB Verbund dataset.
However, TNW publishes annual reports that contain aggregated subscription statistics.

Example source (2022): https://www.tnw.ch/assets/files/content/TNW_Jahresrueckblick_2022_Web.pdf
Similar reports exist for other years and appear to contain comparable information.

The TNW report provides counts of monthly and yearly subscriptions sold in 2022, along with their cantonal 
distribution (Basel-Stadt, Basel-Landschaft, Aargau, Solothurn):
- Monthly subscriptions
    - 680,814 monthly subscriptions sold in 2022
    - Equivalent to 56,735 valid subscriptions on an average day
    - Cantonal distribution (average daily valid subscriptions):
        - Basel-Stadt: 44.1 % → 25,020
        - Basel-Landschaft: 43.6 % → 24,736
        - Aargau: 8.5 % → 4,822
        - Solothurn: 3.8 % → 2,156
- Yearly subscriptions
    - 79,869 yearly subscriptions sold in 2022
    - Cantonal distribution:
        - Basel-Stadt: 37.0 % → 29,552
        - Basel-Landschaft: 47.8 % → 38,177
        - Aargau: 9.5 % → 7,588
        - Solothurn: 5.7 % → 4,553

These values can be added to the returned dataframe at the cantonal level to compensate for the missing TNW data.

Ratio of monthly to yearly subscriptions
Using the TNW data, we estimate that one yearly subscription corresponds to approximately 0.71 simultaneously 
valid monthly subscriptions.

This ratio varies substantially by canton:
- Basel-Stadt: 0.85
- Basel-Landschaft: 0.65
- Aargau: 0.64
- Solothurn: 0.47

Interpretation: These differences likely reflect heterogeneous mobility patterns: Urban areas (e.g. Basel-Stadt) 
show a stronger preference for flexible monthly subscriptions. More rural cantons may require subscriptions 
covering multiple zones, making yearly subscriptions financially more attractive.

Subscription prices
The list below compares monthly and yearly subscription prices across several Swiss PT communities.
- TNW (all zones?)  86 monthly  824 yearly -> 1 year ~ 9.6 months
- Lausanne (1 zone) 71 monthly  639 yearly -> 1 year ~ 9.0 months
- Geneva (Geneva)   70 monthly  500 yearly -> 1 year ~ 7.1 months
- Libero (BE-SO)    82 monthly  738 yearly -> 1 year ~ 9.0 months
- Ostwind (1 zone)  72 monthly  648 yearly -> 1 year ~ 9.0 months
- engadinmobil      69 monthly  621 yearly -> 1 year ~ 9.0 months
- ZVV               88          813        -> 1 year ~ 9.2 months
- AWelle            95          855        -> 1 year ~ 9.0 months
- OndeVerte         76          684        -> 1 year ~ 9.0 months
- GA                440         3995       -> 1 year ~ 9.1 months
-> Geneva’s yearly subscription is unusually inexpensive relative to its monthly price.

Comparison with synthetic population data (January 2026)
A comparison is performed against a 100 % synthetic population generated in January 2026 
(see https://polybox.ethz.ch/index.php/s/RWF8J9XoYJYM4Jt).
- Halbtax
    - Total numbers are broadly correct
    - Spatial distribution is not well captured
- GA
    - The synthetic population contains ~80 % more GA subscriptions than reported in the SBB dataset
    - The Mikrozensus (MZ) also reports GA ownership rates substantially higher than SBB data
    - Possible explanation:
        - Employees and former employees of PT operators, as well as their partners and children,
          benefit from free or heavily discounted GA (FVP-GA)
        - According to parliamentary information (https://www.parlament.ch/de/ratsbetrieb/suche-curia-vista/geschaeft?AffairId=20217829),
          around 100,000 such GAs were valid in 2021
        - Is this explaining the full discrepancy? Not sure.
- Verbund
    - The MZ questionnaire does not distinguish between monthly and yearly subscriptions
    - As a result, synthetic population counts are substantially higher than SBB counts
    - Applying the TNW-based correction «1 yearly subscription → 1.71 monthly + yearly subscriptions»
    substantially improves agreement between datasets
    - Exception: Geneva
        - Synthetic and SBB counts already match well without correction
        - The low cost of Geneva’s yearly subscription likely contributes to this behavior
"""


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.postal_codes")


def execute(context):
    data_path = context.config("data_path")

    verbunde = pd.read_csv(f"{data_path}/pt_reference_data/verbunde.csv", sep = ";")
    ga_ht    = pd.read_csv(f"{data_path}/pt_reference_data/generalabo-halbtax-mit-bevolkerungsdaten.csv", sep = ";")

    ga_ht.columns    = ["year", "postal_code", "ga", "ga_flag", "ht", "ht_flag", "population", "ga_percentage", "ht_percentage"]
    verbunde.columns = ["year", "postal_code", "network", "count", "flag"]

    ga_ht    = ga_ht[["year", "postal_code", "population", "ga", "ht"]]
    verbunde = verbunde.pivot_table(index = ["year", "postal_code"],
                        columns = "network",
                        values = "count",
                        fill_value = 0).reset_index()
    
    for col in verbunde.columns:
        verbunde[col] = verbunde[col].astype(int)

    for col in ga_ht.columns:
        if col in ["year", "postal_code", "ga", "ht", "population"]:
            ga_ht[col] = ga_ht[col].fillna(0).astype(int)

    df = ga_ht.merge(verbunde, how = "outer", on = ["year", "postal_code"])

    int_cols = [c for c in df.columns if c not in ["year", "postal_code"]]
    df[int_cols] = df[int_cols].fillna(0).astype(int)

    for col in ["BÜGA", "FlexTax Schafhausen", "OSTWIND Ausland",
                "OTV-VVV / OTV-VHB", "TV BEO", "TV BEO (nach Libero)",
                "Tarifverbund Davos", "Tarifverbund Klosters",
                "VagABOnd", "Z-Pass FlexTax", "mobilis"]:
        del df[col]
    
    df = df.rename(columns={"Léman Pass": "LemanPass",
                            "Passepartout": "Passept",
                            "Tarifverbund Schwyz": "Schwyz",
                            "Tarifverbund Zug": "Zug",
                            "Z-Pass A-Welle": "ZPassAW",
                            "Z-Pass OSTWIND": "ZPassOst",
                            "Z-Pass Schwyz/Zug": "ZPassSZ",
                            "engadinmobil": "engadin",
                            "Onde Verte": "OndeVerte",
                            "Arcobaleno": "Ticino"
                            })
    
    verbund_cols = [
            'A-Welle', 'Ticino', 'Frimobil', 'Libero', 'LemanPass', 'Mobilis',
            'OSTWIND', 'OndeVerte', 'Passept', 'Schwyz', 'Zug', 'TransReno',
            'Vagabond', 'ZPassAW', 'ZPassOst', 'ZPassSZ', 'ZVV', 'engadin', 'unireso'
        ]
        
    df["verbund"] = df[verbund_cols].sum(axis=1)

    df = df[["year", "postal_code", "ga", "ht", "verbund"]]
    df = df.rename(columns = {"ga": "N_ga_sbb", "ht": "N_ht_sbb", "verbund": "N_va_sbb"})

    postal_codes = context.stage("data.spatial.postal_codes")
    postal_codes["postal_code"] = postal_codes["postal_code"].astype(int)
    postal_codes = postal_codes.dissolve(by = "postal_code", as_index = False)

    cantons = context.stage("data.spatial.cantons")
    overlaps = gpd.overlay(
        postal_codes[["postal_code", "geometry"]],
        cantons[["canton_id", "canton_name", "geometry"]],
        how="intersection"
    )
    
    overlaps["overlap_area"] = overlaps.geometry.area
    dominant_pc_to_canton = (
        overlaps
        .sort_values(["postal_code", "overlap_area"], ascending=[True, False])
        .drop_duplicates(subset="postal_code")
        .drop(columns="overlap_area")
    )
    
    postal_with_canton = (
        postal_codes
        .merge(
            dominant_pc_to_canton[["postal_code", "canton_id", "canton_name"]],
            on="postal_code",
            how="left"
        )
    )

    liechtenstein_postal_codes = [
        9485, 9486, 9487, 9488, 9489,
        9490, 9491, 9492, 9493, 9494,
        9495, 9496, 9497, 9498
    ]
    
    for li_zip in liechtenstein_postal_codes:
        postal_with_canton.loc[postal_with_canton["postal_code"] == li_zip, "canton_id"] = 27
        postal_with_canton.loc[postal_with_canton["postal_code"] == li_zip, "canton_name"] = "Liechtenstein"
    
    postal_with_canton = postal_with_canton[postal_with_canton["canton_name"]!= "Liechtenstein"]
    postal_with_canton = postal_with_canton[["postal_code", "canton_id", "canton_name"]]

    df = postal_with_canton.merge(df, on = "postal_code")

    df["canton_id"] = df["canton_id"].astype(int)
    df = df.sort_values(by = ["postal_code", "year"], ascending = True)

    return df