"""
International organizations in Switzerland: staff estimates, location, and coordinates.

Purpose: fill the gap left by STATENT (Swiss business census), which does not cover
organizations with extraterritorial/diplomatic status (accord de siège, accord fiscal,
accord sur les privilèges et immunités) or similar special-status bodies (sports
federations, some NGOs).

Fields:
- staff: best available estimate of on-site employees (int), or None if not found.
- staff_confidence: "high" / "medium" / "low" / "aggregate" (covers multiple orgs) / "not_found"
- staff_year: year the figure refers to, where known (int) or None
- city: municipality of the HQ/workplace
- lat, lon: approximate WGS84 coordinates of the HQ building. None if not determined
  (e.g. aggregate categories with no single site).
- source_note: brief note on where the figure came from and caveats

NOTE ON AGGREGATES: A few entries (ngo_aggregate_geneva, permanent_missions_consulates,
sports_federations_vaud_aggregate) are NOT individual organizations but sums covering
many organizations/entities. Keep these separate from the per-organization entries when
allocating jobs to specific coordinates - they cannot be pinned to one location.

TPG SURVEY LAYER: international_organizations_switzerland (the hand-researched dict
below) is layered with a colleague-provided workplace survey, data_path/statent/
tpg_OI-ONG_nb-employees_20260921_with_coordinates.xlsx (73 Geneva-area orgs/NGOs,
address-geocoded, most with a headcount size bracket and some with an exact count).
See apply_tpg_survey for the merge policy: orgs matched by name (TPG_NAME_TO_KEY) to
an existing entry get their coordinates replaced by the survey's geocode, and their
staff figure replaced UNLESS the survey only gives a bracket and the existing entry is
already "high"/"medium" confidence (in which case the existing better-sourced figure
is kept). Unmatched survey rows become new entries.

Stage: data.statent.international_organizations

Produces, in the stage's cache folder (context.path()):
  - international_organizations_map.html   interactive map (folium, same
                                            swisstopo basemap as
                                            data.cross_border.cross_border_points_counts)
                                            with a blue dot per organization
                                            sized by staff count, or a gray
                                            dot for organizations with no
                                            staff figure - see build_map.
                                            Organizations with no
                                            coordinates (mostly the
                                            multi-organization AGGREGATE__
                                            entries) are excluded from the
                                            map (logged) but still returned.

Returns one row per entry in international_organizations_switzerland (staff,
staff_confidence, staff_year, city, lat, lon, source_note, plus an
is_aggregate flag - see NOTE ON AGGREGATES above).
"""

import logging
import os
import re

import folium
import pandas as pd

logger = logging.getLogger("synpp")

# Free, no-key WMTS basemap - see analysis/pt/interactive_map.py's module
# docstring for why plain OSM/cartodbpositron tiles are not used here.
_TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"

MIN_RADIUS = 5
MAX_EXTRA_RADIUS = 20

# Fixed marker radius/color for organizations with no staff figure at all -
# see build_map.
UNKNOWN_STAFF_RADIUS = 4
UNKNOWN_STAFF_COLOR = "#7f7f7f"

# --- TPG workplace survey (see module docstring, TPG SURVEY LAYER) ---

TPG_SURVEY_PATH = "statent/tpg_OI-ONG_nb-employees_20260921_with_coordinates.xlsx"

# The sheet's headcount size brackets, mapped to a point estimate (bracket
# midpoint; "(A) >= 1000" is open-ended so a judgment-call value is used).
TPG_BRACKET_MIDPOINT = {
    "(E) moins de 50": 25,
    "(D) 50 à 99": 75,
    "(C) 100 à 199": 150,
    "(B) 200 à 999": 600,
    "(A) >= 1000": 1200,
}

# Rows that duplicate another row in the SAME sheet (the same organization
# surveyed twice under a different name/address) - dropped, keeping the row
# with the exact employee count.
TPG_DUPLICATE_NAMES = (
    "Foundation for Innovative New Diagnostics (FIND)",  # dup of "FIND"
    "Organisation Internationale du Travail (OIT)",       # dup of "Bureau international du Travail"
)

# TPG sheet name -> key in international_organizations_switzerland, for the
# ~30 organizations that are already in that hand-researched dict (under a
# different, usually English, name). Matched by hand against name + address;
# anything not listed here becomes a new entry - see apply_tpg_survey.
TPG_NAME_TO_KEY = {
    "Bureau international du Travail": "ILO (International Labour Organization)",
    "Centre Henri Dunant pour le Dialogue Humanitaire": "HD Centre (Centre for Humanitarian Dialogue)",
    "Centre international de déminage humanitaire - Genève": "GICHD (Geneva International Centre for Humanitarian Demining)",
    "CERN": "CERN",
    "Commission Electrotechnique Internationale": "IEC (International Electrotechnical Commission)",
    "Conseil oecuménique des Eglises": "World Council of Churches (WCC)",
    "Drugs for Neglected Diseases initiative (DNDi)": "DNDi (Drugs for Neglected Diseases initiative)",
    "Fédération internationale des Sociétés de la Croix-Rouge et du Croissant-Rouge - FISCR":
        "IFRC (International Federation of Red Cross and Red Crescent Societies)",
    "FIND": "FIND (Foundation for Innovative New Diagnostics)",
    "GAVI Alliance": "Gavi, the Vaccine Alliance",
    "GCERF": "GCERF",
    "Haut Commissariat des Nations unies pour les réfugiés (UNHCR)": "UNHCR (UN Refugee Agency)",
    "International Air Transport Association (IATA)": "IATA (International Air Transport Association)",
    "International Trade Center": "ITC (International Trade Centre)",
    "Le Comité international de la Croix-Rouge (CICR)": "ICRC (International Committee of the Red Cross)",
    "Médecins Sans Frontières": "MSF (Medecins Sans Frontieres) - Switzerland section & International Office",
    "MMV Medicines for Malaria Venture": "MMV (Medicines for Malaria Venture)",
    "ONUG": "UN Office at Geneva (UNOG secretariat)",
    "Organisation internationale de normalisation (ISO)": "ISO (International Organization for Standardization)",
    "Organisation internationale des migrations": "IOM (International Organization for Migration)",
    "Organisation météréologique mondiale": "WMO (World Meteorological Organization)",
    "Organisation mondiale de la propriété intellectuelle": "WIPO (World Intellectual Property Organization)",
    "Organisation Mondiale de la Santé": "WHO (World Health Organization)",
    "Organisation mondiale du commerce": "WTO (World Trade Organization)",
    "The Global Alliance for Improved Nutrition": "GAIN (Global Alliance for Improved Nutrition)",
    "The Global Fund": "Global Fund (GFATM)",
    "UNAIDS": "UNAIDS (Joint UN Programme on HIV/AIDS)",
    "Union internationale de télécommunicationee": "ITU (International Telecommunication Union)",
    "Union Internationale des Transports Routiers (IRU)": "IRU (International Road Transport Union)",
    "World Economic Forum": "WEF (World Economic Forum)",
}


def configure(context):
    context.config("data_path")


def _tpg_city(address):
    """Best-effort municipality from the sheet's free-text address field."""
    parts = [part.strip() for part in str(address).split(",") if part.strip()]
    if not parts:
        return None
    city = parts[-2] if parts[-1].upper() == "FRANCE" and len(parts) >= 2 else parts[-1]
    return re.sub(r"^\d+\s*", "", city).title()


def load_tpg_survey(context):
    """Reads and cleans the TPG survey sheet - see module docstring."""
    path = os.path.join(context.config("data_path"), TPG_SURVEY_PATH)
    df = pd.read_excel(path)

    df = df[~df["Nom"].isin(TPG_DUPLICATE_NAMES)]

    missing_coords = df[df["coordinates"].isna()]
    if len(missing_coords) > 0:
        logger.warning(
            "%d TPG survey entries have no coordinates and are dropped: %s",
            len(missing_coords), missing_coords["Nom"].tolist(),
        )
    df = df[df["coordinates"].notna()].copy()

    coordinates = df["coordinates"].str.split(",", expand = True).astype(float)
    df["lat"] = coordinates[0]
    df["lon"] = coordinates[1]

    df["staff_exact"] = pd.to_numeric(df["Nombre d'employés"], errors = "coerce")
    df["staff_bracket"] = df["Taille de l'organisation"]
    df["staff_bracket_midpoint"] = df["staff_bracket"].map(TPG_BRACKET_MIDPOINT)
    df["city"] = df["Adresse principale"].apply(_tpg_city)

    return df


def apply_tpg_survey(context, df):
    """
    Layers the TPG survey (load_tpg_survey) on top of df (built from
    international_organizations_switzerland) - see module docstring's TPG
    SURVEY LAYER section for the merge policy.
    """
    df_tpg = load_tpg_survey(context)

    matched_names = []
    for _, row in df_tpg.iterrows():
        key = TPG_NAME_TO_KEY.get(row["Nom"])
        if key is None:
            continue
        matched_names.append(row["Nom"])

        matches = df.index[df["name"] == key]
        if len(matches) == 0:
            logger.warning("TPG survey entry '%s' maps to unknown key '%s' - skipped.", row["Nom"], key)
            continue
        i = matches[0]

        has_exact = pd.notna(row["staff_exact"])
        keep_existing_estimate = not has_exact and df.loc[i, "staff_confidence"] in ("high", "medium")

        if not keep_existing_estimate:
            if has_exact:
                df.loc[i, "staff"] = row["staff_exact"]
                df.loc[i, "staff_confidence"] = "high"
                note = f"{row['staff_exact']:.0f} employees (exact)."
            else:
                df.loc[i, "staff"] = row["staff_bracket_midpoint"]
                df.loc[i, "staff_confidence"] = "low"
                note = f"size bracket {row['staff_bracket']}, bracket midpoint used."
            df.loc[i, "staff_year"] = 2026
            df.loc[i, "source_note"] = (
                str(df.loc[i, "source_note"])
                + " | Updated per colleague's TPG workplace survey (Sep 2026): " + note
            )

        # The survey's coordinates come from geocoding the org's actual
        # address, so they're preferred over the original dict's hand
        # estimates even when the staff figure itself is kept.
        df.loc[i, "lat"] = row["lat"]
        df.loc[i, "lon"] = row["lon"]

    df_new = df_tpg[~df_tpg["Nom"].isin(matched_names)]
    has_exact = df_new["staff_exact"].notna()

    detail = pd.Series("Exact employee count.", index = df_new.index).where(
        has_exact,
        "No exact count given; bracket (" + df_new["staff_bracket"].astype(str) + ") midpoint used.",
    )

    df_new_rows = pd.DataFrame({
        "name": df_new["Nom"],
        "staff": df_new["staff_exact"].where(has_exact, df_new["staff_bracket_midpoint"]),
        "staff_confidence": pd.Series("high", index = df_new.index).where(has_exact, "low"),
        "staff_year": 2026,
        "city": df_new["city"],
        "lat": df_new["lat"],
        "lon": df_new["lon"],
        "source_note": (
            "Colleague's TPG workplace survey (Sep 2026). Address: " + df_new["Adresse principale"]
            + ". " + detail
        ),
    })

    return pd.concat([df, df_new_rows], ignore_index = True)


def execute(context):
    df = pd.DataFrame.from_dict(international_organizations_switzerland, orient = "index")
    df.index.name = "name"
    df = df.reset_index()

    for column in ("staff", "staff_year", "lat", "lon"):
        df[column] = pd.to_numeric(df[column], errors = "coerce")

    df = apply_tpg_survey(context, df)

    df["is_aggregate"] = df["staff_confidence"] == "aggregate"

    missing_coords = df[df["lat"].isna() | df["lon"].isna()]
    if len(missing_coords) > 0:
        logger.warning(
            "%d entries have no coordinates and are excluded from the map (the AGGREGATE__* "
            "entries covering multiple organizations, or single organizations whose site "
            "wasn't determined): %s",
            len(missing_coords),
            missing_coords["name"].tolist(),
        )

    build_map(df, os.path.join(context.path(), "international_organizations_map.html"))

    return df


def build_map(df, output_path):
    """
    One folium map with a CircleMarker per organization with known
    coordinates: blue, sized by staff count, for organizations with a known
    staff figure; a fixed-size gray dot for organizations with no staff
    figure at all. Organizations with no coordinates (mostly the
    multi-organization AGGREGATE__ entries, see module docstring) are
    excluded here - they are still returned by execute() though.
    """

    df_map = df[df["lat"].notna() & df["lon"].notna()].copy()

    center = [df_map["lat"].mean(), df_map["lon"].mean()]
    m = folium.Map(location = center, zoom_start = 9, tiles = None)
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    layer = folium.FeatureGroup(name = "Staff count", show = True)
    df_known = df_map[df_map["staff"].notna()]
    global_max = max(df_known["staff"].max(), 1) if len(df_known) > 0 else 1

    for _, row in df_map.iterrows():
        has_staff = pd.notna(row["staff"])
        staff_text = f"{row['staff']:,.0f}" if has_staff else "unknown"

        popup = (
            f"<b>{row['name']}</b><br>"
            f"Staff: {staff_text} (confidence: {row['staff_confidence']}"
            + (f", {row['staff_year']:.0f}" if pd.notna(row["staff_year"]) else "")
            + f")<br>{row['city']}<br><i>{row['source_note']}</i>"
        )
        tooltip = f"{row['name']}: {staff_text} staff"

        if has_staff:
            radius = MIN_RADIUS + MAX_EXTRA_RADIUS * (row["staff"] / global_max) ** 0.5
            folium.CircleMarker(
                location = (row["lat"], row["lon"]),
                radius = radius,
                color = "#1f77b4", weight = 1.2, fill = True,
                fill_color = "#1f77b4", fill_opacity = 0.75,
                popup = folium.Popup(popup, max_width = 320),
                tooltip = tooltip,
            ).add_to(layer)
        else:
            folium.CircleMarker(
                location = (row["lat"], row["lon"]),
                radius = UNKNOWN_STAFF_RADIUS,
                color = UNKNOWN_STAFF_COLOR, weight = 1.2, fill = True,
                fill_color = UNKNOWN_STAFF_COLOR, fill_opacity = 0.75,
                popup = folium.Popup(popup, max_width = 320),
                tooltip = tooltip,
            ).add_to(layer)

    layer.add_to(m)
    folium.LayerControl(collapsed = False).add_to(m)

    m.save(output_path)


international_organizations_switzerland = {

    # ------------------------------------------------------------------
    # Intergovernmental organizations (IOs) - Geneva canton, accord de siège
    # ------------------------------------------------------------------
    "CERN": {
        "staff": 2500,
        "staff_confidence": "high",
        "staff_year": None,
        "city": "Meyrin",
        "lat": 46.234026664517124, 
        "lon": 6.046623441336757,
        "source_note": "CERN's own figure (~2,500 employed staff); straddles France-Switzerland "
                        "border, legal seat in Geneva. Excludes >12,200 visiting scientists.",
    },
    "UN Office at Geneva (UNOG secretariat)": {
        "staff": 1600,
        "staff_confidence": "high",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.22422093083935, 
        "lon": 6.139795773949035,
        "source_note": "Palais des Nations secretariat only, not the whole UN 'family' in Geneva "
                        "(which is ~9,500 across all agencies combined).",
    },
    "WHO (World Health Organization)": {
        "staff": 1800,
        "staff_confidence": "high",
        "staff_year": 2025,
        "city": "Geneva",
        "lat": 46.2327568799887,
        "lon": 6.134273957548879,
        "source_note": "Campus had 2,400; ~40% of HQ posts (~1,000) being cut through mid-2026 "
                        "plus ~100 relocated. 1,800 is a post-cuts approximation, not confirmed final.",
    },
    "ICRC (International Committee of the Red Cross)": {
        "staff": 1130,
        "staff_confidence": "high",
        "staff_year": 2025,
        "city": "Geneva",
        "lat": 46.22166921600033, 
        "lon": 6.125621025271391,
        "source_note": "Post-cuts figure (Oct-Nov 2025); was ~1,400 before 240-270 positions cut.",
    },
    "WIPO (World Intellectual Property Organization)": {
        "staff": 1705,
        "staff_confidence": "high",
        "staff_year": 2023,
        "city": "Geneva",
        "lat": 46.22208271564581, 
        "lon": 6.137074450604425,
        "source_note": "Total agents as of 31/12/2023; 61.3% (1,045) permanent. Mostly Geneva-based "
                        "(few field offices).",
    },
    "ILO (International Labour Organization)": {
        "staff": 1255,
        "staff_confidence": "high",
        "staff_year": 2025,
        "city": "Geneva",
        "lat": 46.228860655262,
        "lon": 6.134670811789216,
        "source_note": "Geneva HQ specifically (of 3,654 global staff), per Le Temps, July 2025.",
    },
    "IOM (International Organization for Migration)": {
        "staff": 1000,
        "staff_confidence": "medium",
        "staff_year": 2025,
        "city": "Geneva",
        "lat": 46.233217212366654,
        "lon":  6.133161854569449,
        "source_note": "Estimated: ~250 cuts reported as ~20 per cent of Geneva staff (swissinfo, Mar 2025) "
                        "=> implies ~1,250 pre-cuts, ~1,000 post-cuts.",
    },
    "UNHCR (UN Refugee Agency)": {
        "staff": 800,
        "staff_confidence": "high",
        "staff_year": 2025,
        "city": "Geneva",
        "lat": 46.22065089035003, 
        "lon": 6.141107978068235,
        "source_note": "Tribune de Genève, April 2025, post major restructuring/cuts.",
    },
    "WTO (World Trade Organization)": {
        "staff": 620,
        "staff_confidence": "high",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.224096482516444,
        "lon":  6.148958484333984,
        "source_note": "Centre William Rappard. All WTO staff based at Geneva HQ.",
    },
    "ITU (International Telecommunication Union)": {
        "staff": 700,
        "staff_confidence": "high",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.22080166470073, 
        "lon": 6.136919648293679,
        "source_note": "ITU's own procurement page",
    },
    "WMO (World Meteorological Organization)": {
        "staff": 350,
        "staff_confidence": "high",
        "staff_year": 2024,
        "city": "Geneva",
        "lat": 46.22329184379723, 
        "lon": 6.146612621755805,
        "source_note": "UN Today: 'some 350 staff, mostly at WMO headquarters in Geneva'.",
    },
    "ITC (International Trade Centre)": {
        "staff": 300,
        "staff_confidence": "high",
        "staff_year": 2017,
        "city": "Geneva",
        "lat": 46.21758555269407,
        "lon":  6.141417442660982,
        "source_note": "Wikipedia infobox (2017); joint WTO/UNCTAD mandate.",
    },
    "EFTA (European Free Trade Association)": {
        "staff": 30,
        "staff_confidence": "high",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.2197,
        "lon": 6.1418,
        "source_note": "EFTA's own site: ~90 total staff, 'a third of whom are in Geneva' "
                        "(rest in Brussels/Luxembourg).",
    },
    "UNCTAD": {
        "staff": 930,
        "staff_confidence": "low",
        "staff_year": 2025,
        "city": "Geneva",
        "lat": 46.22793247425092,
        "lon":  6.142081749406182,
        "source_note": "Secretariat at UNOG/Palais des Nations; "
                        "https://rocketreach.co/unctad-management_b5c62040f42e0ca4",
    },
    "UPOV": {
        "staff": 31,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2244,
        "lon": 6.1432,
        "source_note": "LinkedIn company size bracket '11-50 employees' (Sep 2026); using midpoint. Location defaulted to UN headquarters",
    },
    "IPU (Inter-Parliamentary Union)": {
        "staff": 130,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.22717781264427, 
        "lon": 6.121658856536776,
        "source_note": "https://www.linkedin.com/company/inter-parliamentary-union/: 51-200 employees, averaged 130",
    },
    "South Centre": {
        "staff": 31,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.21399708306527, 
        "lon": 6.111252170290486, 
        "source_note": "LinkedIn company size bracket '11-50 employees' (Sep 2026); using midpoint.",
    },
    "ACWL (Advisory Centre on WTO Law)": {
        "staff": 30,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.21912570169344,
        "lon":  6.13411160839932,
        "source_note": "https://www.linkedin.com/company/advisory-centre-on-wto-law-acwl/: 11-50 employees, averaged to 30",
    },
    "ALIPH": {
        "staff": 30,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.216612593953585,
        "lon":  6.14849160358524,
        "source_note": "https://www.linkedin.com/company/aliphfoundation/: 11-50 employees, averaged to 30",
    },
    "GCERF": {
        "staff": 31,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.22370031368849, 
        "lon": 6.148058584999299,
        "source_note": "LinkedIn company size bracket '11-50 employees' (Sep 2026); using midpoint.",
    },
    "ATT Secretariat (Arms Trade Treaty Secretariat)": {
        "staff": 4,
        "staff_confidence": "high",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2180,
        "lon": 6.1380,
        "source_note": "Avenue de France 23, Geneva. ATT's own site lists a team of 4 staff "
                        "members.",
    },
    "OSCE Court of Conciliation and Arbitration": {
        "staff": 10,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.22048563149988, 
        "lon": 6.142784870319713,
        "source_note": "Not found. Defaulted to 10.",
    },
    "OIPC (International Civil Protection Organisation)": {
        "staff": None,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.1916048922903, 
        "lon": 6.1242250752957235,
        "source_note": "Not found.",
    },
    "BIE-UNESCO (International Bureau of Education)": {
        "staff": 19,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2316235784783, 
        "lon": 6.131449036490019,
        "source_note": "LinkedIn company page lists 19 employees (Sep 2026); size bracket shown "
                        "is a coarser '11-50 employees'.",
    },
    "IEC (International Electrotechnical Commission)": {
        "staff": 126,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.22077135650839, 
        "lon": 6.139106429598038,
        "source_note": "LinkedIn company size bracket '51-200 employees' (Sep 2026); using midpoint.",
    },

    # ------------------------------------------------------------------
    # Fiscal-agreement / privilege-and-immunity organizations
    # ------------------------------------------------------------------
    "ISO (International Organization for Standardization)": {
        "staff": 153,
        "staff_confidence": "low",
        "staff_year": 2008,
        "city": "Geneva",
        "lat": 46.22082626506031, 
        "lon": 6.098581525584169,
        "source_note": "Only figure found is from ISO's own 'ISO in figures 2008' doc - very "
                        "likely outdated. Central Secretariat full-time staff.",
    },
    "IATA (International Air Transport Association)": {
        "staff": None,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Le Grand-Saconnex (Geneva Airport)",
        "lat": 46.2381,
        "lon": 6.1089,
        "source_note": "HQ is Montreal; Geneva is the 'Executive Office'. Global staff ~1,600-4,100 "
                        "depending on source; Geneva-specific split not found.",
    },
    "DNDi (Drugs for Neglected Diseases initiative)": {
        "staff": 428,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.21970816197836, 
        "lon": 6.136710170290787,
        "source_note": "LinkedIn company page lists 428 employees, bracket '201-500' (Sep 2026); "
                        "this is DNDi's global headcount across Geneva HQ plus Brazil, DRC, India, "
                        "Japan, Kenya, Malaysia and a US affiliate - not Geneva-specific.",
    },
    "FIND (Foundation for Innovative New Diagnostics)": {
        "staff": 126,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.23015052559344, 
        "lon": 6.1265462196718055,
        "source_note": "LinkedIn company size bracket '51-200 employees' (Sep 2026); using midpoint. "
                        "Global total incl. regional hubs in Kenya, India, South Africa, Viet Nam - "
                        "not Geneva-specific.",
    },
    "GAIN (Global Alliance for Improved Nutrition)": {
        "staff": 126,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2204807694136, 
        "lon": 6.138636170897912,
        "source_note": "LinkedIn company size bracket '51-200 employees' (Sep 2026); using midpoint.",
    },
    "MMV (Medicines for Malaria Venture)": {
        "staff": 126,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.224375412832636,
        "lon":  6.1011552665954065,
        "source_note": "LinkedIn company size bracket '51-200 employees' (Sep 2026); using midpoint.",
    },
    "GICHD (Geneva International Centre for Humanitarian Demining)": {
        "staff": 65,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.220772024812646, 
        "lon": 6.143670656797006,
        "source_note": "GICHD's own site/LinkedIn: '65 members of staff representing 19 "
                        "nationalities' (Sep 2026).",
    },
    "HD Centre (Centre for Humanitarian Dialogue)": {
        "staff": 351,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.21884007262702,
        "lon":  6.137240202529399,
        "source_note": "LinkedIn company size bracket '201-500 employees' (Sep 2026); using "
                        "midpoint. Geneva HQ handles oversight/fundraising/HR only - most staff "
                        "work in regional hubs and country offices, so this overstates Geneva-based "
                        "headcount.",
    },
    "Interpeace": {
        "staff": 350,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Geneva",
        "lat": 46.22000552561988, 
        "lon": 6.143774525173255,
        "source_note": "https://www.linkedin.com/company/interpeace/ -> mid point 350.",
    },
    "MPP (Medicines Patent Pool)": {
        "staff": 78,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2205116888203, 
        "lon": 6.13871608521202,
        "source_note": "LinkedIn company page lists 78 employees (Sep 2026); size bracket shown "
                        "is a coarser '11-50 employees'.",
    },
    "ICoCA (International Code of Conduct Association)": {
        "staff": 27,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.23027011103278,
        "lon":  6.128286562641179,
        "source_note": "LinkedIn company page lists 27 employees (Sep 2026); size bracket shown "
                        "is a coarser '2-10 employees', likely stale.",
    },
    "GARDP (Global Antibiotic Research and Development Partnership)": {
        "staff": 126,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.219356819332795, 
        "lon": 6.136967605560228,
        "source_note": "LinkedIn company size bracket '51-200 employees' (Sep 2026); using midpoint.",
    },

    "WEF (World Economic Forum)": {
        "staff": 740,
        "staff_confidence": "high",
        "staff_year": 2024,
        "city": "Cologny",
        "lat": 46.225083871474475, 
        "lon": 6.191712283079087,
        "source_note": "WEF's 2023-24 Annual Report: 780 employees operate from Cologny HQ; "
                        "careers page more recently says 'over 600'. Midpoint used.",
    },
    "Gavi, the Vaccine Alliance": {
        "staff": 800,
        "staff_confidence": "high",
        "staff_year": 2026,
        "city": "Le Grand-Saconnex",
        "lat": 46.23009167042389,
        "lon": 6.126402922591274,
        "source_note": "Third-party workforce data range 719-851 (2026); small DC office exists too.",
    },
    "Global Fund (GFATM)": {
        "staff": 2600,
        "staff_confidence": "high",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.23009167042389, 
        "lon":6.126402922591274,
        "source_note": "Global Fund states explicitly: 'All staff are based in Geneva' "
                        "(no country offices).",
    },

    # ------------------------------------------------------------------
    # Non-Geneva IOs / financial institutions
    # ------------------------------------------------------------------
    "BIS (Bank for International Settlements)": {
        "staff": 1100,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Basel",
        "lat": 47.55076939751749, 
        "lon": 7.595028568782748,
        "source_note": "Wikipedia infobox: 1,300 total staff (incl. small Hong Kong/Mexico City "
                        "offices); older BIS annual reports (2004-2009) cited ~550-570, now outdated.",
    },
    "UPU (Universal Postal Union)": {
        "staff": 100,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Bern",
        "lat": 46.93901363414785, 
        "lon": 7.473198301061509,
        "source_note": "Not found. HQ: Weltpoststrasse 4, Bern. Assumed 100.",
    },

    # ------------------------------------------------------------------
    # Sports bodies
    #
    # FIFA, IOC and UEFA are all listed below for mapping purposes. A
    # proximity check against real STATENT (data/statent/statent.py's
    # get_international_organizations) found STATENT establishments right
    # next to each of them, classified under NOGA 931900 ("other sports
    # activities"), with employee counts in the same order of magnitude
    # (FIFA: 985 vs. our 800 estimate at 34.7m; IOC: 528 vs. 804 at 29.1m;
    # UEFA: 304 vs. 800 at 11.0m) - plausible evidence STATENT already
    # counts them.
    # ------------------------------------------------------------------
    "IOC (International Olympic Committee)": {
        "staff": 804,
        "staff_confidence": "medium",
        "staff_year": 2024,
        "city": "Lausanne",
        "lat": 46.518069469812055, 
        "lon": 6.596912178377091,
        "source_note": " 804 employees (2024) per ZoomInfo; "
                        "some sources cite '630+' at the Lausanne HQ specifically vs. a broader "
                        "'804' workforce figure. A proximity check found a plausible STATENT match "
                        "(528 employees, NOGA 931900) 29.1m away.",
    },
    "FIFA (Federation Internationale de Football Association)": {
        "staff": 800,
        "staff_confidence": "medium",
        "staff_year": 2024,
        "city": "Zurich",
        "lat": 47.3966,
        "lon": 8.5065,
        "source_note": "FIFA-Strasse 20, Zurich. ~800 FIFA staff have a Swiss employment contract "
                        "(2024); FIFA's global headcount (4,754 in 2024) is spread across Zurich, "
                        "Paris, Miami and regional offices, so the global figure would badly "
                        "overstate the Zurich site. A proximity check found a plausible STATENT "
                        "match (985 employees, NOGA 931900) 34.7m away.",
    },
    "UEFA (Union of European Football Associations)": {
        "staff": 800,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Nyon",
        "lat": 46.3825,
        "lon": 6.2371,
        "source_note": "House of European Football campus, Route de Geneve 46, Nyon (House of "
                        "European Football + La Clairiere + Bois-Bougy buildings combined, ~800 "
                        "staff); UEFA's total worldwide headcount (2,037 as of March 2026) includes "
                        "staff outside Nyon and would overstate the campus figure. A proximity "
                        "check found a plausible STATENT match (304 employees, NOGA 931900) 11.0m "
                        "away.",
    },

    # ------------------------------------------------------------------
    # Vaud-based international sports federations (individually named, on
    # top of AGGREGATE__sports_federations_vaud_excl_ioc below - the
    # aggregate is kept only as an external cross-check, not to avoid
    # listing these). Coordinates are approximate street/building-level
    # estimates, not verified geocodes. Many federations share the "Maison
    # du Sport International" building (Av. de Rhodanie 54, Lausanne) -
    # flagged per entry.
    #
    # Deliberately NOT added: ICF (International Canoe Federation) moved
    # its HQ to Budapest, no longer Vaud-based; GAISF was dissolved in
    # Sept 2023 (successor SportAccord, also Lausanne, not researched here).
    # ------------------------------------------------------------------
    "UCI (Union Cycliste Internationale)": {
        "staff": 194,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Aigle",
        "lat": 46.31852041340734, 
        "lon": 6.933342064751977,
        "source_note": "Third-party estimates conflict: 145 (RocketReach) vs. 242 (ContactOut); "
                        "average used. Allee Ferdi Kubler 12, Aigle.",
    },
    "FIVB (Federation Internationale de Volleyball)": {
        "staff": 234,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.50743993814206, 
        "lon": 6.635781116828869,
        "source_note": "Third-party estimates conflict: 228 vs. 239 employees; average used.",
    },
    "FIG (Federation Internationale de Gymnastique)": {
        "staff": 30,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.517117527434976, 
        "lon": 6.63505951118149,
        "source_note": "FIG's own site: HQ in Lausanne employs ~25-35 people (midpoint 30). A "
                        "separately-reported '183 total employees' figure is treated as global/"
                        "not HQ-specific and excluded. Avenue de la Gare 12.",
    },
    "World Archery": {
        "staff": 10,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.51575361763418, 
        "lon": 6.608840533675249,
        "source_note": "World Archery's own site: the Office at Maison du Sport International "
                        "currently has 10 employees (the separate World Archery Excellence Centre "
                        "has its own independent staff, not counted here).",
    },
    "CAS (Court of Arbitration for Sport)": {
        "staff": None,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Lausanne",
        "lat": 46.5195,
        "lon": 6.6285,
        "source_note": "Not found - no staff count located for the Lausanne secretariat/Court "
                        "Office.",
    },
    "FEI (Federation Equestre Internationale)": {
        "staff": 103,
        "staff_confidence": "medium",
        "staff_year": 2025,
        "city": "Lausanne",
        "lat": 46.51201505779624, 
        "lon": 6.628400985582545,
        "source_note": "FEI's own about page: 103 employees from 25 countries at the Lausanne "
                        "HQ (near Lausanne-Sebeillon/Delices station) - corrects an earlier, "
                        "incorrect 'Vevey' assumption. A separately-reported '~475 employees "
                        "(May 2025)' figure looks implausible for this org's scale and is excluded.",
    },
    "FISU (International University Sports Federation)": {
        "staff": 45,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.52117317136309, 
        "lon": 6.582627786904209,
        "source_note": "Third-party estimate: ~45 employees and consultants based at HQ "
                        "(Batiment Synathlon, UNIL-Centre campus, Dorigny).",
    },
    "WBSC (World Baseball Softball Confederation)": {
        "staff": 69,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Pully",
        "lat": 46.5093,
        "lon": 6.6602,
        "source_note": "Third-party estimates conflict sharply: 34 vs. 103 employees; average "
                        "used. Avenue General-Guisan 45, Pully (adjacent to Lausanne).",
    },
    "FIH (International Hockey Federation)": {
        "staff": 118,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.527699764489235, 
        "lon": 6.630525703633005,
        "source_note": "Third-party estimate: 118 employees (most specific of several wildly "
                        "conflicting counts, from '11-50' to '201-500' depending on source). Rue "
                        "du Valentin 61.",
    },
    "FIE (International Fencing Federation)": {
        "staff": 102,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.515708522813306, 
        "lon": 6.608801590174954,
        "source_note": "Third-party estimate: 102 employees, in tension with a separately-shown "
                        "'11-50' LinkedIn bracket. Maison du Sport International, Av. de Rhodanie 54.",
    },
    "ITTF (International Table Tennis Federation)": {
        "staff": 200,
        "staff_confidence": "low",
        "staff_year": 2025,
        "city": "Lausanne",
        "lat": 46.515593161994566,
        "lon":  6.6094762542277605,
        "source_note": "Third-party estimate: ~200 employees (May 2025), staff spread across 6 "
                        "continents so this overstates Lausanne-specific headcount; a separate "
                        "RocketReach figure of 68 conflicts sharply. Av. de Rhodanie 54B.",
    },
    "ISU (International Skating Union)": {
        "staff": 78,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.509967254234205, 
        "lon": 6.629240852476557,
        "source_note": "LinkedIn size bracket conflicts between sources ('11-50' vs '51-200'); "
                        "midpoint of the combined range used. Chemin de Brillancourt 4.",
    },
    "World Triathlon": {
        "staff": 35,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Lausanne",
        "lat": 46.515360486830005, 
        "lon": 6.609302495360776,
        "source_note": "World Triathlon's own Secretary General job posting: ~35 staff based in "
                        "Lausanne and working remotely, with additional offices in Madrid and "
                        "Vancouver. A separately-reported '192 employees' figure looks implausible "
                        "and is excluded.",
    },
    "FISA / World Rowing": {
        "staff": None,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Lausanne",
        "lat": 46.515360486830005,
        "lon": 6.609302495360776,
        "source_note": "Not found - no staff count located. Maison du Sport International, Av. "
                        "de Rhodanie 54.",
    },
    "FIBA (International Basketball Federation)": {
        "staff": 126,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Mies",
        "lat": 46.2962453950795, 
        "lon": 6.166567401059711,
        "source_note": "LinkedIn size bracket '51-200' (midpoint used); a separately-reported "
                        "'1,495 employees' / '1,001-5,000' figure looks implausible for this org's "
                        "HQ scale (likely includes referees/officials or a data error) and is "
                        "excluded. Route Suisse 5, Mies.",
    },

    # ------------------------------------------------------------------
    # NGOs and conservation bodies
    # ------------------------------------------------------------------
    "IUCN (International Union for Conservation of Nature)": {
        "staff": 150,
        "staff_confidence": "not_found",
        "staff_year": None,
        "city": "Gland",
        "lat": 46.41556863240133, 
        "lon": 6.27803714160435,
        "source_note": "IUCN's own site: ~1,000 staff across 50+ countries (global Secretariat "
                        "total); no Gland-specific breakout found. Assumed 150.",
    },
    "WWF International": {
        "staff": 70,
        "staff_confidence": "low",
        "staff_year": 2016,
        "city": "Gland",
        "lat": 46.4183,
        "lon": 6.2571,
        "source_note": "Dated: swissinfo (2016) reported 170 Switzerland-based staff before a "
                        "restructuring relocated ~100 abroad, implying ~70 remaining. Likely "
                        "changed since.",
    },
    "World Council of Churches (WCC)": {
        "staff": 199,
        "staff_confidence": "low",
        "staff_year": None,
        "city": "Le Grand-Saconnex",
        "lat": 46.229647310469005,
        "lon": 6.128278759652945,
        "source_note": "Third-party estimate (ContactOut). Ecumenical Centre, Chemin du Pommier.",
    },

    # ------------------------------------------------------------------
    # Additional organizations identified in a later pass (batch 2, Sep 2026).
    # Coordinates are approximate (street/building level from the address
    # found, not a verified geocode). Several entries below use a global
    # headcount because no Geneva-specific split was found - flagged in
    # each source_note; treat "low" confidence entries as rough estimates.
    #
    # Deliberately NOT added: UCI, FIBA, FIVB, FIG, CAS, World Archery and
    # other Vaud-based sports federations - these are almost certainly
    # already folded into AGGREGATE__sports_federations_vaud_excl_ioc below,
    # so adding them individually would double-count.
    # ------------------------------------------------------------------
    "UNAIDS (Joint UN Programme on HIV/AIDS)": {
        "staff": 19,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2334,
        "lon": 6.1354,
        "source_note": "Geneva Solutions (2026): global workforce cut from ~600 to <300, and the "
                        "Geneva HQ itself cut from 127 to 19 as functions relocated to Bonn. "
                        "20 Avenue Appia, next to WHO.",
    },
    "IFRC (International Federation of Red Cross and Red Crescent Societies)": {
        "staff": 350,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.22168760230457, 
        "lon": 6.125648103800616,
        "source_note": "IFRC states its ~2,500-strong secretariat is 86% based in 5 regional and "
                        "sub-regional offices; the remaining ~14% (~350) covers Geneva HQ plus other "
                        "locations - not a clean Geneva-only figure. Chemin des Crets 17, "
                        "Petit-Saconnex.",
    },
    "MSF (Medecins Sans Frontieres) - Switzerland section & International Office": {
        "staff": 400,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.22808653823579, 
        "lon": 6.129132104546716,
        "source_note": "MSF Switzerland's own figures: ~400 staff across its Geneva and Zurich "
                        "offices (Operational Centre Geneva alone is ~300). The MSF International "
                        "Office is also headquartered in Geneva but is organizationally distinct and "
                        "not separately counted here.",
    },
    "International Baccalaureate (IB) - Geneva Foundation Office": {
        "staff": 12,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2095,
        "lon": 6.1435,
        "source_note": "IB's own site: Foundation Office in Geneva has ~12 staff (Director "
                        "General, legal/compliance, IP/tax); >750 employees globally across all IB "
                        "offices, so the global figure would badly overstate the Geneva site.",
    },
    "UNITAR (UN Institute for Training and Research)": {
        "staff": 400,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.223324989667596, 
        "lon": 6.146452495176367,
        "source_note": "UNITAR's own site: ~400 staff and collaborators; this is the global total "
                        "across Geneva HQ plus New York, Hiroshima and Bonn offices, not "
                        "Geneva-specific.",
    },
    "UNRISD (UN Research Institute for Social Development)": {
        "staff": 20,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2258,
        "lon": 6.1421,
        "source_note": "UNRISD is a small, Geneva-only institute; sources describe a core staff "
                        "of around 20. Avenue de la Paix 8-14.",
    },
    "UNDRR (UN Office for Disaster Risk Reduction)": {
        "staff": 120,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.22333429656879, 
        "lon": 6.146440242338404,
        "source_note": "UNDRR's own site describes 'a small, nimble work team of around 120 "
                        "staff'; unclear whether this is Geneva HQ only or includes the 5 regional "
                        "offices (Nairobi, Panama City, Cairo, Bangkok, Brussels). 9-11 rue de "
                        "Varembe.",
    },
    "GCSP (Geneva Centre for Security Policy)": {
        "staff": 191,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2254,
        "lon": 6.1425,
        "source_note": "Maison de la Paix, Chemin Eugene-Rigot 2D, Geneva. Third-party counts "
                        "conflict (124 per one source vs. a LinkedIn bracket implying ~191); "
                        "higher estimate kept to match the earlier proximity-check note. A "
                        "proximity check found a plausible STATENT match (89 employees, NOGA "
                        "949901) 5.1m away.",
    },
    "DCAF (Geneva Centre for Security Sector Governance)": {
        "staff": 220,
        "staff_confidence": "low",
        "staff_year": 2023,
        "city": "Geneva",
        "lat": 46.22230747405726, 
        "lon": 6.143791992943609,
        "source_note": "DCAF's own site: >220 staff, but explicitly across 16 offices worldwide "
                        "(Africa, Europe, Middle East, Latin America) - global, not Geneva-specific.",
    },
    "Small Arms Survey": {
        "staff": 43,
        "staff_confidence": "medium",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2255,
        "lon": 6.1409,
        "source_note": "Third-party estimate: ~43 employees (LinkedIn bracket separately shown "
                        "as '11-50'). Based at the Graduate Institute's Maison de la Paix.",
    },
    "IRU (International Road Transport Union)": {
        "staff": 100,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.2144,
        "lon": 6.1296,
        "source_note": "La Voie-Creuse 16, Geneva. Third-party counts conflict sharply (51-200 "
                        "LinkedIn bracket vs. a separate 245-employee figure); midpoint of the "
                        "bracket used. A proximity check found a near-exact STATENT match (93 "
                        "employees, NOGA 941200) 2.6m away.",
    },
    "World Scout Bureau": {
        "staff": 130,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Geneva",
        "lat": 46.1943182299708, 
        "lon": 6.142878703335503,
        "source_note": "WOSM's own site: ~120-130 professional staff, but explicitly spread "
                        "across 9 locations worldwide (6 regional offices), not Geneva-only. Rue "
                        "Henri-Christine 5.",
    },
    "IIHF (International Ice Hockey Federation)": {
        "staff": 28,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Zurich",
        "lat": 47.3660,
        "lon": 8.5310,
        "source_note": "Conflicting third-party counts: 24 employees (ZoomInfo) vs. 32 staff "
                        "members (IIHF's own site); midpoint used. Brandschenkestrasse 50.",
    },
    "IHF (International Handball Federation)": {
        "staff": 19,
        "staff_confidence": "low",
        "staff_year": 2019,
        "city": "Basel",
        "lat": 47.5453,
        "lon": 7.5905,
        "source_note": "Dated figure: 19 staff members reported for 2017-2021. Peter "
                        "Merian-Strasse 23.",
    },
    "Basel Institute on Governance": {
        "staff": 150,
        "staff_confidence": "low",
        "staff_year": 2026,
        "city": "Basel",
        "lat": 47.55055517168494, 
        "lon": 7.58071870065973,
        "source_note": "Basel Institute's own site: 'global team of over 150 staff from 32 "
                        "countries', explicitly including staff duty-stationed abroad (Lima "
                        "regional office, Indonesia, Latin America, Sub-Saharan Africa) - global, "
                        "not Basel-only.",
    },

    # ------------------------------------------------------------------
    # Aggregates - NOT single organizations, cannot be pinned to one location
    # ------------------------------------------------------------------
    "AGGREGATE__ngo_geneva_with_jobs": {
        "staff": 3834,
        "staff_confidence": "aggregate",
        "staff_year": 2024,
        "city": "Geneva (canton-wide)",
        "lat": None,
        "lon": None,
        "source_note": "OCSTAT/CAGI: 250 NGOs in Geneva canton with >=1 job, totalling 3,834 jobs. "
                        "Covers many distinct organizations - do not treat as one worksite.",
    },
    "AGGREGATE__permanent_missions_consulates_geneva": {
        "staff": 4274,
        "staff_confidence": "aggregate",
        "staff_year": 2025,
        "city": "Geneva (canton-wide)",
        "lat": None,
        "lon": None,
        "source_note": "OCSTAT Nov 2025 report. Staff of permanent missions/consulates accredited "
                        "to UN/other IOs in Geneva - many distinct locations (embassies/missions).",
    },
    "AGGREGATE__sports_federations_vaud_excl_ioc": {
        "staff": 2150,
        "staff_confidence": "aggregate",
        "staff_year": None,
        "city": "Vaud canton (mostly Lausanne area)",
        "lat": None,
        "lon": None,
        "source_note": "City of Lausanne / Canton Vaud / IOC-commissioned AISTS study: ~60 "
                        "international sports federations, excluding the IOC itself, generating "
                        "~2,150 jobs and CHF 1.07bn/year.",
    },
    "AGGREGATE__intergovernmental_orgs_geneva_total": {
        "staff": 29011,
        "staff_confidence": "aggregate",
        "staff_year": 2025,
        "city": "Geneva (canton-wide)",
        "lat": None,
        "lon": None,
        "source_note": "OCSTAT Nov 2025: total across all 38 Geneva-based IOs (siege + fiscal + "
                        "privileges/immunities agreements). Cross-check total: individually "
                        "identified orgs above sum to well under this - remaining gap is in the "
                        "'not_found' entries and WHO/IOM/ICRC post-cuts uncertainty.",
    },
}

