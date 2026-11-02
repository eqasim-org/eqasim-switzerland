"""Audit provisional campuses, nominal compatibility and OD destination coverage."""

import argparse
from html import escape
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium

from analysis.tertiary_education_od import InputContext
from data.od.education_matrix import prepare_observations
from data.statent.statent import read_statent, impute_locations
from data.statent.education_institutions import build_registry, DEFAULT_CAMPUS_DISTANCE
from data.structural_survey.structural_survey import TERTIARY_INSTITUTIONS
from data.od.tertiary_education_smoothing import estimate_profiles


def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.statent.education_institutions")
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")
    from data.statent.education_enrolment import DEFAULT_MAPPING
    context.config("data_path")
    context.config("education_institution_links", str(Path(context.config("data_path")) / DEFAULT_MAPPING))
    context.config("analysis_path")
    context.config("education_campus_cluster_distance", DEFAULT_CAMPUS_DISTANCE)
    context.stage("data.od.tertiary_education_matrix")


def create_map(registry, output_path, institution_links=None, reference_sites=None):
    """Show provisional site clusters, their review flags and named link evidence."""
    if institution_links is None:
        # Generic registry maps can omit names; never read source-relative data.
        institution_links = pd.DataFrame(columns=["statent_enterprise_id", "institution_name", "match_status"])
    labels = {}
    for enterprise, group in institution_links.groupby("statent_enterprise_id"):
        labels["STATENT_ENTERPRISE_" + enterprise] = "; ".join(
            "%s (%s)" % (row.institution_name, row.match_status) for row in group.itertuples())
    campus_table = registry["campuses"]
    campuses = gpd.GeoDataFrame(
        campus_table, geometry=gpd.points_from_xy(campus_table["x"], campus_table["y"]),
        crs="EPSG:2056").to_crs("EPSG:4326")
    map_view = folium.Map(location=[46.8, 8.2], zoom_start=8, tiles=None)
    folium.TileLayer(
        tiles="https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/"
              "default/current/3857/{z}/{x}/{y}.jpeg",
        attr='<a href="https://www.swisstopo.admin.ch/">&copy; swisstopo</a>',
        name="Swisstopo national map", max_native_zoom=18, max_zoom=20,
    ).add_to(map_view)
    colors = {"UH": "blue", "FH_PH": "green", "higher_vocational": "orange"}
    campus_layer = folium.FeatureGroup(name="Provisional campus clusters", show=True)
    for row in campuses.itertuples():
        label = labels.get(row.institution_id, "Unnamed STATENT institution")
        popup = (f"<b>{escape(label)}</b><br>Provisional campus<br>{row.campus_id}<br>"
                 f"Types: {row.institution_types}<br>Establishments: {row.establishments}<br>"
                 f"Eligible teaching sites: {row.eligible_sites}<br>Zones: {row.destination_zones}<br>"
                 f"Extent: {row.extent_m:.0f} m<br>Grouping review: {row.grouping_review}<br>"
                 "Candidate names are provisional; student capacity is unset.")
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x], radius=5,
            color="red" if row.grouping_review else colors.get(row.institution_types, "purple"),
            fill=True, tooltip=escape(label) + " — " + row.campus_id,
            popup=folium.Popup(popup, max_width=550),
        ).add_to(campus_layer)
    campus_layer.add_to(map_view)
    site_table = registry["establishments"]
    sites = gpd.GeoDataFrame(
        site_table, geometry=gpd.points_from_xy(site_table["x"], site_table["y"]),
        crs="EPSG:2056").to_crs("EPSG:4326")
    site_layer = folium.FeatureGroup(name="Individual establishments", show=False)
    for row in sites.itertuples():
        label = labels.get(row.institution_id, "Unnamed STATENT institution")
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x], radius=3,
            color=colors.get(row.institution_type, "purple") if row.eligible_teaching_site else "gray",
            tooltip=escape(label) + " — " + str(row.statent_local_id),
            popup=folium.Popup(f"<b>{escape(label)}</b><br>Establishment: {row.statent_local_id}<br>{row.campus_id}<br>"
                               f"NOGA: {row.noga}<br>Eligible teaching site: {row.eligible_teaching_site}<br>"
                               f"Destination zone: {row.zone_id}", max_width=550),
        ).add_to(site_layer)
    site_layer.add_to(map_view)
    if reference_sites is not None:
        references = gpd.GeoDataFrame(
            reference_sites, geometry=gpd.points_from_xy(reference_sites["x"], reference_sites["y"]),
            crs="EPSG:2056").to_crs("EPSG:4326")
        reference_layer = folium.FeatureGroup(name="Official location reference points", show=False)
        for row in references.itertuples():
            folium.Marker(
                [row.geometry.y, row.geometry.x],
                tooltip=escape(row.institution_name + " — " + row.reference_site),
                popup=folium.Popup(escape(row.address) + '<br><a href="' + escape(row.source_url, quote=True)
                                  + '" target="_blank">Official source</a><br>Reference point; not a campus boundary.',
                                  max_width=550),
                icon=folium.Icon(color="black", icon="info-sign"),
            ).add_to(reference_layer)
        reference_layer.add_to(map_view)
    folium.LayerControl().add_to(map_view)
    map_view.save(str(Path(output_path) / "campuses.html"))


def audit(df_statent, registry, df_survey, df_zones, output_path, profiles=None, institution_links=None):
    """Compare campus/type coverage with OD evidence before using it for assignment."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, table in registry.items():
        table.to_csv(output_path / (name + ".csv"), index=False)
    create_map(registry, output_path, institution_links=institution_links)
    (output_path / "README.md").write_text(
        "# Provisional tertiary institution and campus registry\n\n"
        "Institutions are anonymous STATENT enterprise groups, not verified university "
        "identities. Campuses are connected clusters of nearby sites within each "
        "enterprise, using the configured distance. The sensitivity table tests "
        "250, 500 and 1,000 metres. Cluster extent and type conflicts flag review.\n\n"
        "Individual establishment zone links are preserved because campuses can "
        "cross municipalities or quarters. Municipality destination matching also "
        "allows establishments in its quarters; quarter matching is exact.\n\n"
        "Compatibility is nominal local-NOGA evidence, not a list of programmes "
        "actually offered. Ancillary sites and conflicting local/enterprise types "
        "are retained for review but not automatically eligible. Student capacity "
        "and verified institution names remain unset; employment is not enrolment.\n\n"
        "Observed compatibility tables retain all usable tertiary OD observations. "
        "Missing candidates do not establish that the survey is wrong. They require "
        "provider review, programme information or zone-selection rules before "
        "assignment. The smoothed compatibility table reports the probability mass "
        "requiring that review for each origin and study level.\n\n"
        "campuses.html is an interactive map of provisional clusters and establishments. "
        "Blue = UH, green = FH/PH, orange = higher vocational, red = cluster review, "
        "gray = establishments not automatically eligible. Its swisstopo basemap requires "
        "an internet connection.\n",
        encoding="utf-8")
    sensitivity = []
    for distance in [250.0, 500.0, 1000.0]:
        alternative = build_registry(df_statent, distance)
        sensitivity.append({"distance_m": distance, "institutions": len(alternative["institutions"]),
                            "campuses": len(alternative["campuses"]),
                            "review_campuses": alternative["campuses"]["grouping_review"].sum()})
    pd.DataFrame(sensitivity).to_csv(output_path / "clustering_sensitivity.csv", index=False)

    compatibility = registry["compatibility"]
    zone_candidates = pd.concat([
        compatibility.assign(destination_zone_id=compatibility["zone_id"], zone_match="exact"),
        compatibility.assign(destination_zone_id=compatibility["zone_municipality_id"], zone_match="municipality"),
    ]).drop_duplicates(["destination_zone_id", "enterprise_id", "study_level"])
    zone_candidates.to_csv(output_path / "zone_candidates.csv", index=False)
    counts = zone_candidates.groupby(["destination_zone_id", "study_level"]).agg(
        eligible_sites=("enterprise_id", "nunique"), campuses=("campus_id", "nunique"),
        institutions=("institution_id", "nunique"),
    ).reset_index()
    counts.merge(df_zones, left_on="destination_zone_id", right_on="zone_id", how="left").to_csv(
        output_path / "zone_compatibility.csv", index=False)

    observations = prepare_observations(df_survey, tertiary_only=True)
    # Keep programme labels and review flags for auditing, not for filtering the pooled OD sample.
    details = df_survey.loc[observations.index, ["study_level", "study_institution_conflict"]]
    observations = observations.join(details)
    support = observations.groupby(["survey_year", "education_zone_id", "study_level"]).agg(
        respondents=("weight", "size"), weighted_persons=("weight", "sum"),
        study_institution_conflicts=("study_institution_conflict", "sum"),
    ).reset_index()
    support = support.merge(counts, left_on=["education_zone_id", "study_level"],
                            right_on=["destination_zone_id", "study_level"], how="left")
    support[["eligible_sites", "campuses", "institutions"]] = support[["eligible_sites", "campuses", "institutions"]].fillna(0).astype(int)
    support = support.merge(df_zones, left_on="education_zone_id", right_on="zone_id", how="left")
    support.to_csv(output_path / "observed_zone_compatibility.csv", index=False)
    missing = support[support["eligible_sites"].eq(0)]
    missing.to_csv(output_path / "zones_without_compatible_sites.csv", index=False)
    if profiles is not None:
        probability_support = []
        for level in sorted(compatibility["study_level"].unique()):
            eligible_zones = counts.loc[counts["study_level"].eq(level), "destination_zone_id"]
            probability = profiles.loc[:, profiles.columns.isin(eligible_zones)].sum(axis=1)
            probability_support.append(pd.DataFrame({
                "origin_zone_id": profiles.index, "study_level": level,
                "compatible_probability": probability.to_numpy(),
                "requires_review_or_reselection_probability": 1 - probability.to_numpy()}))
        pd.concat(probability_support, ignore_index=True).to_csv(
            output_path / "smoothed_zone_compatibility.csv", index=False)
    support["without_compatible_site"] = support["respondents"].where(support["eligible_sites"].eq(0), 0)
    return support.groupby("survey_year").agg(
        observations=("respondents", "sum"), without_compatible_site=("without_compatible_site", "sum"))


def execute(context):
    zones = context.stage("data.spatial.zones")
    pdf, cdf = context.stage("data.od.tertiary_education_matrix")
    # Audit all domestic origin rows against the actual smoothed stage output.
    rows = zones["zone_level"].isin(["municipality", "quarter"]).to_numpy()
    profiles = pd.DataFrame(pdf[rows], index=zones.loc[rows, "zone_id"], columns=zones["zone_id"])
    return audit(context.stage("data.statent.statent"),
                 context.stage("data.statent.education_institutions"),
                 context.stage("data.structural_survey.structural_survey"),
                 zones, Path(context.config("analysis_path")) / "education_institutions", profiles,
                 pd.read_csv(context.config("education_institution_links"), dtype=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    context = InputContext(args.data_path)
    # Retain original row-based IDs while imputing only the small candidate subset.
    statent = read_statent(args.data_path)
    selected = statent["noga"].isin(TERTIARY_INSTITUTIONS) | statent["enterprise_noga"].isin(TERTIARY_INSTITUTIONS)
    statent = impute_locations(context, statent[selected].copy())
    registry = build_registry(statent, context.config("education_campus_cluster_distance"))
    survey = context.stage("data.structural_survey.structural_survey")
    zones = context.stage("data.spatial.zones")
    municipalities, mapping = context.stage("data.spatial.municipalities")
    # The same estimator yields small profiles without expanding a dense matrix.
    profiles, diagnostics = estimate_profiles(
        prepare_observations(survey, tertiary_only=True), zones, municipalities,
        context.stage("data.spatial.quarters"), mapping,
        context.config("tertiary_od_prior_strength"), context.config("tertiary_od_neighbors"),
        context.config("tertiary_od_bandwidth"))
    from data.statent.education_enrolment import DEFAULT_MAPPING
    links = pd.read_csv(Path(args.data_path) / DEFAULT_MAPPING, dtype=str)
    print(audit(statent, registry, survey, zones, args.output_path, profiles, links).to_string())


if __name__ == "__main__":
    main()
