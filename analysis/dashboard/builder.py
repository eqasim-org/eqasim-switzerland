from __future__ import annotations

import json
from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html


MODE_COLORS = {
    "car": "#1f77b4",
    "pt": "#ff7f0e",
    "walk": "#2ca02c",
    "bike": "#d62728",
    "car_passenger": "#9467bd",
    "truck": "#7f7f7f",
}


def _read_csv(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    for sep in [";", ",", "\t"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    return pd.read_csv(path)


def _find_simulation_root(candidate: str | Path):
    root = Path(candidate).resolve()
    if root.name.startswith("simulation_output"):
        return root
    if root.is_dir() and (root / "scorestats.csv").exists():
        return root
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("simulation_output"):
            return subdir
    return root


def _load_mode_stats(root: Path):
    path = root / "modestats.csv"
    if not path.exists():
        return pd.DataFrame(columns=["iteration", "mode", "share"])
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["iteration", "mode", "share"])
    long = df.melt(id_vars=["iteration"], var_name="mode", value_name="share")
    return long.sort_values(["iteration", "mode"]).reset_index(drop=True)


def _load_score_stats(root: Path):
    path = root / "scorestats.csv"
    if not path.exists():
        return pd.DataFrame(columns=["iteration", "avg_executed", "avg_average", "avg_best", "avg_worst"])
    return _read_csv(path)


def _load_count_summary(root: Path):
    stats = {}
    stats_path = root / "compare_counts_weekdays" / "flow_comparison_stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as handle:
                stats = json.load(handle)
        except Exception:
            stats = {}

    target_path = root / "compare_counts_weekdays" / "target_flow.csv"
    sim_path = root / "output_traffic_flow_daily_counts.csv"
    count_df = None
    if target_path.exists() and sim_path.exists():
        observed = _read_csv(target_path)
        simulated = _read_csv(sim_path)
        if observed is not None and simulated is not None:
            observed = observed.rename(columns={"count": "observed"}) if "count" in observed.columns else observed
            if "dailyCount" in simulated.columns:
                simulated = simulated.rename(columns={"dailyCount": "simulated"})
            count_df = observed.merge(simulated[["linkId", "simulated"]], on="linkId", how="outer")
            count_df["absolute_error"] = (count_df["simulated"] - count_df["observed"]).fillna(0)
            count_df["relative_error_pct"] = 100 * count_df["absolute_error"] / count_df["observed"].replace(0, pd.NA)
    return stats, count_df


def _load_population_summary(root: Path):
    persons = None
    trips = None
    person_candidates = [root / "output_persons.csv.gz", root / "switzerland_persons.csv"]
    for candidate in person_candidates:
        if candidate.exists():
            persons = _read_csv(candidate)
            break

    trip_candidates = [root / "output_trips.csv.gz", root / "switzerland_trips.csv"]
    for candidate in trip_candidates:
        if candidate.exists():
            trips = _read_csv(candidate)
            break
    return persons, trips


def _network_html_options(root: Path):
    counts_dir = root / "compare_counts_weekdays"
    if not counts_dir.exists():
        return []
    items = []
    for html_path in sorted(counts_dir.glob("*.html")):
        if "counts_on_network" in html_path.name:
            label = html_path.name.replace("counts_on_network_", "").replace(".html", "").title()
            items.append({"label": label or "Overview", "value": str(html_path)})
    if not items:
        overall = counts_dir / "Switzerland_counts_comparaison.html"
        if overall.exists():
            items.append({"label": "Switzerland overview", "value": str(overall)})
    return items


def _build_cards(root: Path):
    persons, trips = _load_population_summary(root)
    score = _load_score_stats(root)
    values = {
        "population": f"{len(persons):,}" if persons is not None else "-",
        "trips": f"{len(trips):,}" if trips is not None else "-",
        "avg_score": f"{score['avg_executed'].iloc[-1]:.2f}" if not score.empty and "avg_executed" in score.columns else "-",
    }
    return [
        {"title": "Population", "value": values["population"], "color": "#1f77b4"},
        {"title": "Trips", "value": values["trips"], "color": "#2ca02c"},
        {"title": "Avg score", "value": values["avg_score"], "color": "#ff7f0e"},
        {"title": "Outputs", "value": "1", "color": "#9467bd"},
    ]


def create_dashboard_app(simulation_root):
    root = _find_simulation_root(simulation_root)
    score_df = _load_score_stats(root)
    mode_df = _load_mode_stats(root)
    flow_stats, count_df = _load_count_summary(root)
    network_options = _network_html_options(root)
    persons, trips = _load_population_summary(root)
    cards = _build_cards(root)

    def overview_content():
        overview = []

        score_fig = px.line(
            score_df,
            x="iteration",
            y=[c for c in score_df.columns if c != "iteration"],
            title="Simulation score evolution",
            template="plotly_white",
        ) if not score_df.empty else None
        if score_fig is not None:
            score_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            overview.append(html.Div([html.H3("Score evolution"), dcc.Graph(figure=score_fig, config={"displayModeBar": False})], style={"marginBottom": "20px"}))

        if not mode_df.empty:
            final_modes = mode_df[mode_df["iteration"] == mode_df["iteration"].max()].copy()
            mode_fig = px.bar(
                final_modes,
                x="mode",
                y="share",
                color="mode",
                color_discrete_map=MODE_COLORS,
                title="Final mode shares",
                template="plotly_white",
            )
            mode_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis_tickformat="%")
            overview.append(html.Div([html.H3("Final mode shares"), dcc.Graph(figure=mode_fig, config={"displayModeBar": False})], style={"marginBottom": "20px"}))

        if persons is not None and "age" in persons.columns:
            age = persons["age"].dropna()
            if len(age):
                age_fig = px.histogram(age, nbins=25, title="Synthetic population age distribution", template="plotly_white")
                age_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                overview.append(html.Div([html.H3("Population age distribution"), dcc.Graph(figure=age_fig, config={"displayModeBar": False})]))

        return overview or [html.Div("No overview data available.")]

    def counts_content():
        figures = []
        if flow_stats:
            metric_items = []
            for key in ["r2", "rmse", "mae", "bias", "mape"]:
                if key in flow_stats:
                    metric_items.append(html.Div(f"{key.upper()}: {flow_stats[key]}", style={"padding": "8px 12px", "borderRadius": "12px", "background": "#f1f5f9", "marginRight": "10px", "marginBottom": "10px"}))
            figures.append(html.Div(metric_items, style={"display": "flex", "flexWrap": "wrap", "marginBottom": "20px"}))

        if count_df is not None and not count_df.empty:
            scatter = px.scatter(
                count_df.dropna(subset=["observed", "simulated"]),
                x="observed",
                y="simulated",
                title="Observed vs simulated counts",
                hover_name="linkId",
                template="plotly_white",
            )
            scatter.add_shape(type="line", x0=count_df["observed"].min(), y0=count_df["observed"].min(), x1=count_df["observed"].max(), y1=count_df["observed"].max(), line=dict(dash="dash", color="black"))
            scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            figures.append(dcc.Graph(figure=scatter, config={"displayModeBar": False}))

            error_hist = px.histogram(count_df["absolute_error"].dropna(), nbins=30, title="Absolute count error distribution", template="plotly_white")
            error_hist.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            figures.append(dcc.Graph(figure=error_hist, config={"displayModeBar": False}))
        else:
            figures.append(html.Div("No count-validation data available."))
        return figures

    def network_content():
        plots = []
        if network_options:
            plots.append(dcc.Dropdown(id="network-dropdown", options=[{"label": item["label"], "value": item["value"]} for item in network_options], value=network_options[0]["value"], clearable=False, style={"maxWidth": "500px", "marginBottom": "15px"}))
            plots.append(html.Iframe(src=network_options[0]["value"], style={"width": "100%", "height": "760px", "border": "0"}))
        elif count_df is not None and not count_df.empty:
            volume = px.scatter(
                count_df.dropna(subset=["simulated", "observed"]),
                x="linkId",
                y="simulated",
                title="Network volume profile",
                template="plotly_white",
            )
            volume.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Link ID", yaxis_title="Simulated daily traffic")
            plots.append(dcc.Graph(figure=volume, config={"displayModeBar": False}))
        else:
            plots.append(html.Div("No network maps or network volumes available."))
        return plots

    def population_content():
        blocks = []
        if persons is not None and "age" in persons.columns:
            age = persons["age"].dropna()
            if len(age):
                age_fig = px.histogram(age, nbins=25, title="Age distribution", template="plotly_white")
                blocks.append(dcc.Graph(figure=age_fig, config={"displayModeBar": False}))

        if trips is not None and not trips.empty:
            mode_cols = [c for c in ["mode", "main_mode", "trip_mode"] if c in trips.columns]
            if mode_cols:
                trip_mode = trips[mode_cols[0]].dropna().astype(str)
                if len(trip_mode):
                    counts = trip_mode.value_counts().reset_index()
                    counts.columns = ["mode", "count"]
                    fig = px.bar(counts, x="mode", y="count", color="mode", color_discrete_map=MODE_COLORS, template="plotly_white", title="Trips by mode")
                    blocks.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

        return blocks or [html.Div("No population or trip data available.")]

    def mode_content():
        plots = []
        if mode_df.empty:
            return [html.Div("No mode-share data available.")]

        line_fig = px.line(
            mode_df,
            x="iteration",
            y="share",
            color="mode",
            color_discrete_map=MODE_COLORS,
            title="Mode share evolution",
            template="plotly_white",
        )
        line_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis_tickformat="%")
        plots.append(dcc.Graph(figure=line_fig, config={"displayModeBar": False}))

        final_modes = mode_df[mode_df["iteration"] == mode_df["iteration"].max()].copy()
        bar_fig = px.bar(
            final_modes,
            x="mode",
            y="share",
            color="mode",
            color_discrete_map=MODE_COLORS,
            title="Final mode shares",
            template="plotly_white",
        )
        bar_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), yaxis_tickformat="%")
        plots.append(dcc.Graph(figure=bar_fig, config={"displayModeBar": False}))
        return plots

    app = dash.Dash(__name__, title="Simulation dashboard")
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Simulation dashboard", style={"margin": 0, "fontSize": "2.1rem"}),
                    html.Div(f"Source: {root}", style={"color": "#4a5a6a", "marginTop": "6px"}),
                ],
                style={
                    "padding": "22px 28px",
                    "background": "linear-gradient(135deg, #14213d 0%, #1d4e89 100%)",
                    "color": "white",
                    "borderRadius": "16px",
                    "boxShadow": "0 10px 30px rgba(20, 33, 61, 0.18)",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(item["title"], style={"fontSize": "0.75rem", "letterSpacing": "0.08em", "textTransform": "uppercase", "color": "#5d7085"}),
                            html.Div(item["value"], style={"fontSize": "2rem", "fontWeight": "700", "marginTop": "12px"}),
                        ],
                        style={
                            "background": "white",
                            "borderRadius": "14px",
                            "padding": "20px 18px",
                            "borderTop": f"4px solid {item['color']}",
                            "boxShadow": "0 8px 24px rgba(15, 23, 42, 0.05)",
                        },
                    )
                    for item in cards
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "16px", "marginBottom": "20px"},
            ),
            dcc.Tabs(
                value="overview",
                children=[
                    dcc.Tab(label="Overview", value="overview", children=overview_content()),
                    dcc.Tab(label="Counts & validation", value="counts", children=counts_content()),
                    dcc.Tab(label="Network volumes", value="network", children=network_content()),
                    dcc.Tab(label="Population & trips", value="population", children=population_content()),
                    dcc.Tab(label="Mode choice", value="mode", children=mode_content()),
                ],
                style={"marginTop": "10px"},
            ),
        ],
        style={"padding": "20px 18px 40px", "background": "#edf3f8", "minHeight": "100vh"},
    )
    return app


def build_dashboard(simulation_root):
    root = _find_simulation_root(simulation_root)
    dashboard_dir = root / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    html_path = dashboard_dir / "dashboard.html"
    cards = _build_cards(root)
    cards_html = "".join(
        f'<div class="card"><div class="muted">{item["title"]}</div><div style="font-size:2rem;font-weight:700;margin-top:10px;">{item["value"]}</div></div>'
        for item in cards
    )
    html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8" />
            <title>Simulation dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #edf3f8; margin: 0; padding: 32px; }}
                .panel {{ max-width: 1100px; margin: 0 auto; background: white; border-radius: 16px; padding: 24px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); }}
                .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0 30px; }}
                .card {{ flex: 1 1 180px; background: white; border-radius: 12px; padding: 18px; border-top: 5px solid #1f77b4; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
                .muted {{ color: #5c6b7c; }}
            </style>
        </head>
        <body>
            <div class="panel">
                <h1>Simulation dashboard</h1>
                <div class="muted">{root}</div>
                <div class="cards">
                    {cards_html}
                </div>
                <p>Use the interactive dashboard from the package entry point:</p>
                <pre>python -m analysis.dashboard.app --directory {root}</pre>
            </div>
        </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)
