import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from original_app import *

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Parameterized graph function
def create_graph(canton_name):
    return generate_html(canton_name)

tab_params = {
    'tab-1': {'canton_name': 'Zürich'},
    'tab-2': {'canton_name': 'Basel-Stadt'},
    'tab-3': {'canton_name': 'St. Gallen'},
    'tab-4': {'canton_name': 'Bern'},
    'tab-5': {'canton_name': 'Fribourg'},
    'tab-6': {'canton_name': 'Vaud'},
    'tab-7': {'canton_name': 'Ticino'},
    'tab-8': {'canton_name': 'Aargau'},
    'tab-9': {'canton_name': 'Genève'},
    'tab-10': {'canton_name': 'Solothurn'},
    'tab-11': {'canton_name': 'Jura'},
    'tab-12': {'canton_name': 'Valais'},
    'tab-13': {'canton_name': 'Luzern'},
    'tab-14': {'canton_name': 'Basel-Landschaft'},
    'tab-15': {'canton_name': 'Neuchâtel'},
    'tab-16': {'canton_name': 'Thurgau'},
    'tab-17': {'canton_name': 'Uri'},
    'tab-18': {'canton_name': 'Schwyz'},
    'tab-19': {'canton_name': 'Nidwalden'},
    'tab-20': {'canton_name': 'Glarus'},
    'tab-21': {'canton_name': 'Graubünden'},
    'tab-22': {'canton_name': 'Schaffhausen'},
    'tab-23': {'canton_name': 'Zug'},
    'tab-24': {'canton_name': 'Obwalden'},
    'tab-25': {'canton_name': 'Appenzell Ausserrhoden'},
    'tab-26': {'canton_name': 'Appenzell Innerrhoden'}
}

canton_list = [
    'Zürich', 'Basel-Stadt', 'St. Gallen', 'Bern', 'Fribourg', 'Vaud', 
    'Ticino', 'Aargau', 'Genève', 'Solothurn', 'Jura', 'Valais', 
    'Luzern', 'Basel-Landschaft', 'Neuchâtel', 'Thurgau', 'Uri', 
    'Schwyz', 'Nidwalden', 'Glarus', 'Graubünden', 'Schaffhausen', 
    'Zug', 'Obwalden', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden'
]

app.layout = dbc.Container([
    html.H2("Comparison of Microcensus and Synthetic Data (Canton)", className="mt-4"),
    dbc.Tabs(
        [dbc.Tab(label=canton, tab_id=f"tab-{i+1}") 
         for i, canton in enumerate(canton_list)],
        id="tabs-example",
        active_tab="tab-1",
        class_name="mb-4"
    ),
    
    html.Div(id="tabs-content-example")
], fluid=True)

@app.callback(
    Output("tabs-content-example", "children"),
    Input("tabs-example", "active_tab")
)
def render_content(active_tab):
    params = tab_params.get(active_tab, {})
    children = create_graph(**params)
    return html.Div(
        children=children
    )

if __name__ == "__main__":
    app.run(debug=True)
