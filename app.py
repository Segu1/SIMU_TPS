from dash import Dash, callback, Output, Input, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

# Fijamos semilla (opcional, para reproducibilidad)
np.random.seed(42)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
)

div_selector = dbc.Select(
    id="distribution_selector",
    options=[
        {"label": "Normal", "value": "normal"},
        {"label": "Uniforme", "value": "uniforme"},
    ],
    value="normal",
    className="mb-2",
)

number_of_data = dbc.Input(
    id="data_input",           # <— id simple para usarlo en el callback
    type="number",
    min=1,
    max=1000000,
    step=1,
    value=100000,             # valor por defecto útil
    required=True,
    className="mb-2"
)

app.layout = html.Div([
    html.H1("Histograma interactivo con callback"),
    dcc.Slider(
        id="bins-slider",
        min=5, max=25, step=5,
        value=15,                                 # <— que esté dentro del rango
        marks={i: str(i) for i in range(5, 26, 5)},
        className="mb-3"
    ),
    number_of_data,
    div_selector,
    dcc.Graph(id="histograma")
], className="p-3")

@callback(
    Output("histograma", "figure"),
    Input("bins-slider", "value"),
    Input("distribution_selector", "value"),
    Input("data_input", "value"),
)
def update_histogram(bins, selector, n):
    # intentar convertir n a int si es string
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0  # valor por defecto si es None o inválido

    if n < 1:
        return px.histogram(pd.DataFrame({"valores": []}), x="valores", nbins=bins)

    if selector == "normal":
        valores = np.random.randn(n)
    else:  # uniforme
        valores = np.random.uniform(0, 1, n)

    data = pd.DataFrame({"valores": valores})

    fig = px.histogram(
        data,
        x="valores",
        nbins=bins,
        title=f"Histograma ({selector}) con {bins} bins y n={n:,}",
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True)


