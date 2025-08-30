import dash
from dash import callback, Output, Input, html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import GeneradorDeDistribuciones
import GenerarTabla
import plotly.express as px
import pandas as pd

# 👇 REGISTRO CORRECTO DE LA PÁGINA (sin el prefijo global)
dash.register_page(
    __name__,
    path="/uniforme",
    name="Uniforme",
    title="Uniforme",
    order=1
)

number_of_data = dbc.InputGroup([
    dbc.Label("Cantidad de datos:", html_for="data_input_n"),
    dbc.Input(
        id="data_input_n",
        type="number",
        min=1,
        max=1_000_000,
        step=1,
        value=100_000,
        required=True,
        className="mb-2"
    )
], className="")

A = dbc.InputGroup([
    dbc.Label("A: "),
    dbc.Input(
    id="data_input_A",
    type="number",
    step="any",
    value=0,
    required=True,
    className="mb-2"
)])

B = dbc.InputGroup([
    dbc.Label("B: "),
    dbc.Input(
    id="data_input_B",
    type="number",
    step="any",
    value=0,
    required=True,
    className="mb-2"
)])

msj_error = html.P(id="mensaje_error", className="text-danger")
div = html.Div(id="table")


layout = html.Div([
    html.H1("Distribución uniforme"),
    dcc.Slider(
        id="bins-slider",
        min=5, max=25, step=5,
        value=15,
        marks={i: str(i) for i in range(5, 25, 5)},
        className="mb-3"
    ),
    number_of_data,
    A,
    B,
    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")

@callback(
    Output("table", "children"),
    Output("mensaje_error", "children"),
    Output("histograma", "figure"),
    Input("bins-slider", "value"),
    Input("data_input_A", "value"),
    Input("data_input_B", "value"),
    Input("data_input_n", "value"),
)
def update_histogram(bins, a, b, n):
    # Normalizamos n
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0

    empty_fig = go.Figure()  # figura vacía válida
    empty_table = []         # sin hijos en el div "table"

    if n < 1:
        msj = "La cantidad de valores debe ser mayor o igual que 1"
        return empty_table, msj, empty_fig

    if a is None or b is None:
        return empty_table, "", empty_fig

    if a >= b:
        msj = "B debe ser mayor que A"
        return empty_table, msj, empty_fig

    # Datos y figura
    valores = GeneradorDeDistribuciones.generar_uniforme(a, b, n)
    data = pd.DataFrame({"valores": valores})
    fig = px.histogram(
        data,
        x="valores",
        nbins=bins,
        title=f"Histograma de la dist. uniforme, con {bins} bins y n={n:,}",
    )

    # Ojo: no pises el nombre de la función importada
    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins)

    # ORDEN correcto: (tabla, mensaje, figura)
    return tabla_comp, "", fig
