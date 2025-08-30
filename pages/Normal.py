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
    path="/normal",
    name="Normal",
    title="Normal",
    order=1
)

number_of_data = dbc.InputGroup([
    dbc.Label("Cantidad de datos:", html_for="data_input_n"),
    dbc.Input(
        id="data_input_n_normal",
        type="number",
        min=1,
        max=1_000_000,
        step=1,
        value=10_000,
        required=True,
        className="mb-2"
    )
], className="")

media = dbc.InputGroup([
    dbc.Label("Media: "),
    dbc.Input(
    id="data_input_media",
    type="number",
    step="any",
    value=0,
    required=True,
    className="mb-2"
)])

desv_estandar = dbc.InputGroup([
    dbc.Label("Desviación estandar: "),
    dbc.Input(
    id="data_input_desv_estandar",
    type="number",
    step="any",
    min=0,
    value=0,
    required=True,
    className="mb-2"
)])

msj_error = html.P(id="mensaje_error", className="text-danger")
div = html.Div(id="table")

layout = html.Div([
    html.H1(
        "Distribución normal",
        className="text-center pb-2 bg-primary text-white rounded-pill w-50 mx-auto d-block"
    ),
    dcc.Slider(
        id="bins-slider",
        min=5, max=25, step=5,
        value=15,
        marks={i: str(i) for i in range(5, 26, 5)},
        className="mb-3"
    ),
    number_of_data,
    media,
    desv_estandar,
    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")

@callback(
    Output("table", "children", allow_duplicate=True),
    Output("mensaje_error", "children", allow_duplicate=True),
    Output("histograma", "figure", allow_duplicate=True),
    Input("bins-slider", "value"),
    Input("data_input_media", "value"),
    Input("data_input_desv_estandar", "value"),
    Input("data_input_n_normal", "value"),
    prevent_initial_call=True
)
def update_histogram(bins, media, desv_estandar, n):
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

    if media is None or desv_estandar is None:
        return empty_table, "", empty_fig

    #if desv_estandar == 0:
       # msj = "La desviación debe ser mayor que cero"
       # return empty_table, msj, empty_fig

    # Datos y figura
    valores = GeneradorDeDistribuciones.generar_normal(media, desv_estandar, n)
    data = pd.DataFrame({"valores": valores})
    fig = px.histogram(
        data,
        x="valores",
        nbins=bins,
        title=f"Histograma de la dist. normal, con {bins} intervalo, media {media}, desviación estandar {desv_estandar}"
              f" y n={n:,} datos",
    )

    # Ojo: no pises el nombre de la función importada
    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins)

    # ORDEN correcto: (tabla, mensaje, figura)
    return tabla_comp, "", fig
