import dash
from dash import callback, Output, Input, State, html, dcc, no_update, ctx  # NEW: State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import GeneradorDeDistribuciones
import GenerarTabla

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
])

A = dbc.InputGroup([
    dbc.Label("A: "),
    dbc.Input(
        id="data_input_A",
        type="number",
        step="any",
        value=0,
        required=True,
        className="mb-2"
    )
])

B = dbc.InputGroup([
    dbc.Label("B: "),
    dbc.Input(
        id="data_input_B",
        type="number",
        step="any",
        value=1,   # SUGERENCIA: 1 para que A < B por defecto
        required=True,
        className="mb-2"
    )
])

msj_error = html.P(id="mensaje_error", className="text-danger")
div = html.Div(id="table")

layout = html.Div([
    html.H1(
        "Distribución uniforme",
        className="text-center pb-2 bg-primary text-white rounded-pill w-50 mx-auto d-block"
    ),
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

    # NEW: botón de descarga centrado
    html.Div(
        dbc.Button("Descargar CSV", id="btn_download", color="success", className="rounded-pill px-4"),
        className="text-center my-2"
    ),

    dcc.Download(id="download_csv"),

    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")


@callback(
    Output("download_csv", "data"),
    Output("table", "children"),
    Output("mensaje_error", "children"),
    Output("histograma", "figure"),
    Input("bins-slider", "value"),
    Input("data_input_A", "value"),
    Input("data_input_B", "value"),
    Input("data_input_n", "value"),
    Input("btn_download", "n_clicks"),     # NEW: botón como Input
)
def update_histogram(bins, a, b, n, n_clicks):
    # Normalizar n
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0

    empty_fig = go.Figure()
    empty_table = []
    download = no_update  # por defecto NO descargamos

    # Validaciones
    if n < 1:
        msj = "La cantidad de valores debe ser mayor o igual que 1"
        return download, empty_table, msj, empty_fig

    if a is None or b is None:
        return download, empty_table, "", empty_fig

    if a >= b:
        msj = "B debe ser mayor que A"
        return download, empty_table, msj, empty_fig

    # Datos y figura
    valores = GeneradorDeDistribuciones.generar_uniforme(a, b, n)
    data = pd.DataFrame({"valores": valores})

    xmin = data["valores"].min()
    xmax = data["valores"].max()
    bin_size = (xmax - xmin) / bins if bins and bins > 0 else (xmax - xmin or 1)

    fig = px.histogram(
        data,
        x="valores",
        title=f"Histograma de la dist. uniforme, con {bins} bins y n={n:,}",
    )
    fig.update_traces(
        xbins=dict(start=xmin, end=xmax, size=bin_size),
        marker=dict(line=dict(color="black", width=1))
    )

    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins)

    # NEW: si el trigger fue el botón, generamos la descarga
    if ctx.triggered_id == "btn_download":
        download = dcc.send_data_frame(data.to_csv, "uniforme_datos.csv", index=False)

    # Orden de Outputs: (download, tabla, mensaje, figura)
    return download, tabla_comp, "", fig
