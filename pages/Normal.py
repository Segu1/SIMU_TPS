import dash
from dash import callback, Output, Input, html, dcc, no_update
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
    # NEW: botón de descarga centrado
    html.Div(
        dbc.Button("Descargar CSV", id="btn_download_normal", color="success", className="rounded-pill px-4"),
        className="text-center my-2"
    ),
    dcc.Download(id="download_csv_normal"),
    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")

@callback(
    Output("download_csv_normal", "data"),
    Output("table", "children", allow_duplicate=True),
    Output("mensaje_error", "children", allow_duplicate=True),
    Output("histograma", "figure", allow_duplicate=True),
    Input("bins-slider", "value"),
    Input("data_input_media", "value"),
    Input("data_input_desv_estandar", "value"),
    Input("data_input_n_normal", "value"),
    Input("btn_download_normal", "n_clicks"),   # <-- ESCUCHÁ EL BOTÓN
    prevent_initial_call=True           # opcional: render inicial
)
def update_histogram(bins, media, desv_estandar, n, n_clicks):
    # Normalizar n
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0

    empty_fig = go.Figure()
    empty_table = []
    download = no_update  # por defecto NO descargamos

    # Validaciones (devolvé SIEMPRE 4 valores en el mismo orden)
    if n < 1:
        msj = "La cantidad de valores debe ser mayor o igual que 1"
        return download, empty_table, msj, empty_fig

    if media is None or desv_estandar is None:
        return download, empty_table, "", empty_fig

    if desv_estandar <= 0:
        msj = "La desviación estándar debe ser mayor que 0"
        return download, empty_table, msj, empty_fig

    # Datos y figura
    valores = GeneradorDeDistribuciones.generar_normal(media, desv_estandar, n)
    data = pd.DataFrame({"valores": valores})

    fig = px.histogram(
        data,
        x="valores",
        nbins=bins,
        title=f"Histograma de la dist. normal, con {bins} intervalos, media {media}, "
              f"desviación estándar {desv_estandar} y n={n:,} datos",
    )

    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins)

    # Disparar descarga SOLO si el trigger fue el botón
    if dash.ctx.triggered_id == "btn_download_normal" and n_clicks:
        download = dcc.send_data_frame(data.to_csv, "normal_datos.csv", index=False)

    return download, tabla_comp, "", fig
