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

import numpy as np

def truncar(x, decimales=4):
    factor = 10.0 ** decimales
    return np.trunc(x * factor) / factor

@callback(
    Output("download_csv_normal", "data"),
    Output("table", "children", allow_duplicate=True),
    Output("mensaje_error", "children", allow_duplicate=True),
    Output("histograma", "figure", allow_duplicate=True),
    Input("bins-slider", "value"),
    Input("data_input_media", "value"),
    Input("data_input_desv_estandar", "value"),
    Input("data_input_n_normal", "value"),
    Input("btn_download_normal", "n_clicks"),
    prevent_initial_call=True
)
def update_histogram(bins, mu, sigma, n, n_clicks):
    import numpy as np

    empty_fig = go.Figure()
    empty_table = []
    download = no_update

    # --- Validate inputs ---
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        return download, empty_table, "La cantidad de valores debe ser mayor o igual que 1", empty_fig

    if mu is None or sigma is None:
        return download, empty_table, "", empty_fig
    try:
        mu = float(mu); sigma = float(sigma)
    except (TypeError, ValueError):
        return download, empty_table, "Media y desviación deben ser numéricos", empty_fig
    if sigma <= 0:
        return download, empty_table, "La desviación estándar debe ser mayor que 0", empty_fig

    try:
        B = int(bins)
    except (TypeError, ValueError):
        B = 15
    B = max(5, min(25, B))

    # --- Data ---
    try:
        valores = GeneradorDeDistribuciones.generar_normal(mu, sigma, n)
    except Exception:
        valores = np.random.normal(loc=mu, scale=sigma, size=n)

    #print("Primeros 10 valores generados:", valores[:10])
    #print("Media real:", np.mean(valores))
    #print("Desv. estándar real:", np.std(valores))

    df = pd.DataFrame({"valores": valores})
    s = df["valores"].dropna()

    # --- Bin edges with last bin closed (include xmax) ---
    xmin = float(s.min()); xmax = float(s.max())

    if xmin == xmax:
        eps = np.finfo(float).eps
        edges = np.array([xmin, np.nextafter(xmin + eps, np.inf)], dtype=float)
        B = 1
    else:
        edges = np.linspace(xmin, xmax, B + 1, dtype=float)
        # tiny nudge so the last edge is strictly greater (numpy.histogram closes last bin)
        edges[-1] = np.nextafter(edges[-1], np.inf)

    # --- Frequency table using same edges (visual truncation to 4 decimals) ---
    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins=edges, decimales=4)

    # --- Histogram bars with hover (interval + frequency) ---
    counts, _ = np.histogram(s.to_numpy(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths  = np.diff(edges)

    # show real xmax (not nextafter) in hover for last bin
    display_rights = edges[1:].copy()
    display_rights[-1] = xmax

    hover_text = [
        f"Intervalo: [{l:.4f}, {r:.4f}{']' if i == len(counts) - 1 else ')'}<br>"
        f"Frecuencia: {c:,d}"
        for i, (l, r, c) in enumerate(zip(edges[:-1], display_rights, counts))
    ]

    fig = go.Figure()
    fig.add_bar(
        x=centers,
        y=counts,
        width=widths,
        name="Frecuencia",
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    )
    fig.update_layout(
        title=(f"Histograma de la dist. normal — {B} intervalos, "
               f"media={mu}, σ={sigma}, n={n:,}"),
        xaxis_title="valores",
        yaxis_title="Frecuencia"
    )
    # --- Descarga CSV ---
    if dash.ctx.triggered_id == "btn_download_normal" and n_clicks:
        download = dcc.send_data_frame(
            df.to_csv,
            "normal_datos.csv",
            index=False,
            sep=";",        # ; para que Excel en español abra bien
            decimal=",",    # coma decimal
            encoding="utf-8-sig"
        )



    return download, tabla_comp, "", fig