import dash
from dash import callback, Output, Input, html, dcc, no_update, ctx, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
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
        value=1,   # A < B por defecto
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
        marks={i: str(i) for i in range(5, 26, 5)},  # 5..25
        className="mb-3"
    ),
    number_of_data,
    A,
    B,

    html.Div(
        dbc.Button("Descargar CSV", id="btn_download", color="success", className="rounded-pill px-4"),
        className="text-center my-2"
    ),
    dcc.Download(id="download_csv_uniforme"),
    dcc.Store(id="store_uniforme_df"),  # 👈 cache del último dataset

    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")


@callback(
    Output("download_csv_uniforme", "data"),
    Output("table", "children", allow_duplicate=True),
    Output("mensaje_error", "children", allow_duplicate=True),
    Output("histograma", "figure", allow_duplicate=True),
    Output("store_uniforme_df", "data"),   # 👈 guardamos df aquí
    Input("bins-slider", "value"),
    Input("data_input_A", "value"),
    Input("data_input_B", "value"),
    Input("data_input_n", "value"),
    Input("btn_download", "n_clicks"),
    State("store_uniforme_df", "data"),    # 👈 leemos df cacheado
    prevent_initial_call=True
)
def update_histogram(bins, a, b, n, n_clicks, cached_df):
    empty_fig = go.Figure()
    empty_table = []
    download = no_update
    store_out = no_update

    # Si el trigger es el botón de descarga → NO recalcular
    if ctx.triggered_id == "btn_download" and n_clicks:
        if cached_df is None:
            return no_update, no_update, "No hay datos generados para descargar.", no_update, no_update
        df_cached = pd.DataFrame(cached_df)
        return dcc.send_data_frame(
            df_cached.to_csv,
            "uniforme_datos.csv",
            index=False, sep=";", decimal=",", encoding="utf-8-sig"
        ), no_update, "", no_update, no_update

    # --- Validaciones (cuando cambian sliders/inputs) ---
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        return no_update, empty_table, "La cantidad de valores debe ser mayor o igual que 1", empty_fig, None
    if a is None or b is None:
        return no_update, empty_table, "", empty_fig, None
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        return no_update, empty_table, "A y B deben ser numéricos", empty_fig, None
    if a >= b:
        return no_update, empty_table, "B debe ser mayor que A", empty_fig, None

    try:
        B = int(bins)
    except (TypeError, ValueError):
        B = 15
    B = max(5, min(25, B))

    # --- Datos (recalcular) ---
    try:
        valores = GeneradorDeDistribuciones.generar_uniforme(a, b, n)
    except Exception:
        valores = np.random.uniform(low=a, high=b, size=n)

    df = pd.DataFrame({"valores": valores})
    s = df["valores"].dropna()

    # --- Bordes muestrales (último cerrado) ---
    xmin = float(s.min())
    xmax = float(s.max())
    if xmin == xmax:
        eps = np.finfo(float).eps
        edges = np.array([xmin, np.nextafter(xmin + eps, np.inf)], dtype=float)
        B = 1
    else:
        edges = np.linspace(xmin, xmax, B + 1, dtype=float)
        edges[-1] = np.nextafter(edges[-1], np.inf)  # cerrar último

    # --- Tabla (mismos bordes) ---
    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins=edges)

    # --- Histograma con hover consistente ---
    counts, _ = np.histogram(s.to_numpy(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths  = np.diff(edges)

    display_rights = edges[1:].copy()
    display_rights[-1] = xmax  # mostrar límite real en el último

    hover_text = [
        f"Intervalo: [{l:.4f}, {r:.4f}{']' if i == len(counts) - 1 else ')'}<br>"
        f"Frecuencia: {int(c):,d}"
        for i, (l, r, c) in enumerate(zip(edges[:-1], display_rights, counts))
    ]

    fig = go.Figure()
    fig.add_bar(
        x=centers, y=counts, width=widths,
        name="Frecuencia", text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    )
    fig.update_layout(
        title=f"Uniforme muestral [{xmin}, {xmax}] con {B} bins y n={n:,}",
        xaxis_title="valores", yaxis_title="Frecuencia",
    )

    # Guardar df en store para futuras descargas
    store_out = df.to_dict(orient="list")

    return no_update, tabla_comp, "", fig, store_out
