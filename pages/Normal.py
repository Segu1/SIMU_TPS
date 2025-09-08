import dash
from dash import callback, Output, Input, html, dcc, no_update, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import GeneradorDeDistribuciones
import GenerarTabla
import pandas as pd
import numpy as np
from dash import ctx

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
    )
])

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
    )
])

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
    html.Div(
        dbc.Button("Descargar datos", id="btn_download_normal_combinado",
               color="primary", className="rounded-pill px-4"),
    className="text-center my-2"
        ),
    dcc.Download(id="download_csv_normal_combinado"),
    
    dcc.Store(id="store_normal_df"),  # 👈 cache local del último dataset
    
    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")

@callback(
    Output("download_csv_normal_combinado", "data"),
    Output("table", "children", allow_duplicate=True),
    Output("mensaje_error", "children", allow_duplicate=True),
    Output("histograma", "figure", allow_duplicate=True),
    Output("store_normal_df", "data"),           # 👈 guardamos df acá
    Input("bins-slider", "value"),
    Input("data_input_media", "value"),
    Input("data_input_desv_estandar", "value"),
    Input("data_input_n_normal", "value"),
    Input("btn_download_normal_combinado", "n_clicks"),
    State("store_normal_df", "data"), 
    prevent_initial_call=True
)
def update_histogram(bins, mu, sigma, n, n_clicks):
    import numpy as np

    empty_fig = go.Figure()
    empty_table = []
    download = no_update

        store_out = no_update

    trigger = dash.ctx.triggered_id

    # --- Si el trigger es descargar: NO recalcular, usar cache ---
    if trigger == "btn_download_normal" and n_clicks:
        if cached_df is None:
            return no_update, no_update, "No hay datos generados para descargar.", no_update, no_update
        df_cached = pd.DataFrame(cached_df)
        return dcc.send_data_frame(
            df_cached.to_csv,
            "normal_datos.csv",
            index=False, sep=";", decimal=",", encoding="utf-8-sig"
        ), no_update, "", no_update, no_update

    # --- Validación de entradas (cuando cambian sliders/inputs) ---
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        return download, empty_table, "La cantidad de valores debe ser mayor o igual que 1", empty_fig

    if mu is None or sigma is None:
        return no_update, empty_table, "", empty_fig, None
    try:
        mu = float(mu); sigma = float(sigma)
    except (TypeError, ValueError):
        return no_update, empty_table, "Media y desviación deben ser numéricos", empty_fig, None
    if sigma <= 0:
        return no_update, empty_table, "La desviación estándar debe ser mayor que 0", empty_fig, None

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
    tabla_comp, df_tabla = GenerarTabla.tabla_frecuencia(valores, bins=edges, decimales=4, return_df=True)
    
    # Ajuste estético: último límite superior = xmax real
    df_tabla = df_tabla.copy()
    if len(df_tabla) > 0:
        df_tabla.loc[df_tabla.index[-1], "Limite superior )"] = round(xmax, 4)

    # --- Histogram bars with hover (interval + frequency) ---
    counts, _ = np.histogram(s.to_numpy(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths  = np.diff(edges)

    display_rights = edges[1:].copy()
    display_rights[-1] = xmax
    hover_text = [
        f"Intervalo: [{l:.4f}, {r:.4f}{']' if i == len(counts) - 1 else ')'}<br>"
        f"Frecuencia: {int(c):,d}"
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
    # Descarga combinada
    if ctx.triggered_id == "btn_download_normal_combinado" and n_clicks:
        n_datos, n_tabla = len(df), len(df_tabla)
        n_max = max(n_datos, n_tabla)

        def pad(col, length):
            lst = list(col)
            lst += [np.nan] * (length - len(lst))
            return lst

        comb = pd.DataFrame({
            "Datos": pad(df["valores"], n_max),
            "": [""] * n_max,
            "Intervalo": pad(df_tabla.get("Intervalo", []), n_max),
            "Limite inferior [": pad(df_tabla.get("Limite inferior [", []), n_max),
            "Limite superior )": pad(df_tabla.get("Limite superior )", []), n_max),
            "Frecuencia observada": pad(df_tabla.get("Frecuencia observada", []), n_max),
        })

        download = dcc.send_data_frame(
            comb.to_csv,
            "normal_datos.csv",
            index=False, sep=";", decimal=",", encoding="utf-8-sig",
            float_format="%.4f", na_rep=""
        )

    return download, tabla_comp, "", fig, store_out