import dash
from dash import callback, Output, Input, html, dcc, no_update, ctx
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
        marks={i: str(i) for i in range(5, 25, 5)},
        className="mb-3"
    ),
    number_of_data,
    A,
    B,

    html.Div(
    dbc.Button("Descargar datos", id="btn_download_combinado",
               color="primary", className="rounded-pill px-4"),
    className="text-center my-2"
    ),
    dcc.Download(id="download_csv_uniforme_combinado"),

    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")

@callback(
    Output("download_csv_uniforme_combinado", "data"),
    Output("table", "children"),
    Output("mensaje_error", "children"),
    Output("histograma", "figure"),
    Input("bins-slider", "value"),
    Input("data_input_A", "value"),
    Input("data_input_B", "value"),
    Input("data_input_n", "value"),
    Input("btn_download_combinado", "n_clicks"),
    prevent_initial_call=False
)
def update_histogram(bins, a, b, n, n_clicks_comb):
    # Normalizar n
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0

    empty_fig = go.Figure()
    empty_table = []
    download_combinado = no_update

    # Validaciones
    if n < 1:
        return download_combinado, empty_table, "La cantidad de valores debe ser mayor o igual que 1", empty_fig
    if a is None or b is None:
        return download_combinado, empty_table, "", empty_fig
    if a >= b:
        return download_combinado, empty_table, "B debe ser mayor que A", empty_fig

    # 1) Datos
    valores = GeneradorDeDistribuciones.generar_uniforme(a, b, n)
    df = pd.DataFrame({"valores": valores})
    s = df["valores"].dropna()

    # 2) Bordes muestrales (misma malla para histo y tabla)
    xmin = float(s.min())
    xmax = float(s.max())

    if xmin == xmax:
        eps = np.finfo(float).eps
        edges = np.array([xmin, np.nextafter(xmin + eps, np.inf)], dtype=float)
        effective_bins = 1
    else:
        effective_bins = int(bins)
        edges = np.linspace(xmin, xmax, effective_bins + 1, dtype=float)
        edges[-1] = np.nextafter(edges[-1], np.inf)

    # 3) Tabla de frecuencias (grid + df para exportar)
    tabla_comp, df_tabla = GenerarTabla.tabla_frecuencia(valores, bins=edges, return_df=True)

    # Opcional estético: el último límite superior que se vea como xmax real
    df_tabla = df_tabla.copy()
    if len(df_tabla) > 0:
        df_tabla.loc[df_tabla.index[-1], "Limite superior )"] = round(xmax, 4)

    # 4) Histograma
    counts, _ = np.histogram(s.to_numpy(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    display_rights = edges[1:].copy()
    display_rights[-1] = xmax  # último cerrado a derecha

    hover_text = [
        f"Intervalo: [{l:.4f}, {r:.4f}{']' if i == len(counts) - 1 else ')'}<br>Frecuencia: {c:,d}"
        for i, (l, r, c) in enumerate(zip(edges[:-1], display_rights, counts))
    ]

    fig = go.Figure()
    fig.add_bar(
        x=centers, y=counts, width=widths, name="Frecuencia",
        text=hover_text, hovertemplate="%{text}<extra></extra>",
    )
    fig.update_layout(
        title=f"Uniforme muestral [{xmin}, {xmax}] con {effective_bins} bins y n={n:,}",
        xaxis_title="valores", yaxis_title="Frecuencia",
    )

    # 5) Exportar combinado (Datos | col vacía | Tabla)
    if ctx.triggered_id == "btn_download_combinado" and n_clicks_comb:
        n_datos = len(df)
        n_tabla = len(df_tabla)
        n_max = max(n_datos, n_tabla)

        # Padding helper
        def pad(col, length):
            lst = list(col)
            lst += [np.nan] * (length - len(lst))
            return lst

        col_datos = pad(df["valores"], n_max)
        col_blanco = [""] * n_max

        comb = pd.DataFrame({
            "Datos": col_datos,
            "": col_blanco,  # separador
            "Intervalo": pad(df_tabla.get("Intervalo", []), n_max),
            "Limite inferior [": pad(df_tabla.get("Limite inferior [", []), n_max),
            "Limite superior )": pad(df_tabla.get("Limite superior )", []), n_max),
            "Frecuencia observada": pad(df_tabla.get("Frecuencia observada", []), n_max),
        })

        download_combinado = dcc.send_data_frame(
            comb.to_csv,
            "uniforme_datos.csv",
            index=False,
            sep=";", decimal=",", encoding="utf-8-sig",
            float_format="%.4f",
            na_rep=""  # celdas vacías limpias
        )

    return download_combinado, tabla_comp, "", fig


