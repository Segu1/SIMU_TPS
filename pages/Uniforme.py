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
        dbc.Button("Descargar CSV", id="btn_download", color="success", className="rounded-pill px-4"),
        className="text-center my-2"
    ),
    dcc.Download(id="download_csv_uniforme"),

    msj_error,
    dcc.Graph(id="histograma", className="inline-block"),
    div
], className="p-3")


@callback(
    Output("download_csv_uniforme", "data"),
    Output("table", "children"),
    Output("mensaje_error", "children"),
    Output("histograma", "figure"),
    Input("bins-slider", "value"),
    Input("data_input_A", "value"),
    Input("data_input_B", "value"),
    Input("data_input_n", "value"),
    Input("btn_download", "n_clicks"),
    prevent_initial_call=False  # Render inicial
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
        return download, empty_table, "La cantidad de valores debe ser mayor o igual que 1", empty_fig
    if a is None or b is None:
        return download, empty_table, "", empty_fig
    if a >= b:
        return download, empty_table, "B debe ser mayor que A", empty_fig
    # 1) Datos
    valores = GeneradorDeDistribuciones.generar_uniforme(a, b, n)
    df = pd.DataFrame({"valores": valores})
    s = df["valores"].dropna()

    # 2) Bordes muestrales
    xmin = float(s.min())
    xmax = float(s.max())

    if xmin == xmax:
        eps = np.finfo(float).eps
        edges = np.array([xmin, np.nextafter(xmin + eps, np.inf)], dtype=float)
        effective_bins = 1
    else:
        effective_bins = int(bins)
        edges = np.linspace(xmin, xmax, effective_bins + 1, dtype=float)
        # abrimos SOLO el último borde para la tabla/corte
        edges[-1] = np.nextafter(edges[-1], np.inf)
        print("Edges: ", edges)

    # 3) Tabla (misma malla y cierre)
    tabla_comp = GenerarTabla.tabla_frecuencia(valores, bins=edges)

    # 4) Histograma (alinear 'end' para evitar recorte interno de Plotly)
    bin_size = edges[1] - edges[0]
    print("Bin size: ", bin_size)
    start = edges[0]
    # ¡OJO! No uses edges[-1] directo porque ya lo abriste con nextafter
    # y puede no ser múltiplo exacto de bin_size. En su lugar:
    end_aligned = start + (bin_size * effective_bins)
    end_aligned = np.nextafter(end_aligned, np.inf)  # ahora sí, abrí apenas el final
    print('End aligned: ', end_aligned)
    print('Max: ', max(valores))
    print('Min:', min(valores))
    end_inclusive = np.nextafter(end_aligned, np.inf)

    fig = px.histogram(
        df,
        x="valores",
        title=f"Uniforme muestral [{xmin}, {xmax}] con {effective_bins} bins y n={n:,}",
    )
    fig.update_traces(
        xbins=dict(start=start, end=end_aligned + np_inclusive, size=bin_size),
        histfunc="count"
    )

    # 5) Descarga solo si clickeaste el botón
    if ctx.triggered_id == "btn_download" and n_clicks:
        download = dcc.send_data_frame(df.to_csv, "uniforme_datos.csv", index=False)

    return download, tabla_comp, "", fig

