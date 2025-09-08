import dash
from dash import callback, Output, Input, html, dcc, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import GeneradorDeDistribuciones
import GenerarTabla

dash.register_page(
    __name__,
    path="/exponencial",
    name="Exponencial",
    title="Exponencial",
    order=1
)

number_of_data = dbc.InputGroup([
    dbc.Label("Cantidad de datos:", html_for="data_input_n"),
    dbc.Input(
        id="data_input_n_exponencial",
        type="number",
        min=1,
        max=1_000_000,
        step=1,
        value=10_000,
        required=True,
        className="mb-2"
    )
], className="")

lambdan = dbc.InputGroup([
    dbc.Label("Lambda: "),
    dbc.Input(
        id="data_input_lambdan_exponencial",
        type="number",
        step="any",
        value=0,
        required=True,
        className="mb-2"
    )
])

msj_error = html.P(id="mensaje_error_exponencial", className="text-danger")
div = html.Div(id="table_exponencial")

layout = html.Div([
    html.H1(
        "Distribución exponencial",
        className="text-center pb-2 bg-primary text-white rounded-pill w-50 mx-auto d-block"
    ),
    dcc.Slider(
        id="bins-slider-exponencial",
        min=5,
        max=25,
        step=5,
        value=15,
        marks={i: str(i) for i in range(5, 26, 5)},
        className="mb-3"
    ),
    number_of_data,
    lambdan,
    html.Div(
    dbc.Button("Descargar datos", id="btn_download_exponencial_combinado",
               color="primary", className="rounded-pill px-4"),
    className="text-center my-2"
        ),
    dcc.Download(id="download_csv_exponencial_combinado"),
    
    msj_error,
    dcc.Graph(id="histograma_exponencial", className="inline-block"),
    div
], className="p-3")


from dash import ctx  # si no lo tenés ya importado al inicio

@callback(
    Output("download_csv_exponencial_combinado", "data"),
    Output("table_exponencial", "children", allow_duplicate=True),
    Output("mensaje_error_exponencial", "children", allow_duplicate=True),
    Output("histograma_exponencial", "figure", allow_duplicate=True),
    Input("bins-slider-exponencial", "value"),
    Input("data_input_lambdan_exponencial", "value"),
    Input("data_input_n_exponencial", "value"),
    Input("btn_download_exponencial_combinado", "n_clicks"),
    prevent_initial_call=True
)
def update_histogram_exponencial(bins, lambdan, n, n_clicks):
    import numpy as np
    empty_fig = go.Figure()
    empty_table = []
    download = no_update

    # Validaciones
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        return download, empty_table, "La cantidad de valores debe ser mayor o igual que 1", empty_fig
    if lambdan is None or lambdan <= 0:
        return download, empty_table, "Lambda debe ser mayor que 0", empty_fig

    # Datos
    try:
        valores = GeneradorDeDistribuciones.generar_exponencial(lambdan, n)
    except Exception:
        valores = np.random.exponential(scale=1.0 / lambdan, size=n)

    df = pd.DataFrame({"valores": valores})
    s = df["valores"].dropna().to_numpy()

    # Bordes de bins (último cerrado)
    xmin = float(np.min(s)); xmax = float(np.max(s))
    if xmin == xmax:
        eps = np.finfo(float).eps
        edges = np.array([xmin, np.nextafter(xmin + eps, np.inf)], dtype=float)
        bins = 1
    else:
        xmax_closed = np.nextafter(xmax, np.inf)
        edges = np.linspace(xmin, xmax_closed, int(bins) + 1, dtype=float)

    # Conteo para figura
    counts, _ = np.histogram(s, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    display_rights = edges[1:].copy()
    display_rights[-1] = xmax
    hover_text = [
        f"Intervalo: [{l:.4f}, {r:.4f}{']' if i == len(counts) - 1 else ')'}<br>Frecuencia: {int(c):,d}"
        for i, (l, r, c) in enumerate(zip(edges[:-1], display_rights, counts))
    ]

    fig = go.Figure()
    fig.add_bar(x=centers, y=counts, width=widths, name="Frecuencia",
                text=hover_text, hovertemplate="%{text}<extra></extra>")
    fig.update_layout(
        title=(f"Histograma de la dist. exponencial — {bins} intervalos, λ={lambdan}, n={n:,}"),
        xaxis_title="valores", yaxis_title="Frecuencia"
    )

    # Tabla de frecuencias (grid + DF)
    tabla_comp, df_tabla = GenerarTabla.tabla_frecuencia(valores, bins=edges, decimales=4, return_df=True)

    # Ajuste estético: último límite superior = xmax real
    df_tabla = df_tabla.copy()
    if len(df_tabla) > 0:
        df_tabla.loc[df_tabla.index[-1], "Limite superior )"] = round(xmax, 4)

    # Descarga combinada
    if ctx.triggered_id == "btn_download_exponencial_combinado" and n_clicks:
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
            "exponencial_datos.csv",
            index=False, sep=";", decimal=",", encoding="utf-8-sig",
            float_format="%.4f", na_rep=""
        )

    return download, tabla_comp, "", fig
