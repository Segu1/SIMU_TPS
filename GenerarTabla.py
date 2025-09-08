import pandas as pd
import numpy as np
import dash_ag_grid as dag

def tabla_frecuencia(datos, bins=None, ordenar="asc", decimales=4, return_df=False):
    """
    Tabla de frecuencias con columnas:
    - Intervalo (1..k)
    - Limite inferior [ (cerrado)
    - Limite superior ) (abierto)
    - Frecuencia observada

    Si bins es int: usa bordes uniformes entre min y max con intervalos [izq, der).
    Ajusta SOLO el último borde con nextafter para incluir xmax en el último intervalo.
    Si bins es secuencia: se usa como bordes explícitos (también con ese ajuste del último).
    """

    s = pd.Series(datos).dropna()
    n = len(s)

    if n == 0:
        grid_vacio = dag.AgGrid(
            columnDefs=[{"field": "Mensaje"}],
            rowData=[{"Mensaje": "No hay datos"}],
            columnSize="sizeToFit",
            dashGridOptions={"domLayout": "autoHeight"},
            style={"height": None, 'width': '100%', 'margin': 'auto'},
        )
        return (grid_vacio, pd.DataFrame()) if return_df else grid_vacio

    # --- Bin/categorización ---
    if bins is not None:
        if isinstance(bins, int):
            xmin = s.min()
            xmax = s.max()

            if xmin == xmax:
                categorias = pd.Series(
                    [pd.Interval(left=xmin, right=xmax, closed="both")] * n,
                    index=s.index
                )
                bordes = np.array([xmin, xmax], dtype=float)
            else:
                bordes = np.linspace(xmin, xmax, bins + 1)
                # mover SOLO el último borde para incluir xmax
                bordes[-1] = np.nextafter(bordes[-1], np.inf)
                categorias = pd.cut(s, bins=bordes, include_lowest=True, right=False, precision=8)
        else:
            bordes = np.array(bins, dtype=float)
            # mover SOLO el último borde para incluir el máximo real
            bordes[-1] = np.nextafter(bordes[-1], np.inf)
            categorias = pd.cut(s, bins=bordes, include_lowest=True, right=False)

        frec = categorias.value_counts(sort=False)
        intervals = frec.index.to_list()

        limites_inf = [iv.left for iv in intervals]
        limites_sup = [iv.right for iv in intervals]
        num_intervalo = list(range(1, len(frec) + 1))

        df = pd.DataFrame({
            "Intervalo": num_intervalo,
            "Limite inferior [": limites_inf,
            "Limite superior )": limites_sup,
            "Frecuencia observada": frec.values.astype(int),
        })

        if decimales is not None:
            df["Limite inferior ["] = np.round(df["Limite inferior ["], decimales)
            df["Limite superior )"] = np.round(df["Limite superior )"], decimales)

        columnas = [
            {"field": "Intervalo", "headerName": "Intervalo", "width": 110,
             "valueFormatter": {"function": "d3.format(',d')(params.value)"}},
            {"field": "Limite inferior [", "headerName": "Limite inferior [", "width": 160},
            {"field": "Limite superior )", "headerName": "Limite superior )", "width": 160},
            {"field": "Frecuencia observada", "headerName": "Frecuencia observada", "width": 180,
             "valueFormatter": {"function": "d3.format(',d')(params.value)"}},
        ]

    else:
        # Sin binning: valores exactos
        frec = s.value_counts()
        if ordenar == "asc":
            frec = frec.sort_index()
        elif ordenar == "desc":
            frec = frec.sort_index(ascending=False)

        df = pd.DataFrame({
            "Valor": frec.index.astype(str),
            "Frecuencia observada": frec.values.astype(int),
        })

        columnas = [
            {"field": "Valor", "headerName": "Valor", "width": 220},
            {"field": "Frecuencia observada", "headerName": "Frecuencia observada", "width": 180,
             "valueFormatter": {"function": "d3.format(',d')(params.value)"}},
        ]

    tabla = dag.AgGrid(
        columnDefs=columnas,
        rowData=df.to_dict("records"),
        columnSize="sizeToFit",
        dashGridOptions={"domLayout": "autoHeight"},
        style={"height": None, "width": "100%", "margin": "auto"},
        defaultColDef={
            "resizable": False,
            "sortable": False,
            "suppressMovable": True,
            "wrapHeaderText": True,
            "autoHeaderHeight": True
        }
    )

    return (tabla, df) if return_df else tabla
