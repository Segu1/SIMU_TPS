import pandas as pd
import numpy as np
import dash_ag_grid as dag

def tabla_frecuencia(datos, bins=None, ordenar="asc", decimales=4):
    """
    Genera una tabla de frecuencias para un vector de datos.
    Trunca VISUALMENTE los extremos de los intervalos a `decimales` sin alterar el binning.
    """

    def _trunc(x, n):
        # truncado (no redondeo) a n decimales, preserva signo
        factor = 10 ** n
        return np.trunc(x * factor) / factor

    def _fmt_interval(iv: pd.Interval, n: int) -> str:
        # arma etiqueta con truncado visual según el tipo de cerrado
        left_br = "[" if iv.closed in ("left", "both") else "("
        right_br = "]" if iv.closed in ("right", "both") else ")"
        l = _trunc(iv.left, n)
        r = _trunc(iv.right, n)
        # usar formato fijo con n decimales
        return f"{left_br}{l:.{n}f}, {r:.{n}f}{right_br}"

    s = pd.Series(datos).dropna()
    n = len(s)

    if n == 0:
        return dag.AgGrid(
            columnDefs=[{"field": "Mensaje"}],
            rowData=[{"Mensaje": "No hay datos"}],
            columnSize="sizeToFit",
            dashGridOptions={"domLayout": "autoHeight"},
            style={"height": None, 'width': '100%', 'margin': 'auto'},
        )

    # Bin/categorización
    if bins is not None:
        if isinstance(bins, int):
            categorias = pd.cut(s, bins=bins, include_lowest=True, right=True)
        else:
            categorias = pd.cut(s, bins=bins, include_lowest=True, right=True)

        frec = categorias.value_counts(sort=False)

        # etiquetas SOLO visuales con truncado
        indice = [ _fmt_interval(iv, decimales) for iv in frec.index ]
        header_valor = "Intervalo"
    else:
        frec = s.value_counts()
        if ordenar == "asc":
            frec = frec.sort_index()
        elif ordenar == "desc":
            frec = frec.sort_index(ascending=False)
        indice = frec.index.astype(str).tolist()
        header_valor = "Valor"

    fr = frec / n
    porc = fr * 100.0
    fac = frec.cumsum()
    pac = porc.cumsum()

    df = pd.DataFrame({
        header_valor: indice,
        "Frecuencia": frec.values,
        "Frecuencia Relativa": fr.values,
        "Porcentaje": porc.values,
        "Frecuencia Acumulada": fac.values,
        "Porcentaje Acumulado": pac.values
    })

    columnas = [
        {'field': header_valor, 'headerName': header_valor, 'width': 220},
        {'field': 'Frecuencia', 'headerName': 'Frecuencia', 'width': 120,
         'valueFormatter': {"function": """d3.format(",d")(params.value)"""}
        },
        {'field': 'Frecuencia Relativa', 'headerName': 'Frec. Relativa', 'width': 140,
         'valueFormatter': {"function": f"""d3.format(",.{decimales}f")(params.value)"""}
        },
        {'field': 'Porcentaje', 'headerName': '%', 'width': 110,
         'valueFormatter': {"function": f"""d3.format(",.{decimales}f")(params.value)"""}
        },
        {'field': 'Frecuencia Acumulada', 'headerName': 'Frec. Acum.', 'width': 130,
         'valueFormatter': {"function": """d3.format(",d")(params.value)"""}
        },
        {'field': 'Porcentaje Acumulado', 'headerName': '% Acum.', 'width': 120,
         'valueFormatter': {"function": f"""d3.format(",.{decimales}f")(params.value)"""}
        },
    ]

    tabla = dag.AgGrid(
        columnDefs=columnas,
        rowData=df.to_dict("records"),
        columnSize="sizeToFit",
        dashGridOptions={"domLayout": "autoHeight"},
        style={"height": None, 'width': '100%', 'margin': 'auto'},
        defaultColDef={"resizable": False, "sortable": False, 'suppressMovable': True,
                       "wrapHeaderText": True, "autoHeaderHeight": True}
    )
    return tabla
