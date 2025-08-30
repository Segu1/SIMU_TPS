import pandas as pd
import numpy as np
import dash_ag_grid as dag

def tabla_frecuencia(datos, bins=None, ordenar="asc", decimales=4):
    """
    Genera una tabla de frecuencias para un vector de datos.

    - Sin truncado en etiquetas de intervalos (se muestran tal cual, usando str()).
    - Si `bins` es int: se construyen bordes con np.linspace(xmin, xmax, bins+1),
      respetando exactamente xmin, xmax y amplitud uniforme.
    - Intervalos tipo [izq, der) con include_lowest=True para incluir el mínimo,
      y el máximo cae en el último intervalo porque el borde final es exactamente xmax.
    """

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

    # --- Bin/categorización ---
    if bins is not None:
        if isinstance(bins, int):
            xmin = s.min()
            xmax = s.max()

            if xmin == xmax:
                categorias = pd.Series([pd.Interval(left=xmin, right=xmax, closed="both")] * n, index=s.index)
            else:
                edges = np.linspace(xmin, xmax, bins + 1)
                # ⚠️ mover SOLO el último borde para incluir xmax
                edges[-1] = np.nextafter(edges[-1], np.inf)
                categorias = pd.cut(s, bins=edges, include_lowest=True, right=False)

        else:
            edges = np.array(bins, dtype=float)
            # ⚠️ también en este caso: mover SOLO el último borde
            edges[-1] = np.nextafter(edges[-1], np.inf)
            categorias = pd.cut(s, bins=edges, include_lowest=True, right=False)

        frec = categorias.value_counts(sort=False)

        # Etiquetas sin truncado/round: usamos str() del Interval
        indice = [str(iv) for iv in frec.index]
        header_valor = "Intervalo"

    else:
        # Categórico/numérico sin binning: valores exactos
        frec = s.value_counts()
        if ordenar == "asc":
            frec = frec.sort_index()
        elif ordenar == "desc":
            frec = frec.sort_index(ascending=False)
        indice = frec.index.astype(str).tolist()
        header_valor = "Valor"

    # --- Métricas ---
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

    # --- Definición de columnas para AG Grid ---
    columnas = [
        {'field': header_valor, 'headerName': header_valor, 'width': 260},
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
        {'field': 'Porcentaje Acumulado', 'headerName': '% Acum.', 'width': 130,
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
