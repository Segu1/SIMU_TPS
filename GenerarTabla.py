import pandas as pd
import dash_ag_grid as dag

def tabla_frecuencia(datos, bins=None, ordenar="asc", decimales=4):
    """
    Genera una tabla de frecuencias para un vector de datos.
    - datos: list/np.array/pd.Series
    - bins:  None → categorías exactas
             int  → cantidad de bins (para numéricos)
             list/np.array → bordes de bins (ej: [0, .2, .4, .6, .8, 1])
    - ordenar: 'asc' | 'desc' (solo aplica cuando bins es None)
    - decimales: cantidad de decimales para las proporciones
    """
    s = pd.Series(datos).dropna()
    n = len(s)

    if n == 0:
        # grid vacío y un mensaje de ayuda
        return dag.AgGrid(
            columnDefs=[{"field": "Mensaje"}],
            rowData=[{"Mensaje": "No hay datos"}],
            columnSize="sizeToFit",
            dashGridOptions={"domLayout": "autoHeight"},
            style={"height": None, 'width': '100%', 'margin': 'auto'},
        )

    # Si hay bins → hacemos intervals tipo [a,b)
    if bins is not None:
        if isinstance(bins, int):
            # pd.cut crea intervalos contiguos (por defecto right=True: (a,b])
            # para estilo [a,b) usamos right=False
            categorias = pd.cut(s, bins=bins, include_lowest=True,right=True)
        else:
            categorias = pd.cut(s, bins=bins, include_lowest=True, right=True)
        frec = categorias.value_counts(sort=False)
        indice = frec.index.astype(str)  # interval -> string
    else:
        # Categórico/numérico sin binning
        frec = s.value_counts()
        if ordenar == "asc":
            frec = frec.sort_index()
        elif ordenar == "desc":
            frec = frec.sort_index(ascending=False)
        indice = frec.index.astype(str)

    fr = frec / n
    porc = fr * 100.0
    fac = frec.cumsum()
    pac = porc.cumsum()

    df = pd.DataFrame({
        "Valor/Bin": indice,
        "Frecuencia": frec.values,
        "Frecuencia Relativa": fr.values,
        "Porcentaje": porc.values,
        "Frecuencia Acumulada": fac.values,
        "Porcentaje Acumulado": pac.values
    })

    # Armamos columnas para AG Grid (con formateadores d3)
    columnas = [
        {'field': 'Valor/Bin', 'headerName': 'Valor / Intervalo', 'width': 220},
        {'field': 'Frecuencia', 'headerName': 'Frecuencia', 'width': 120,
         'valueFormatter': {"function": """d3.format(",d")(params.value)"""}
        },
        {'field': 'Frecuencia Relativa', 'headerName': 'Frec. Relativa', 'width': 140,
         'valueFormatter': {"function": f"""d3.format(",.{decimales}f")(params.value)"""}
        },
        {'field': 'Porcentaje', 'headerName': '%', 'width': 100,
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
