import dash_bootstrap_components as dbc
from dash import html

def navbar() -> dbc.Navbar:
    barra_de_navegacion = dbc.Navbar(
        dbc.Container([
            # Logo o Título
            dbc.NavbarBrand("Distribuciones", class_name="fw-bold fs-4 text-primary"),

            # Menú de navegación
            dbc.Nav([
                dbc.NavItem(
                    dbc.NavLink("Uniforme", href="/distribuciones/uniforme/", active="exact"),
                ),
                dbc.NavItem(
                    dbc.NavLink("Normal", href="/distribuciones/normal", active="exact"),
                ),
                dbc.NavItem(
                    dbc.NavLink("Exponencial", href="/distribuciones/exponencial", active="exact"),
                ),
            ], class_name="ms-auto gap-3")
        ], style={'maxWidth': '1200px', 'minWidth': '1200px'}),

        color="light",   # fondo claro
        dark=False,      # texto oscuro
        sticky="top",
        class_name="shadow-sm mb-3 py-2 border-bottom"
    )

    return barra_de_navegacion
