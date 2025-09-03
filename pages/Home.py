# pages/home.py
import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/",
    name="Inicio",
    title="Simuladores — Inicio",
    order = 1)

def layout():
    return html.Div([
        dbc.Container([
            # Hero
            dbc.Row([
                dbc.Col([
                    html.H1("Simuladores de Distribuciones", className="display-6 fw-bold mb-2"),
                    html.P(
                        "Explorá distribuciones aleatorias, generá muestras y visualizá histogramas con parámetros interactivos.",
                        className="lead text-muted mb-3"
                    ),
                    dbc.ButtonGroup([
                        dbc.Button("Uniforme", href="/distribuciones/uniforme/", color="primary"),
                        dbc.Button("Normal", href="/distribuciones/normal", color="secondary", outline=True),
                        dbc.Button("Exponencial", href="/distribuciones/exponencial", color="secondary", outline=True),
                    ])
                ], md=8),
            ], className="py-4"),

            html.Hr(className="my-4"),

            # Quick cards
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Uniforme", class_name="fw-semibold"),
                    dbc.CardBody([
                        html.P("Valores equiprobables en un intervalo [a, b]."),
                        dbc.Button("Ir a Uniforme", href="/distribuciones/uniforme/", color="primary", size="sm")
                    ])
                ], class_name="shadow-sm h-100"), md=4),

                dbc.Col(dbc.Card([
                    dbc.CardHeader("Normal", class_name="fw-semibold"),
                    dbc.CardBody([
                        html.P("Distribución gaussiana parametrizada por μ y σ."),
                        dbc.Button("Ir a Normal", href="/distribuciones/normal", color="primary", size="sm")
                    ])
                ], class_name="shadow-sm h-100"), md=4),

                dbc.Col(dbc.Card([
                    dbc.CardHeader("Exponencial", class_name="fw-semibold"),
                    dbc.CardBody([
                        html.P("Modelo de tiempo entre eventos con parámetro λ."),
                        dbc.Button("Ir a Exponencial", href="/distribuciones/exponencial", color="primary", size="sm")
                    ])
                ], class_name="shadow-sm h-100"), md=4),
            ], className="g-3"),

            html.Hr(className="my-4"),

            # How to use
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Cómo usar", class_name="fw-semibold"),
                    dbc.CardBody([
                        html.Ul([
                            html.Li("Elegí una distribución en el menú o con los botones de arriba."),
                            html.Li("Ajustá los parámetros y la cantidad de datos."),
                            html.Li("Observá histograma, tabla y métricas en tiempo real.")
                        ], className="mb-0")
                    ])
                ], class_name="shadow-sm"), md=7),

                dbc.Col(dbc.Card([
                    dbc.CardHeader("Atajos rápidos", class_name="fw-semibold"),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem(html.Span([
                                "Uniforme → ", html.Code("/distribuciones/uniforme/")
                            ])),
                            dbc.ListGroupItem(html.Span([
                                "Normal → ", html.Code("/distribuciones/normal")
                            ])),
                            dbc.ListGroupItem(html.Span([
                                "Exponencial → ", html.Code("/distribuciones/exponencial")
                            ])),
                        ], flush=True)
                    ])
                ], class_name="shadow-sm"), md=5),
            ], className="g-3 mb-4"),

        ], style={'maxWidth': '1200px', 'minWidth': '1200px'})
    ])
