import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from component import Navbar

app = Dash(
    __name__,
    title="TP 2",
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    use_pages=True,
    pages_folder="pages",
    requests_pathname_prefix="/distribuciones/",
    routes_pathname_prefix="/distribuciones/",
    suppress_callback_exceptions=True
)

# 👇 Permite callbacks duplicados entre páginas con allow_duplicate
app.config.prevent_initial_callbacks = "initial_duplicate"

server = app.server  # útil para gunicorn/uwsgi

app.layout = html.Div([
    Navbar.navbar(),
    dash.page_container  # renderiza la página activa
])

if __name__ == "__main__":
    app.run(debug=True, port=8067)


