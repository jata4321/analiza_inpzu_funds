import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [3, 1, 2, 3]
})

fig = px.line(df, x="x", y="y")

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
