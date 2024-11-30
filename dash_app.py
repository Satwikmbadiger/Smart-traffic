import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("Charts and Graphs"),
        dcc.Graph(
            id='My-Graph',
            figure={
                'data': [
                    {'x': ["Car", "Bike", "Truck"], 'y': [18, 16, 12], 'type': 'bar', 'name': 'Bar Chart'},
                    {'x': ["Car", "Bike", "Truck"], 'y': [18, 16, 12], 'type': 'line', 'name': 'Line Chart'}
                ],
                'layout': {
                    'title': 'Insights',
                    'xaxis': {'title': 'Category'},
                    'yaxis': {'title': 'Count'}
                }
            }
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
